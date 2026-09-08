"""Production perception loss works under strict CUDA determinism."""

import pytest
import torch
from torch.nn import functional as F

from scripts.smb_perception_training import collision_cross_entropy, train_perception


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_collision_loss_matches_spatial_cpu_value_and_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(72)
    logits = torch.randn(2, 7, 9, 13, generator=generator, requires_grad=True)
    target = torch.randint(7, (2, 9, 13), generator=generator)
    target[:, 2:4, 3:6] = 255
    weight = torch.tensor([0.1, 5.0, 1.0, 4.0, 4.0, 5.0, 3.0])
    reference = F.cross_entropy(logits, target, weight=weight, ignore_index=255)
    reference.backward()
    strict = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        gradients = []
        losses = []
        for _ in range(2):
            candidate = logits.detach().to(device).clone().requires_grad_()
            loss = collision_cross_entropy(candidate, target.to(device), weight=weight.to(device))
            loss.backward()
            losses.append(loss.detach().cpu())
            gradients.append(candidate.grad.cpu())
        torch.testing.assert_close(losses[0], reference.detach())
        torch.testing.assert_close(gradients[0], logits.grad)
        assert torch.equal(losses[0], losses[1])
        assert torch.equal(gradients[0], gradients[1])
    finally:
        torch.use_deterministic_algorithms(strict, warn_only=warn_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_strict_cuda_perception_optimizer_updates(tmp_path):
    import numpy as np

    path = tmp_path / "clip.npz"
    images = np.zeros((2, 240, 256, 3), dtype=np.uint8)
    labels = np.zeros((2, 240, 256), dtype=np.uint8)
    labels[:, 200:] = 2
    labels[:, 188:200, 32:42] = 1
    images[labels == 2] = [100, 80, 50]
    images[labels == 1] = [240, 40, 20]
    np.savez_compressed(path, images=images, labels=labels)
    train = [dict(file=str(path), frames=2, scenario_id="train", split="train")]
    validation = [dict(file=str(path), frames=2, scenario_id="validation", split="validation")]
    events = []
    strict = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        _, metrics = train_perception(
            train,
            validation,
            tmp_path / "model",
            steps=2,
            batch_size=2,
            device="cuda",
            dim=16,
            depth=1,
            log=events.append,
        )
        assert [event["step"] for event in events if event["phase"] == "perception"] == [1, 2]
        assert all(np.isfinite(event["loss"]) for event in events if "loss" in event)
        assert events[-1]["phase"] == "perception_validation"
        assert events[-1]["qualified"] == metrics["qualified"]
        assert metrics["frames"] == 2
        assert (tmp_path / "model" / "perception.pth").is_file()
    finally:
        torch.use_deterministic_algorithms(strict, warn_only=warn_only)


def test_weighted_perception_recovers_probabilities_and_persists_correction(tmp_path):
    from retroagi.core.smb_perception import DenseSMBPerception

    weights = torch.tensor([0.1, 5.0, 1.0, 4.0, 4.0, 5.0, 3.0])
    probability = torch.tensor([0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    weighted = probability * weights
    weighted /= weighted.sum()
    model = DenseSMBPerception(dim=8, depth=1, class_weights=weights.tolist()).eval()
    with torch.no_grad():
        model.pixel_head.weight.zero_()
        model.pixel_head.bias.copy_(weighted.log().repeat(16 * 16))
        frame = torch.zeros(1, 3, 240, 256)
        raw = model(frame, calibrated=False)
        calibrated = model.encode(frame)
    torch.testing.assert_close(raw.semantic_logits.softmax(1)[0, :, 0, 0], weighted)
    torch.testing.assert_close(calibrated.semantic_logits.softmax(1)[0, :, 0, 0], probability)
    assert raw.semantic_ids.unique().tolist() == [1]
    assert calibrated.semantic_ids.unique().tolist() == [0]
    torch.testing.assert_close(calibrated.tokens[0, 0], probability)
    model.save(tmp_path / "perception.pth")
    restored = DenseSMBPerception.load(tmp_path / "perception.pth")
    assert restored.config["class_weights"] == model.config["class_weights"]
    torch.testing.assert_close(restored.encode(frame).semantic_logits, calibrated.semantic_logits)


def test_legacy_perception_defaults_to_uncorrected_logits(tmp_path):
    from retroagi.core.smb_perception import DenseSMBPerception

    model = DenseSMBPerception(dim=8, depth=1).eval()
    model.save(tmp_path / "legacy.pth")
    payload = torch.load(tmp_path / "legacy.pth", weights_only=True)
    del payload["config"]["class_weights"]
    del payload["config"]["refinement_channels"]
    torch.save(payload, tmp_path / "legacy.pth")
    restored = DenseSMBPerception.load(tmp_path / "legacy.pth")
    with torch.no_grad():
        frame = torch.zeros(1, 3, 240, 256)
        torch.testing.assert_close(
            restored.encode(frame).semantic_logits,
            model(frame, calibrated=False).semantic_logits,
        )


@pytest.mark.parametrize("weights", [[1.0] * 6, [0.0] * 7, [-1.0] * 7, [float("nan")] * 7])
def test_invalid_perception_class_weights_rejected(weights):
    from retroagi.core.smb_perception import DenseSMBPerception

    with pytest.raises(ValueError, match="finite positive"):
        DenseSMBPerception(class_weights=weights)


def test_translation_keeps_pixel_label_alignment_and_clips_without_wrapping():
    import numpy as np

    from scripts.smb_perception_training import translated_batch

    class FixedShift:
        def __init__(self):
            self.offsets = iter([3, -2])

        def integers(self, low, high):
            value = next(self.offsets)
            assert low <= value < high
            return value

    labels = np.arange(48, dtype=np.uint8).reshape(1, 6, 8)
    images = np.repeat((labels + 1)[..., None], 3, axis=-1)
    shifted, target = translated_batch(images, labels, FixedShift(), max_x=3, max_y=2)
    np.testing.assert_array_equal(shifted[:, :4, 3:], images[:, 2:, :5])
    np.testing.assert_array_equal(target[:, :4, 3:], labels[:, 2:, :5])
    assert (target[:, 4:] == 255).all()
    assert (target[:, :, :3] == 255).all()
    assert (shifted[:, 4:] == 0).all()
    assert (shifted[:, :, :3] == 0).all()
    assert not np.shares_memory(shifted, images)
    assert not np.shares_memory(target, labels)


def test_refinement_trains_with_vit_and_survives_checkpoint_roundtrip(tmp_path):
    from retroagi.core.smb_perception import DenseSMBPerception

    model = DenseSMBPerception(dim=16, depth=1, refinement_channels=8).eval()
    pixels = torch.rand(1, 3, 240, 256)
    target = torch.zeros(1, 240, 256, dtype=torch.long)
    target[:, 80:92, 231:241] = 1
    output = model(pixels, calibrated=False)
    loss = collision_cross_entropy(output.semantic_logits, target, weight=torch.ones(7))
    loss.backward()
    for parameter in (
        model.backbone.patch_embed.weight,
        model.pixel_head.weight,
        model.refinement[0].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    model.save(tmp_path / "refined.pth")
    restored = DenseSMBPerception.load(tmp_path / "refined.pth")
    assert restored.config["refinement_channels"] == 8
    torch.testing.assert_close(restored.encode(pixels).semantic_logits, output.semantic_logits)
