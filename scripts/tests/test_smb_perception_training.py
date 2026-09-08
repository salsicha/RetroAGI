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
        assert [event["step"] for event in events] == [1, 2]
        assert all(np.isfinite(event["loss"]) for event in events)
        assert metrics["frames"] == 2
        assert (tmp_path / "model" / "perception.pth").is_file()
    finally:
        torch.use_deterministic_algorithms(strict, warn_only=warn_only)
