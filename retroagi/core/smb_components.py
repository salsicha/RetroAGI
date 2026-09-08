"""Versioned, independently replaceable SMB policy components.

A bundle's contracts describe meaning, not merely parameter shapes. Replacing
one component invalidates its gameplay qualification. No weights are fetched
implicitly, and all serialized component payloads contain tensors only.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from retroagi.core.smb_geometry import MOTION_NAMES, STATE_NAMES
from retroagi.core.smb_scene import AVAILABILITY_NAMES

COMPONENT_PREFIXES = {
    "actor": ("agent.", "tactics_network.", "strategy_network.", "world_model_actor_context."),
    "world_model": ("world_model.",),
    "critic": ("critic.",),
    "executor": ("motor_controller.",),
    "auxiliary": ("transition_representation_head.", "reward_head.", "value_head."),
}


@dataclass(frozen=True)
class SMBComponentContract:
    observation_schema: str = "smb_scene_v2"
    physics_profile: str = "nes_land_v1"
    semantic_classes: tuple = (
        "background",
        "mario",
        "platform",
        "coin",
        "goal",
        "enemy",
        "moving_platform",
    )
    hierarchy_lengths: tuple = (8, 16, 64)
    jump_frames: tuple = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32)
    run_actions: tuple = (1, 2)
    wait_duration_scale: float = 1.0
    min_wait_frames: int = 1
    max_wait_frames: int = 32
    viewport: tuple = (256, 240)
    velocity_scales: tuple = (3.0, 8.0)
    physical_features: tuple = STATE_NAMES + MOTION_NAMES
    availability_features: tuple = AVAILABILITY_NAMES
    coordinate_frame: str = "visible_window_pixels"
    world_model_cadence: str = "physical_frame"
    recurrent_reset: str = "episode"
    frame_skip: int = 1
    recurrent_state: bool = False
    decision_mode: str = "greedy"
    scene_encoder: str = "canonical_semantic_v2"

    def __post_init__(self):
        for name in (
            "semantic_classes",
            "hierarchy_lengths",
            "jump_frames",
            "run_actions",
            "viewport",
            "velocity_scales",
            "physical_features",
            "availability_features",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            self.observation_schema != "smb_scene_v2"
            or self.scene_encoder != "canonical_semantic_v2"
        ):
            raise ValueError("Unrecognized canonical scene interface")
        if (
            self.viewport != (256, 240)
            or self.velocity_scales != (3.0, 8.0)
            or self.physical_features != STATE_NAMES + MOTION_NAMES
            or self.availability_features != AVAILABILITY_NAMES
            or self.coordinate_frame != "visible_window_pixels"
            or self.world_model_cadence != "physical_frame"
            or self.recurrent_reset != "episode"
            or self.physics_profile != "nes_land_v1"
            or self.run_actions != (1, 2)
        ):
            raise ValueError("Unsupported feature meanings or physical execution settings")
        if self.hierarchy_lengths != (8, 16, 64) or len(self.jump_frames) != 16:
            raise ValueError("Incompatible hierarchy or primitive shape")
        if self.frame_skip != 1 or self.decision_mode != "greedy":
            raise ValueError("Unsupported component execution contract")
        if any(n < 1 or n > 60 for n in self.jump_frames) or sorted(self.jump_frames) != list(
            self.jump_frames
        ):
            raise ValueError("Physical jump durations must be monotone")

    def manifest(self):
        return asdict(self)


def parameter_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def component_states(model):
    states = {name: {} for name in COMPONENT_PREFIXES}
    for key, value in model.state_dict().items():
        matches = [
            name for name, prefixes in COMPONENT_PREFIXES.items() if key.startswith(prefixes)
        ]
        if len(matches) != 1:
            raise ValueError(f"Unassigned or ambiguous policy component: {key}")
        states[matches[0]][key] = value.detach().cpu().clone()
    return states


def export_bundle(model, directory, *, contract, architecture, perception=None):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("A component bundle must use a fresh directory")
    states = component_states(model)
    directory.mkdir(parents=True)
    manifest = dict(
        version=1,
        contract=contract.manifest(),
        architecture=architecture,
        perception=None,
        components={},
        full_level_qualified=False,
        runtime=(
            getattr(model, "smb_runtime_contract", None).manifest()
            if getattr(model, "smb_runtime_contract", None)
            else None
        ),
    )
    if perception is not None:
        from shutil import copyfile

        from retroagi.core.smb_perception import DenseSMBPerception

        DenseSMBPerception.load(perception)  # validate kind/layout before packaging
        copyfile(perception, directory / "perception.pth")
        manifest["perception"] = dict(
            file="perception.pth",
            sha256=hashlib.sha256((directory / "perception.pth").read_bytes()).hexdigest(),
            interface="smb_scene_v2",
        )
    for name, state in states.items():
        path = directory / f"{name}.pth"
        torch.save(state, path)
        manifest["components"][name] = dict(
            file=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            parameters=parameter_digest(state),
        )
    (directory / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_component(model, directory, name, *, contract, architecture):
    """Replace a compatible component; never carry memory or success claims."""
    if name not in COMPONENT_PREFIXES:
        raise ValueError("Unknown policy component")
    directory = Path(directory)
    manifest = json.loads((directory / "bundle.json").read_text())
    if manifest["version"] != 1 or SMBComponentContract(**manifest["contract"]) != contract:
        raise ValueError("Component observation/physics/runtime contract mismatch")
    if manifest["architecture"] != architecture:
        raise ValueError("Component architecture mismatch")
    entry = manifest["components"][name]
    path = directory / entry["file"]
    if path.resolve().parent != directory.resolve():
        raise ValueError("Component must be inside its bundle")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Component checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = component_states(model)[name]
    if state.keys() != expected.keys() or any(state[k].shape != expected[k].shape for k in state):
        raise ValueError("Component tensor layout mismatch")
    merged = model.state_dict()
    merged.update(state)
    model.load_state_dict(merged, strict=True)
    model.full_level_qualified = False
    model._stance_history = None
    if hasattr(model, "smb_executor"):
        model.smb_executor.reset()
    return entry


def trainable_components(model, names):
    """Select whole components and return a frozen-weight verification baseline."""
    unknown = set(names) - COMPONENT_PREFIXES.keys()
    if unknown:
        raise ValueError(f"Unknown components: {sorted(unknown)}")
    component_states(model)  # fail before changing flags if the partition is incomplete
    prefixes = tuple(p for name in names for p in COMPONENT_PREFIXES[name])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    return {
        name: parameter_digest(state)
        for name, state in component_states(model).items()
        if name not in names
    }


def verify_frozen_components(model, baseline):
    states = component_states(model)
    changed = [
        name for name, digest in baseline.items() if parameter_digest(states[name]) != digest
    ]
    if changed:
        raise AssertionError(f"Frozen component weights changed: {changed}")


def load_bundle(directory, *, device="cpu", perception_path=None):
    """Assemble the shared core and a replaceable canonical perception module."""
    from retroagi.core.smb_learning import make_model
    from retroagi.core.smb_perception import DenseSMBPerception
    from retroagi.core.smb_runtime import attach_runtime

    directory = Path(directory)
    manifest = json.loads((directory / "bundle.json").read_text())
    contract = SMBComponentContract(**manifest["contract"])
    model = make_model(hidden_dim=manifest["architecture"]["hidden_dim"], device=device)
    for name in COMPONENT_PREFIXES:
        load_component(
            model, directory, name, contract=contract, architecture=manifest["architecture"]
        )
    if not manifest.get("runtime"):
        raise ValueError("Playable bundle requires an explicit runtime")
    attach_runtime(model, manifest["runtime"])
    runtime = model.smb_runtime_contract
    if (
        runtime.schema,
        runtime.physics_profile,
        runtime.jump_hold_frames,
        runtime.recurrent_state,
        runtime.wait_duration_scale,
        runtime.min_wait_frames,
        runtime.max_wait_frames,
    ) != (
        contract.observation_schema,
        contract.physics_profile,
        contract.jump_frames,
        contract.recurrent_state,
        contract.wait_duration_scale,
        contract.min_wait_frames,
        contract.max_wait_frames,
    ):
        raise ValueError("Bundle component/runtime contract mismatch")
    entry = manifest.get("perception")
    path = (
        Path(perception_path)
        if perception_path is not None
        else directory / entry["file"] if entry else None
    )
    vision = None
    if path is not None:
        if perception_path is None:
            if (
                path.resolve().parent != directory.resolve()
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise ValueError("Perception checksum/path mismatch")
        vision = DenseSMBPerception.load(path, device=device)
    model.eval()
    return model, vision, manifest
