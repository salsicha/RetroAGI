"""CPU worker processes for independent Block SMB episodes, layouts and replays.

Evaluation episodes, evaluation layouts, teacher routes and demonstration
replays are independent of one another, so a pool of single-threaded CPU
workers runs them concurrently and returns results in submission order. A
batch-of-one policy or vision call is faster on the CPU than on the GPU, so
the workers never touch CUDA. Results equal the in-process path up to CPU/GPU
floating-point rounding; with a CPU run they are identical.
"""

import copy
import io
import multiprocessing
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

_WORKER: dict = {}


def _initialize_worker(vision_bytes, deterministic):
    # Before any CUDA call: workers must not create CUDA contexts.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(deterministic)
    vision = torch.load(io.BytesIO(vision_bytes), map_location="cpu", weights_only=False)
    _WORKER.update(vision=vision, policy=None, policy_key=None, policy_version=None)


def worker_vision():
    return _WORKER["vision"]


def worker_policy(config, path, version, *, training=False):
    """The published policy in this worker, loaded once per version.

    Training rollouts act in training mode (dropout on), as in-process
    training rollouts do; evaluation acts in evaluation mode.
    """
    key = pickle.dumps(config)
    if _WORKER["policy_key"] != key:
        from .train import make_block_smb_model

        _WORKER.update(policy=make_block_smb_model(config), policy_key=key, policy_version=None)
    if _WORKER["policy_version"] != version:
        _WORKER["policy"].load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        _WORKER["policy_version"] = version
    return _WORKER["policy"].train(training)


def cpu_copy(module):
    """A CPU copy of a vision encoder or model; other encoders are shared as is."""
    if isinstance(module, torch.nn.Module):
        return copy.deepcopy(module).to("cpu")
    return module


class BlockSMBWorkerPool:
    """Spawned CPU workers holding a copy of the frozen vision encoder."""

    def __init__(self, workers, vision, *, deterministic=True):
        if workers <= 0:
            raise ValueError("workers must be positive")
        buffer = io.BytesIO()
        torch.save(cpu_copy(vision), buffer)
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        self.workers = int(workers)
        self._directory = tempfile.TemporaryDirectory(prefix="retroagi_policy_")
        self._policy_version = 0
        # Evaluation layouts never change within a run; keep each set once.
        self.sample_sets: dict = {}
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(buffer.getvalue(), bool(deterministic)),
        )

    def map(self, fn, items, *, chunksize=1):
        """Results in submission order, yielded as they become available."""
        return self._executor.map(fn, items, chunksize=chunksize)

    def publish_policy(self, model):
        """Save the current weights for the workers; returns (path, version)."""
        self._policy_version += 1
        path = Path(self._directory.name) / f"policy{self._policy_version}.pth"
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)
        previous = path.with_name(f"policy{self._policy_version - 1}.pth")
        if previous.exists():
            previous.unlink()
        return str(path), self._policy_version

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
