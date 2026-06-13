# Supported Configurations

RetroAGI pins its Python and machine-learning stack so checkpoints and training
runs use a reproducible tensor ABI.

## Version Matrix

| Component | Supported version | Status |
| --- | --- | --- |
| Operating system | Linux x86-64 | Primary development and container platform |
| Python | 3.12.x | Required by `pyproject.toml` |
| PyTorch | 2.9.1 | Pinned runtime dependency |
| torchvision | 0.24.1 | Pinned to the matching PyTorch release |
| CPU | PyTorch 2.9.1 CPU wheel | Supported baseline for tests, environments, and training |
| CUDA 12.8 | PyTorch `cu128` wheel | Primary GPU target; wheel selection is verified, GPU execution is pending |
| CUDA 13.0 | PyTorch `cu130` wheel | Supported container target; Docker and GPU verification are pending |

Python 3.11 and earlier, Python 3.13 and later, ROCm, Apple Metal, and CUDA
versions other than those listed above are not currently supported. They may
work, but are outside the tested project matrix.

CUDA is optional. The synthetic, Block SMB, Full SMB, vision, and test code must
remain functional with the CPU-only PyTorch wheel. GPU acceleration is used
only when PyTorch reports that CUDA is available.

## Installation

Create and activate a Python 3.12 virtual environment before selecting one
PyTorch wheel variant.

### CPU-only

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
```

### CUDA 12.8

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test]'
```

### CUDA 13.0

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[test]'
```

The PyTorch wheel supplies its CUDA runtime libraries. A local CUDA toolkit is
only required when compiling CUDA extensions; an NVIDIA driver compatible with
the selected wheel is still required.

Verify the selected runtime with:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The wheel variants above follow the official
[PyTorch 2.9.1 installation matrix](https://pytorch.org/get-started/previous-versions/#v291).
