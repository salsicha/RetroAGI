# Supported Configurations

RetroAGI pins its Python and machine-learning stack so checkpoints and training
runs use a reproducible tensor ABI.

## Version Matrix

| Component | Supported version | Status |
| --- | --- | --- |
| Operating system | Linux x86-64, macOS Apple Silicon | Linux is the primary container platform; macOS uses native Python/MPS |
| Python | 3.12.x, 3.13.x, 3.14.x | Required by `pyproject.toml` |
| PyTorch | 2.9.1 | Pinned runtime dependency |
| torchvision | 0.24.1 | Pinned to the matching PyTorch release |
| pygame provider | `pygame-ce==2.5.7` | Provides the `pygame` import used by Block SMB |
| stable-retro | `stable-retro==1.0.0` on Python <3.13, pinned upstream source on Python 3.13+ | Source pin is used until a PyPI release carries Python 3.13/3.14 metadata |
| CPU | PyTorch 2.9.1 CPU wheel | Supported baseline for tests, environments, and training |
| CUDA 12.8 | PyTorch `cu128` wheel | Primary GPU target; wheel selection is verified, GPU execution is pending |
| CUDA 13.0 | PyTorch `cu130` wheel | Supported container target; Docker and GPU verification are pending |
| Apple Metal/MPS | PyTorch `mps` backend | Native macOS GPU target |

Python 3.11 and earlier, Python 3.15 and later, ROCm, and CUDA versions other
than those listed above are not currently supported. They may work, but are
outside the tested project matrix.

CUDA is optional. The synthetic, Block SMB, Full SMB, vision, and test code must
remain functional with the CPU-only PyTorch wheel. GPU acceleration is selected
with `retroagi.core.select_device`: `auto` prefers CUDA, then Apple MPS, then
CPU. Explicit `cuda` or `mps` requests fail early if the backend is unavailable.

## Installation

Create and activate a Python 3.12, 3.13, or 3.14 virtual environment before
selecting one PyTorch wheel variant.

### CPU-only

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
```

### CUDA 12.8

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test]'
```

### CUDA 13.0

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[test]'
```

### macOS Apple Silicon

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1
python -m pip install -e '.[test]'
```

The PyTorch wheel supplies its CUDA runtime libraries. A local CUDA toolkit is
only required when compiling CUDA extensions; an NVIDIA driver compatible with
the selected wheel is still required.

For Python 3.13 and 3.14, RetroAGI installs `stable-retro` from pinned upstream
source commit `778186c71e003f7c8a5682187832ba430b8e34b3` because the latest
PyPI release metadata still excludes those Python versions. macOS source builds
may require the Homebrew dependencies listed by the upstream stable-retro macOS
installation guide.

Use `pygame-ce` as the only pygame provider in a virtual environment. If
`pygame` was installed previously, uninstall it before reinstalling RetroAGI so
the `pygame` import resolves to pygame-ce.

Verify the selected runtime with:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.backends.mps.is_available())'
```

The wheel variants above follow the official
[PyTorch 2.9.1 installation matrix](https://pytorch.org/get-started/previous-versions/#v291).
