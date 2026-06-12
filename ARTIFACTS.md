# Artifact Storage

RetroAGI keeps source code and configuration in normal Git. Large binary and
generated training artifacts are stored with Git LFS.

## Git LFS Files

The repository's `.gitattributes` routes these artifact formats through LFS:

- Model weights: `*.pth`, `*.pt`
- Generated datasets: `*.npz`, `*.npy`
- Archives: `*.zip`, `*.tar.gz`, `*.gz`, `*.tgz`, `*.tar.xz`
- Generated images and notebooks: `*.png`, `*.jpg`, `*.gif`, `*.ipynb`

Install Git LFS before cloning or pulling artifacts:

```bash
git lfs install
git lfs pull
```

To clone code without downloading large artifacts immediately:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <repository-url>
cd RetroAGI
git lfs pull --include="data/**,scripts/segmentation/*.pth"
```

Use `git lfs ls-files -s` to inspect tracked artifacts and their sizes.

## Generated Output

`graphify-out/` is local generated analysis output and is intentionally ignored.
Regenerate it with the graphify workflow instead of committing its cache,
reports, or rendered graphs.

Training scripts write their outputs under `data/`. When adding a generated
dataset or checkpoint, verify that it is represented by an LFS pointer before
committing:

```bash
git add data/path/to/artifact.pth
git lfs status
git show :data/path/to/artifact.pth
```

The final command should display a small pointer beginning with:

```text
version https://git-lfs.github.com/spec/v1
```

Do not commit virtual environments, caches, or temporary experiment output.
