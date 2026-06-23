# Full SMB Content Setup

This is the supported local content setup for real Full SMB emulator runs. It
is required for headless Full SMB training, evaluation, transfer checks, and
future play commands that use `stable-retro`.

## Supported Game

| Field | Value |
| --- | --- |
| stable-retro game id | `SuperMarioBros-Nes` |
| RetroAGI stage | `full_smb` |
| CLI selector | `retroagi ... --game smb --stage full` |
| Backend entrypoint | `retro.make(game="SuperMarioBros-Nes")` |
| Required package | `python -m pip install -e '.[full-smb]'` |

Only the stable-retro `SuperMarioBros-Nes` integration is supported for Full
SMB. Other SMB ROM revisions, hacks, or emulator integrations need an explicit
new content spec before they can be used for comparable runs.

## Local Files

ROM files are local user-provided content and must stay outside git. Use this
workspace-local layout:

| Path | Purpose | Commit Policy |
| --- | --- | --- |
| `local/full_smb/roms/` | Temporary staging directory for a legally obtained SMB NES ROM before import. | Ignored by git. |
| `local/full_smb/checksums/SuperMarioBros-Nes.sha256` | Local checksum record for the ROM imported into stable-retro. | Keep with local run notes; do not commit ROM content. |
| `artifacts/full_smb/<run>/content.json` | Run metadata copied from the content spec plus checksum filename/hash when preserving a run. | Safe only if it contains metadata and hashes, not ROM bytes. |

Create the local directories:

```bash
mkdir -p local/full_smb/roms local/full_smb/checksums
```

Copy your legally obtained ROM into `local/full_smb/roms/`, then import it into
stable-retro:

```bash
python -m retro.import local/full_smb/roms
```

Record the checksum locally:

```bash
shasum -a 256 local/full_smb/roms/<your-rom-file>.nes \
  > local/full_smb/checksums/SuperMarioBros-Nes.sha256
```

The checksum file should identify the ROM used for a run, but the ROM itself
must not be committed, bundled, uploaded with artifacts, or redistributed.

## Environment Check

`FullSMBStage()` creates the backend lazily through
`make_stable_retro_env(...)`. When `stable-retro` is not installed or the
`SuperMarioBros-Nes` game has not been imported, RetroAGI raises a
`RuntimeError` that includes:

- the failing stable-retro game id,
- the `full-smb` extra install command,
- the local ROM staging directory,
- the `python -m retro.import local/full_smb/roms` import command,
- the SHA-256 checksum record path,
- the legal/provenance reminder.

After importing the ROM, run the headless capability check before training:

```bash
retroagi check-env --game smb --stage full \
  --seed 0 \
  --steps 4 \
  --frame-skip 2 \
  --output artifacts/full_smb/env_check.json
```

The command uses this content spec, then verifies backend import, game
registration, ROM availability, headless reset, render reset, save/load state,
action stepping, frame-skip metadata, and deterministic seeding. It writes a
JSON report and exits nonzero if any required check fails.
