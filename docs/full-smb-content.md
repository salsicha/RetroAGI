# Full SMB Content Setup

This is the supported local content setup for real Full SMB emulator runs. It
is required for training the Full SMB vision transformer and for playing Full
SMB with the trained agent, both of which use `stable-retro`.

## Supported Game

| Field | Value |
| --- | --- |
| stable-retro game id | `SuperMarioBros-Nes` |
| RetroAGI stage | `full_smb` |
| Play module | `retroagi/stages/full_smb/play.py` (`FullSMBGame`) |
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

After importing the ROM, check that the emulator starts a level and steps:

```bash
python -c 'from retroagi.stages.full_smb.play import FullSMBGame; g = FullSMBGame("Level1-1"); g.reset(); print(g.step(1)[1]); g.close()'
```

It prints how far into the level Mario is (`level_x`) and whether he is
dying. If stable-retro is missing or the
game has not been imported, `retro.make` raises an error naming the game.
