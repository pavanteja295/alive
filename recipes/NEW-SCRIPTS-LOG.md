# New scripts log

Every creator runs through the same scripts, driven by that creator's profile and
settings. A new script is written only when no existing one covers the step, and is
logged here so the next person knows it exists and why. Profiles, settings files and
build records are not scripts and are not logged.

| date | script | step it covers | why nothing existing covered it |
|---|---|---|---|
| 2026-10-04 | `alive` (from voice_clones `stack`) | start, stop, check the live app for any creator | `stack` was the launcher; it moved here, reads `creators/<id>/live.env`, and gained `check` and `build` |
| 2026-10-04 | `recipes/rebuild.py` | one command from YouTube ids to released models | each recipe's status says what is next for a person; nothing ran the five in order, or replayed a creator's recorded decisions |
| 2026-10-04 | `env/setup_externals.sh` | the tracker, renderer and voice code at our commits | our renderer changes existed on one machine only, and the tracker carried an uncommitted edit; both are now patches this applies |
| 2026-10-05 | `env/install.sh` (and `./alive install`) | every Python environment from the frozen lists, then the shared face assets | the six environments were eight hand-copied commands in env/README.md, with source builds (pytorch3d, CUDA extensions) a fresh machine could not get right by reading |
