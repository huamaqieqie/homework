# Folder Structure

- `le-wm/`: source code, configs, assets, and project README.
- `report_materials/`: report figures, metrics, summary text, and main experiment logs.
- `report_materials_heads8/`: heads8-specific report figures, metrics, summaries, and logs.
- `logs/`: root-level run logs collected by type.
  - `logs/training/`: training logs that were previously in the repository root.
  - `logs/evaluation/`: evaluation logs that were previously in the repository root.
  - `logs/setup/`: setup or installation logs.
- `archive/`: non-primary files kept for recovery/reference.
  - `archive/backups/`: `.bak` files and other manual backups.
  - `archive/duplicate_logs/`: logs verified as duplicates of files already stored under `report_materials/`.
- `env_cache.sh`: environment cache script kept at the root for easy sourcing.
