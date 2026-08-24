# VSB Partial-Discharge Foundation

This repository provides the locked, development-only data foundation for the
VSB partial-discharge work. It creates a leakage-audited measurement split and
deterministic raw-window artifacts; it does not implement features or models.

## Install and configure

```powershell
python -m pip install -e ".[dev]"
Copy-Item configs/local.example.json configs/local.json
```

Edit `configs/local.json` so `raw_parquet_path` and `metadata_path` point to
your local legacy inputs. The paths below are examples only; edit them for your
machine. All new outputs belong under `C:/Users/hrfxgfx/Desktop/1112`.

## Development-only workflow

```powershell
vsb-pipeline lock-split `
  --candidate C:/Users/hrfxgfx/Documents/机器学习论文/results_step10_fresh_split/fresh_measurement_split.csv `
  --historical-root C:/Users/hrfxgfx/Desktop/1111 `
  --historical-root C:/Users/hrfxgfx/Documents/机器学习论文/results_step11_group_cv `
  --allow-shrink-holdout `
  --output artifacts/locks/split_lock.json

vsb-pipeline extract-development `
  --config configs/local.json `
  --split-lock artifacts/locks/split_lock.json `
  --receipt artifacts/last_extraction.json

$extraction = Get-Content -Raw artifacts/last_extraction.json | ConvertFrom-Json
vsb-pipeline audit-development `
  --config configs/local.json `
  --split-lock artifacts/locks/split_lock.json `
  --manifest $extraction.manifest_path

python -m pytest -q
```

For a bounded smoke extraction, add `--smoke-test --limit-measurements 3` to
`extract-development`. A smoke manifest verifies only a proper non-empty subset
of development measurements; it is not a full-completion manifest.

Final-holdout extraction and evaluation are intentionally unavailable in this
phase. The audit command accepts only a development manifest and reports JSON;
it exits successfully only when the artifact contract is fully valid.
