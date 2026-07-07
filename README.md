# DTINet

DTINet provides neural-network layers and training tools for classification from
diffusion tensor images. The installable package lives entirely under
`src/dtinet/`; study-specific cohort construction, cluster scripts, and protected
data are intentionally kept outside this public repository.

## Installation

```bash
uv sync
```

The package exposes the training command as `uv run train`.

## Data organization

Place local data under the repository's ignored `data/` directory. A typical
layout is:

```text
data/
├── training/
│   └── ledger.json
├── tensors/
│   ├── scan-001.nii.gz
│   └── scan-002.nii.gz
└── masks/
    ├── scan-001.nii.gz
    └── scan-002.nii.gz
```

`data/` is ignored by Git except for its `.gitkeep`; never commit protected scans,
subject identifiers, generated checkpoints, or real cohort ledgers.

Each ledger record points to files **relative to the configured data root**:

```json
{
  "image_id": "scan-001",
  "subject_id": "subject-001",
  "label": 0,
  "group": "Control",
  "split": "train",
  "tensor_path": "tensors/scan-001.nii.gz",
  "mask_path": "masks/scan-001.nii.gz"
}
```

Absolute paths and paths that escape the data root are rejected. The default data
root is `data/`; override it in YAML or at runtime:

```yaml
data:
  root: data
  ledger: training/ledger.json
```

```bash
uv run train --config configs/dtinetlc.example.yaml \
  --data-root /path/to/local/data
```

## Training

Copy the example configuration to a local, ignored YAML file and adjust it:

```bash
cp configs/dtinetlc.example.yaml configs/local.yaml
uv run train --config configs/local.yaml --validate-only
uv run train --config configs/local.yaml
```

The ledger must assign all scans from a subject to one split. Training writes
checkpoints, history, metrics, and scan- and subject-level predictions under the
configured output directory.
