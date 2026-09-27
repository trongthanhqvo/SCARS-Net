# SCARS-Net

Source-only Pareto-Consistent Reliability Distillation for Physics-Guided UAV
Radio-Frequency Recognition.

**Authors:** Trong Thanh Nguyen, Thi-Thanh-Tan Nguyen, Vu Kien Tran and Le Cuong Nguyen.
Trong Thanh Nguyen, Thi-Thanh-Tan Nguyen and Vu Kien Tran share first authorship;
Le Cuong Nguyen is the corresponding author.

SCARS constructs source-fitted wavelet scattering (W), cyclostationary (C),
energy (E), and STFT (S) representations. SCARS-Net uses independent family
encoders and strict late fusion. PCRD supervises the gate with source-side
physical reliability relations.

The three-dataset runner implements the exploratory `pilot_three_dataset`
protocol. See [Protocol and scope](docs/PROTOCOL.md) for its fixed settings and
differences from the manuscript evaluation. External SOTA implementations and
experimental data are not included.

## Installation

Use Python 3.11. For GPU execution, use Linux with a compatible NVIDIA driver.
The supplied profile uses PyTorch 2.3.1, CUDA 12.1, and NumPy 1.26.4.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
bash setup.sh cuda
```

For CPU-only tests, use `bash setup.sh cpu`. An alternative Conda specification
is provided in `environment.yml`. See [environment notes](docs/ENVIRONMENT.md).
Run all commands from the repository root.

## Installation checks

```bash
python -m pip check
python -m pytest -p no:cacheprovider
python scripts/run_three_dataset_pipeline.py --help
python tools/check_release.py
```

Tests use generated fixtures. Local-data tests require an explicit
`SCARS_LOCAL_DATASET_ROOT`; longer generated-IQ integration tests require
`SCARS_RUN_PILOT_INTEGRATION`. Neither is enabled by default.

A small data-free smoke run:

```bash
python -m scars.cli.run_smoke --output-dir ../scars-smoke --seed 24021
```

## Datasets

Download DroneRFa, DroneRFb-DIR, and DRFF-R2 from their providers.
[Data documentation](docs/DATASETS.md) describes sources, licenses, directory
layouts, IQ keys, labels, and fixed exclusions.

Use the original MAT datasets, not PNG images or exported representation tensors.
Keep datasets and outputs outside the repository.

```bash
export SCARS_DRONERFA_DIR="/path/to/DroneRFa"
export SCARS_DRONERFB_DIR="/path/to/twin_droneRF"
export SCARS_DRFF_R2_DIR="/path/to/DRFF-R2"
export SCARS_RUN_DIR="/path/to/local-ssd/scars-run"
```

## Run the three-dataset pipeline

Preflight, representation selection, and source training:

```bash
python scripts/run_three_dataset_pipeline.py \
  --dronerfa-dir "$SCARS_DRONERFA_DIR" \
  --dronerfb-dir "$SCARS_DRONERFB_DIR" \
  --drff-r2-dir "$SCARS_DRFF_R2_DIR" \
  --output-dir "$SCARS_RUN_DIR" \
  --device cuda \
  --max-windows-per-recording 16 \
  --stop-after train
```

Use `--stop-after preflight` or `--stop-after freeze` to stop earlier.
Preflight hashes dataset files, including held-file integrity information.
Source training uses only the assigned source roles.

After checking the source artifacts, authorize held-target evaluation and
finalization with the same paths and output directory:

```bash
python scripts/run_three_dataset_pipeline.py \
  --dronerfa-dir "$SCARS_DRONERFA_DIR" \
  --dronerfb-dir "$SCARS_DRONERFB_DIR" \
  --drff-r2-dir "$SCARS_DRFF_R2_DIR" \
  --output-dir "$SCARS_RUN_DIR" \
  --device cuda \
  --max-windows-per-recording 16 \
  --authorize-pilot-target \
  --stop-after finalize
```

The result is `$SCARS_RUN_DIR/scars-pilot/results.json`. Keep the complete output
directory, including predictions, checkpoints, source artifacts, configuration
hashes, environment details, and target-access ledger.

## Memory and resume

The pipeline uses chunked transforms and disk-backed relation tensors. Use a local
SSD; the planning estimate is at least 30 GiB free per concurrent fold, and larger
corpora may need more. Folds and models run sequentially. Actual RAM/VRAM use
depends on the data and execution stage; a full-corpus 16 GB RAM fit is not guaranteed.

Rerun the same command to use guarded resume. Preserve the original checkout and
environment for an existing campaign. The optional
`--allow-memory-implementation-amendment` flag permits only compatible pre-target
checkpoint adoption after provenance checks; it is unnecessary for a fresh run.
Do not edit recorded hashes to force reuse across different source trees.

## Repository layout

```text
src/scars/       data loading, representations, models, training, evaluation
configs/         scientific configurations and label contracts
scripts/         pipeline runners and MAT image exporters
tests/           unit and generated-fixture integration tests
results/schema/  experiment output schema
paper/           result templates, validation, and LaTeX export helpers
docs/            datasets, environment, and protocol
tools/           repository checks
```

`paper/` contains reporting helpers, not the complete submission manuscript.
`SOURCE_CHECKSUMS.sha256` records the distributed source/configuration files;
`tools/check_release.py` checks their integrity and scans for excluded artifacts.
After intentional source changes, review the diff and update the affected
checksums before submitting a contribution.

## License and citation

Original code and documentation are licensed under [Apache-2.0](LICENSE).
Datasets and dependencies retain their own terms; see [NOTICE](NOTICE),
[third-party notices](THIRD_PARTY_NOTICES.md), and [dataset licenses](docs/DATASETS.md).

Use [CITATION.cff](CITATION.cff) to cite this software and the associated work.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
