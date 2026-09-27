# Environment and hardware

The installation profile targets Python 3.11, NumPy 1.26.4, SciPy 1.11.4,
scikit-learn 1.4.2, PyYAML 6.0.2, h5py 3.11.0, Pillow 10.4.0,
Matplotlib 3.8.4, XGBoost 2.1.1, psutil 6.0.0, pytest 8.3.2,
PyTorch 2.3.1 and torchvision 0.18.1. CUDA wheels use the 12.1 index.
These are preserved installation pins, not evidence of what was installed on a
previous experiment machine. Transitive dependencies are not fully locked.

Prefer a fresh Linux Python 3.11 environment and the README/setup.sh commands.
`environment.yml` provides an alternative Conda specification.
Do not upgrade NumPy to 2.x in the pinned PyTorch profile. Do not overlay a
research campaign's existing environment with a new environment before resuming.

The frozen confirmatory hardware specification and the pilot admission logic
are distinct. Read `configs/deployment_budget.yaml` and preflight diagnostics for
actual checks; the three-dataset command uses the pilot path. It uses
chunked tensors and disk caches but does not prove every full dataset fits 16 GB
host RAM or every GPU. Device model and usable VRAM are measured at runtime;
advertised product names alone are insufficient to establish the profile.

For a new run, store its exact environment outside the source repository:

```bash
python --version
python -m pip freeze > "$SCARS_RUN_DIR/environment-pip-freeze.txt"
nvidia-smi
```

Create the output directory first if the campaign has not started. The run's
native provenance also records key package versions, hardware, configuration
hashes, and source-tree hash. Keep these files with the campaign. A CPU unit test
pass does not validate CUDA timing, OOM behavior at full corpus scale, or published
numerical results.
