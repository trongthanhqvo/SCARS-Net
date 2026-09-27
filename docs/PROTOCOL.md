# Protocol and scope

The public three-dataset entry point is
`scripts/run_three_dataset_pipeline.py`. It runs the exploratory
`pilot_three_dataset` protocol for binary background/UAV-related RF recognition
using DroneRFa, DroneRFb-DIR, and DRFF-R2.

## Execution contract

- Physical groups are split before windowing. Source fit, selection, calibration,
  and validation roles remain separated from held-domain model evaluation.
- The runner uses 4096 complex samples per window, hop 2048, and a maximum of
  16 windows per recording by default.
- Source selection is retained diagnostically; the deployed pilot uses fixed
  W/C/E/S families as defined in `src/scars/experiment/pilot_policy.py`.
- Learned models use seeds 11, 23, 37, 53, and 71. Source freeze uses seed 24021.
- The sparse-background amendment allocates the two DRFF-R2 background groups
  to fit/validation; other source data supply selection/calibration background.
- Held-target evaluation requires explicit authorization. Preflight reads file
  metadata and hashes file contents for integrity, including held recordings.
- Output remains labeled as exploratory. Hypothesis decisions are computed from
  eligible evidence, not hard-coded to manuscript conclusions.

## Relation to the manuscript

This runner is not an exact reproduction contract for all final manuscript
tables: its fixed W/C/E/S, 16-window cap, and sparse-background roles differ from
the manuscript's selected-family predictor, 64-window pool, and four-group
dataset-class support rule. Do not equate their empirical or inferential results.
Exact reproduction requires the matching campaign source, effective configuration,
and original prediction/provenance artifacts.

External ASA, SDG/MSCAN-L, Open-RFNet-C, and DIAL+MR reimplementations are not included.
Their comparison is a separate campaign. The result helpers in `paper/` validate
and export experiment contracts; they are not the full manuscript renderer.

## Provenance and continuation

Source-tree hashes cover scripts, tests, configurations, and dependency metadata
as well as model code. File renaming or packaging changes can therefore invalidate
resume compatibility even without a change in scientific computation.

Use a new output directory for a new source tree. Retain the original checkout,
environment, and all artifacts for continuation of an existing campaign.
The implementation-amendment flag does not authorize bypassing provenance,
target-access, or configuration checks.

`scars.cli.audit_datasets` accepts an explicit user-owned eligibility registry.
Local inventory snapshots are not distributed as evidence of another user's
dataset availability. The MAT pipeline builds its own preflight artifacts from
the supplied dataset paths.
