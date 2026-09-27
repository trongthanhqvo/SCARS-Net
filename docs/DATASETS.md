# Data acquisition, input layout, and reuse terms

This repository distributes code and configuration only. Download the datasets
from their maintainers and keep the archives, raw IQ, converted images/tensors,
checkpoints, and run manifests outside the Git repository. A code license does
not license third-party data. Preserve the dataset version, original download
terms, citation, and local checksums with each experiment.

## Authoritative sources and license evidence

| Dataset | Acquisition/reference source | License evidence and release policy |
| --- | --- | --- |
| DroneRFa | [Publisher article, DOI 10.11999/JEIT230570](https://jeit.ac.cn/article/doi/10.11999/JEIT230570); [download endpoint identified by the paper](https://jeit.ac.cn/web/data/getData?dataType=Dataset3) | A specific data redistribution license was not verified from the inspected publisher material. Public availability does not establish permission to redistribute. Obtain the current terms from the provider; this release redistributes no samples or derived tensors. |
| DroneRFb-DIR | [ScienceDB dataset](https://www.scidb.cn/en/detail?dataSetId=84cf9101e739402784b1396783881202); [publisher article, DOI 10.11999/JEIT240804](https://jeit.ac.cn/article/doi/10.11999/JEIT240804) | A specific data redistribution license was not verified. Check the downloaded release's terms or request clarification from the maintainers. No raw files, label manifests, or derived tensors are bundled. |
| DRFF-R2 | [Dataset DOI 10.57760/sciencedb.36815](https://doi.org/10.57760/sciencedb.36815); [ScienceDB landing page](https://www.scidb.cn/en/detail?dataSetId=b8a16448c1284fd1be1ded9ccc45be20); [dataset paper](https://arxiv.org/html/2603.00106v1) | The dataset paper's Data Records/Data Availability sections explicitly state CC BY 4.0. Attribute the creators, link the license, and identify changes when sharing permitted derivatives; other applicable rights still apply. This code-only release includes no data even for this dataset. |

Use the official download workflow, including any registration or access request,
and verify the terms accompanying the dataset release you obtain.

The `license_verified` fields in the frozen label contracts record the
original campaign author's local-use attestations. They are not independent
license verification, a public redistribution grant, or a claim that another
user has accepted the data provider's terms. The preserved `derived_tensor_release_reviewed`
fields remain false. Review access rights before executing a new campaign.

## Paths accepted by the three-dataset runner

Supply the directories containing the actual MAT files, not the metadata-only
dataset descriptions. The runner creates an internal symlink layout; users do
not need to move or duplicate their data.

```text
DroneRFa/                         <-- --dronerfa-dir
  T0000_D00_S0000.mat
  T0001_D00_S0000.mat
  T10000_S0000.mat
  ...                             (recursive discovery)

twin_droneRF/                     <-- --dronerfb-dir
  train/
    A1_IN_S0_slice_1.mat
    ...
  test/
    0.mat
    ...
  train_labels.txt
  test_labels.txt

DRFF-R2/                          <-- --drff-r2-dir
  dataset1-single_drone_states/
  dataset2-drone_mixed/
  dataset3-single_drone_hover/
  dataset4-single_drone_dual_frequency/
  dataset5-single_drone_inside_absorbent_cotton/
  dataset6-wifi_mixed/
  dataset7-environment/
    indoor_environment.mat
    outdoor_environment.mat
```

Directory/file examples describe the loader contract, not a bundled dataset.
Keep nested subdirectories in DRFF-R2. Do not rename or merge physical recordings.
The loader uses `h5py`: MAT files must use the expected MATLAB v7.3/HDF5
layout. Older non-HDF5 MAT files are not silently converted.

## IQ semantics and labels

The implementation is `src/scars/data/mat_recordings.py`. HDF5 vector slices are
read as needed and converted to `complex64` using `I + 1j*Q`. Discovery reads
metadata/shapes; preflight content hashing still streams whole files to establish
checksums. Bounded waveform reads do not eliminate this first-run hashing cost.

| Dataset | HDF5 waveform keys | Sample rate used by loader | Background and grouping |
| --- | --- | --- | --- |
| DroneRFa | `RF0_I/RF0_Q`, `RF1_I/RF1_Q`, when present | 100 MS/s | `T0000` is background; grouping uses class and segment (`T..._S...`) and joins streams of the same physical file. Names with or without `_D...` are supported. Missing distance stays missing. Windows obey a 10,000,000-sample continuity block. |
| DroneRFb-DIR | `I`, `Q` | 80 MS/s | `B` / background filenames are negatives. Identity, session, and propagation form the source group; missing sessions are conservatively labeled `unknown_session`. `IN/OUT` means LoS/NLoS, not indoor/outdoor. |
| DRFF-R2 | `RF0_I`, `RF0_Q` | `Fs` metadata, with the loader's documented 100 MS/s default | Subset 7 is background recorded before UAV activation. UAV unit/day/receiver grouping joins related subsets; each background file remains indivisible. Subset 6 contains UAV plus Wi-Fi and is positive, not background. |

DroneRFb `test_labels.txt` maps `original_labeled_filename anonymous_id`, e.g.
`D1_IN_S2_slice_47.mat 0` maps `test/0.mat` to its original identity metadata.
The anonymous ID is not the class label. The loader decodes these labels and the
campaign then constructs its own group-based source/held-dataset folds; native
`train/` and `test/` folders are not used as the cross-dataset split.

The binary ontology is preserved in
`configs/label_contract.three_dataset_binary.json`. DroneRFa controller classes
are included as UAV-related RF. DRFF-R2 subset 2 (multi-emitter mixtures) is
excluded from the single-label arm. The two paths in
`configs/exclusions.three_dataset.json` remain excluded, including after a
download completes. Other incomplete MAT downloads cause a stop. Changing these
rules changes the campaign and is not a packaging operation.

With native rates, 4096 samples represent 40.96 microseconds at 100 MS/s and
51.2 microseconds at 80 MS/s; the release introduces no resampling. The exact
sampling/role amendment is described in [PROTOCOL.md](PROTOCOL.md).

## Images, tensors, and publication

The MAT exporters are inspection utilities. PNGs and montage images are not the
full-pipeline IQ input. Their `tensor.npy` files cannot replace IQ for source-fold
normalization fitting, physical perturbations, cyclic selection, or H2 variants.
The full three-dataset runner consumes the MAT roots above.

Keep provider label files and run-specific recording assignments with local
data. `.gitignore` and `tools/check_release.py` guard against accidentally adding
waveforms, images, arrays, checkpoints, or private run outputs to this release.
Future publication of predictions/checkpoints needs a separate data-rights review.
