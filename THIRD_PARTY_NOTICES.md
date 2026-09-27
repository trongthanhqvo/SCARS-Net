# Third-party software and data

The Apache-2.0 license applies to original project code/documentation for which
the contributors hold the rights. It does not relicense dependencies or datasets.
No external method repository or dataset paper PDF is vendored in this release.

Dependencies are installed from their providers, rather than redistributed in
this repository. Keep their accompanying license/NOTICE files with any future
binary distribution. Primary projects include NumPy, SciPy, scikit-learn, PyYAML,
h5py, Pillow, Matplotlib, XGBoost, psutil, pytest, PyTorch, and torchvision. The
requirements/environment files identify intended versions. Their installed
distribution metadata and upstream LICENSE files are the authority for their
own terms; transitive dependencies retain their licenses as well.

External SOTA implementations (ASA, SDG/MSCAN-L, Open-RFNet-C, DIAL+MR) are not
included. The external registry in this source records ineligibility,
not an embedded implementation. Integrating those projects in a future release
requires retaining each project's license and attribution.

Dataset sources and checked license evidence are documented in
[docs/DATASETS.md](docs/DATASETS.md). Provider access permission, permission for
research use, and permission to redistribute raw/derived data are separate matters.
