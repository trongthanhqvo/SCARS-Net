# Contributing

Use a fresh branch and explain the scientific or engineering purpose of a change.
Run `python -m pytest -p no:cacheprovider` and `python tools/check_release.py`.
The ordinary suite uses generated fixtures and must not require private corpora.
Do not attach raw recordings, secrets, machine-specific paths, or unreviewed
checkpoints/results to issues or pull requests.

Changes to representations, source roles, sampling, nuisances, models, losses,
estimands, or decision rules require an explicit new protocol version. A storage
or packaging fix must not silently change these choices. Keep original run
manifests and use new outputs after a scientific change.

Contributions intended for inclusion are provided under the repository's
Apache-2.0 terms unless explicitly stated otherwise, consistent with Section 5
of the license. Preserve third-party attribution and identify modified files.
