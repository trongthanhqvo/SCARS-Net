#!/usr/bin/env python3
"""Check public-release content without loading datasets or running experiments.

This is a repository hygiene check, not a proof of license ownership or an
exhaustive secret detector. Default: inspect all deliverable files. --staged:
inspect the Git index (including blob contents), even if ignored or later edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".mat", ".aria2", ".npy", ".npz", ".h5", ".hdf5", ".iq", ".bin",
    ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib", ".safetensors",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf", ".zip", ".tar", ".gz",
    ".pem", ".key", ".p12", ".pfx", ".pyc", ".pyo",
}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}
PRIVATE_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/|/(?:Volumes|media)/[^\s\"']+/")
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{32,}\b"),
]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def files(staged):
    if staged:
        for item in git("ls-files", "--stage", "-z").split(b"\0"):
            if not item:
                continue
            metadata, raw_path = item.split(b"\t", 1)
            mode, object_id, stage = metadata.split()
            name = raw_path.decode("utf-8")
            if mode != b"100644" and mode != b"100755":
                yield name, None
            elif stage != b"0":
                yield name, None
            else:
                yield name, git("cat-file", "blob", object_id.decode("ascii"))
    else:
        for path in sorted(ROOT.rglob("*")):
            rel = path.relative_to(ROOT)
            if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in rel.parts):
                continue
            if path.is_symlink():
                yield rel.as_posix(), None
            elif path.is_file():
                yield rel.as_posix(), path.read_bytes()


def check(staged=False):
    issues = []
    content = dict(files(staged))
    if not content:
        issues.append("No release files found")
    for name, blob in content.items():
        path = Path(name)
        if blob is None:
            issues.append(name + ": symlink, submodule, or unresolved index entry")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in {".env", ".DS_Store", "credentials.json"}:
            issues.append(name + ": excluded artifact type")
        if path.parts[0] in {"data", "datasets", "checkpoints", "runs", "wandb"}:
            issues.append(name + ": runtime/data directory")
        if path.parts[0] == "results" and not name.startswith("results/schema/"):
            issues.append(name + ": runtime results must remain outside release")
        if len(blob) > 10 * 1024 * 1024 or b"\x00" in blob:
            issues.append(name + ": large or binary content")
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(name + ": non-text content")
            continue
        if PRIVATE_PATH.search(text):
            issues.append(name + ": machine-specific user/mount path")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            issues.append(name + ": potential credential; inspect privately")
    required = {"README.md", "LICENSE", "CITATION.cff", ".gitignore", "pyproject.toml",
                "requirements.txt", "docs/DATASETS.md", "docs/PROTOCOL.md"}
    for name in sorted(required - content.keys()):
        issues.append(name + ": required release file missing")
    # Verify the distributed source/configuration inventory against its checksums.
    # Intentional edits require reviewed checksum updates.
    manifest = content.get("SOURCE_CHECKSUMS.sha256")
    scientific_count = 0
    if manifest is None:
        issues.append("SOURCE_CHECKSUMS.sha256 missing")
    else:
        expected = {}
        for line in manifest.decode().splitlines():
            digest, name = line.split("  ", 1)
            if name.startswith(("src/", "configs/")):
                expected[name] = digest
        actual = {n for n in content if n.startswith(("src/", "configs/"))}
        for name in sorted(actual ^ expected.keys()):
            issues.append(name + ": scientific source/config file set differs from distributed checksums")
        for name, digest in expected.items():
            if content.get(name) is not None and hashlib.sha256(content[name]).hexdigest() == digest:
                scientific_count += 1
            else:
                issues.append(name + ": scientific source/config bytes differ from distributed checksums")
    return {"status": "PASS" if not issues else "FAIL", "mode": "staged" if staged else "working-tree",
            "file_count": len(content), "verified_source_files": scientific_count, "issues": issues}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Read actual staged Git blobs")
    args = parser.parse_args()
    try:
        report = check(args.staged)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print("Release check failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
