from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys


def sha256_file(path: Path, block_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_tree_sha256(root: Path) -> str:
    """Hash the executable experiment source even when it is not Git-tracked."""
    roots = [root / name for name in ("src", "scripts", "configs", "tests")]
    files = [root / "pyproject.toml", root / "requirements.txt"]
    for directory in roots:
        if directory.is_dir():
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: str(item.relative_to(root))):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def environment_manifest(project_root: Path, config_paths: list[Path]) -> dict[str, object]:
    versions = {}
    for distribution in (
        "numpy",
        "PyYAML",
        "scipy",
        "scikit-learn",
        "torch",
        "h5py",
        "Pillow",
        "xgboost",
        "psutil",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    config_hashes = {}
    for path in config_paths:
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            portable = str(resolved.relative_to(project_root.resolve())).replace("\\", "/")
        except ValueError as error:
            raise ValueError("Configuration provenance must remain inside the project root") from error
        config_hashes[portable] = sha256_file(resolved)
    digest = hashlib.sha256(json.dumps(config_hashes, sort_keys=True).encode()).hexdigest()
    hardware = {"cuda_available": False, "gpu_name": None, "gpu_vram_bytes": None}
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            hardware = {
                "cuda_available": True,
                "gpu_name": properties.name,
                "gpu_vram_bytes": int(properties.total_memory),
            }
    except ImportError:
        pass
    return {
        "git_commit": git_commit(project_root),
        "source_tree_sha256": source_tree_sha256(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": hardware,
        "packages": versions,
        "config_hashes": config_hashes,
        "combined_config_hash": digest,
    }
