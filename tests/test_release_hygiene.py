"""Release-only guards: no dataset access and no scientific protocol changes."""
import importlib.util
from pathlib import Path
import subprocess


def checker():
    path = Path(__file__).resolve().parents[1] / "tools/check_release.py"
    spec = importlib.util.spec_from_file_location("release_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_checks_source_integrity_and_has_no_data():
    report = checker().check()
    assert report["status"] == "PASS", report["issues"]
    assert report["verified_source_files"] == 99


def test_public_entrypoint_uses_stable_configuration_paths():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/run_three_dataset_pipeline.py").read_text()
    for name in (
        "configs/label_contract.three_dataset_binary.json",
        "configs/exclusions.three_dataset.json",
    ):
        assert name in runner
        assert (root / name).is_file()
    assert (root / "environment.yml").is_file()
    assert "environment.yml" in (root / "src/scars/cli/build_pre_target_freeze.py").read_text()


def test_dataset_audit_requires_user_registry():
    import os
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-m", "scars.cli.audit_datasets"],
        cwd=root, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "required: registry" in result.stderr


def test_scan_rejects_data_and_links(tmp_path, monkeypatch):
    module = checker()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "waveform.mat").write_bytes(b"fake test fixture")
    (tmp_path / "linked-data").symlink_to(tmp_path / "waveform.mat")
    report = module.check()
    assert any("waveform.mat: excluded artifact" in issue for issue in report["issues"])
    assert any("linked-data: symlink" in issue for issue in report["issues"])


def test_staged_scan_checks_index_instead_of_clean_working_copy(tmp_path, monkeypatch):
    module = checker()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    document = tmp_path / "example.txt"
    private_path = "/" + "home" + "/" + "example-user" + "/data"
    document.write_text(private_path)
    subprocess.run(["git", "add", "example.txt"], cwd=tmp_path, check=True)
    document.write_text("sanitized working copy")
    report = module.check(staged=True)
    assert any("example.txt: machine-specific" in issue for issue in report["issues"])
