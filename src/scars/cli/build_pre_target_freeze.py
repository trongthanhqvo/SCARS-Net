from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET

import yaml

from scars.baselines import assert_unique_h2_effective_configurations, frozen_external_sota_registry
from scars.experiment.sampling import frozen_sampling_contract
from scars.experiment.pretarget import FROZEN_DATASETS, verify_source_campaign_pre_target
from scars.results.provenance import environment_manifest, sha256_file
from scars.selection.nuisance import registered_nuisance_specs


GATE_ORDER = (
    "nuisance_contract_frozen",
    "h2_bank_unique",
    "h2_statistical_audit_pass",
    "sampling_policy_frozen",
    "five_seed_ensemble_semantics_frozen",
    "external_sota_statuses_frozen",
    "ontology_frozen",
    "group_leakage_audit_pass",
    "source_only_dry_run_pass",
    "invariant_tests_pass",
    "hashes_and_provenance_complete",
    "target_access_count_zero",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    excluded = {".pytest_cache", "__pycache__", "pre_target_freeze", "results"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if any(part in excluded or part.endswith((".pyc", ".pyo")) for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run(
    *,
    output_dir: Path,
    source_campaign_dir: Path | None,
    label_contract: Path | None,
    invariant_test_report: Path | None,
) -> Path:
    if output_dir.exists():
        raise FileExistsError("Freeze output must be a new immutable directory")
    output_dir.mkdir(parents=True)
    project_root = Path(__file__).resolve().parents[3]
    config_paths = sorted((project_root / "configs").glob("*.yaml")) + sorted(
        (project_root / "configs").glob("*.json")
    )
    gates = {name: False for name in GATE_ORDER}
    blockers: list[dict[str, str]] = []

    # Static protocol gates.
    nuisance = [item.__dict__ for item in registered_nuisance_specs()]
    gates["nuisance_contract_frozen"] = bool(nuisance)
    signatures = assert_unique_h2_effective_configurations()
    gates["h2_bank_unique"] = len(signatures) == 12
    statistics = yaml.safe_load((project_root / "configs/statistics.yaml").read_text())
    gates["h2_statistical_audit_pass"] = (
        statistics["hierarchical_h2"]["minimum_held_domains"] >= 3
        and statistics["hierarchical_h2"]["minimum_joint_nonzero_configurations"] >= 8
        and statistics["bootstrap"]["resamples"] == 10_000
    )
    gates["sampling_policy_frozen"] = bool(frozen_sampling_contract())
    gates["five_seed_ensemble_semantics_frozen"] = True
    external = frozen_external_sota_registry()
    gates["external_sota_statuses_frozen"] = all(
        item["implementation_status"] in {"executable", "frozen_ineligible"}
        for item in external
    )

    ontology = None
    if label_contract and label_contract.is_file():
        ontology = _json(label_contract)
        datasets = ontology.get("datasets", {})
        gates["ontology_frozen"] = set(datasets) == FROZEN_DATASETS and all(
            item.get("license_verified")
            and item.get("temporal_adjacency_subsumed_by_split_group_verified")
            and item.get("labels")
            for item in datasets.values()
        )
    if not gates["ontology_frozen"]:
        blockers.append({
            "gate": "ontology_frozen",
            "reason": "author-verified licenses, physical groups, temporal grouping, and exact shared-label mappings are incomplete",
        })

    source_campaign = None
    source_verification = None
    if source_campaign_dir is not None and (source_campaign_dir / "source_campaign.json").is_file():
        source_campaign = _json(source_campaign_dir / "source_campaign.json")
        try:
            ontology_hash = sha256_file(label_contract) if label_contract else None
            source_verification = verify_source_campaign_pre_target(
                source_campaign_dir, expected_label_contract_sha256=ontology_hash
            )
            gates["source_only_dry_run_pass"] = True
            gates["hashes_and_provenance_complete"] = True
            gates["target_access_count_zero"] = source_verification["target_access_count"] == 0
            gates["group_leakage_audit_pass"] = source_verification["group_leakage_audit"] == "passed"
        except (OSError, ValueError, RuntimeError, PermissionError, KeyError) as error:
            blockers.append({"gate": "source_campaign_verification", "reason": str(error)})
    else:
        gates["target_access_count_zero"] = True
    for gate in ("group_leakage_audit_pass", "source_only_dry_run_pass", "hashes_and_provenance_complete"):
        if not gates[gate]:
            blockers.append({"gate": gate, "reason": "complete eligible-data source campaign artifact is absent"})

    if invariant_test_report and invariant_test_report.is_file():
        if invariant_test_report.suffix.lower() == ".xml":
            suite = ET.parse(invariant_test_report).getroot()
            suites = [suite] if suite.tag == "testsuite" else list(suite.findall("testsuite"))
            gates["invariant_tests_pass"] = bool(suites) and all(
                int(item.attrib.get("failures", 0)) == 0
                and int(item.attrib.get("errors", 0)) == 0
                for item in suites
            )
        else:
            report_text = invariant_test_report.read_text(encoding="utf-8")
            gates["invariant_tests_pass"] = " failed" not in report_text.lower() and " error" not in report_text.lower()
    if not gates["invariant_tests_pass"]:
        blockers.append({"gate": "invariant_tests_pass", "reason": "a passing invariant-test report was not supplied"})

    contracts_dir = output_dir / "contracts"
    contracts_dir.mkdir()
    for path in config_paths:
        shutil.copy2(path, contracts_dir / path.name)
    for path in (project_root / "pyproject.toml", project_root / "environment.yml", project_root / "requirements.txt", project_root / "requirements-torch-cu121.txt"):
        if path.is_file():
            shutil.copy2(path, contracts_dir / path.name)
    if label_contract and label_contract.is_file():
        shutil.copy2(label_contract, contracts_dir / "ontology.json")
    if invariant_test_report and invariant_test_report.is_file():
        shutil.copy2(invariant_test_report, output_dir / f"invariant-tests{invariant_test_report.suffix.lower()}")

    frozen_environment = environment_manifest(project_root, config_paths)
    source_tree_sha256 = str(frozen_environment["source_tree_sha256"])
    package_content_sha256 = _tree_hash(project_root)
    source_campaign_sha256 = (
        sha256_file(source_campaign_dir / "source_campaign.json") if source_campaign is not None else None
    )
    ledger = {
        "schema_version": "scars-target-access-ledger-1.0",
        "target_access_count": 0,
        "target_unlock_count": 0,
        "target_performance_inspected": False,
        "authorization": False,
    }
    _atomic_json(output_dir / "target-access-ledger.json", ledger)
    status = "ready" if all(gates.values()) else "blocked"
    manifest = {
        "schema_version": "scars-pre-target-freeze-1.0",
        "status": status,
        "target_access_authorized": False,
        "source_tree_sha256": source_tree_sha256,
        "package_content_sha256": package_content_sha256,
        "source_campaign_sha256": source_campaign_sha256,
        "source_verification": source_verification,
        "environment": frozen_environment,
        "gates": gates,
        "blockers": blockers,
        "h2_effective_signatures": signatures,
        "external_sota_registry": external,
        "sampling_contract": frozen_sampling_contract(),
        "target_access_ledger": "target-access-ledger.json",
    }
    _atomic_json(output_dir / "freeze-manifest.json", manifest)
    manifest_lines = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.name != "MANIFEST.sha256"):
        manifest_lines.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return output_dir / "freeze-manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a fail-closed pre-target freeze package")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-campaign-dir", type=Path)
    parser.add_argument("--label-contract", type=Path)
    parser.add_argument("--invariant-test-report", type=Path)
    return parser


def main() -> int:
    print(run(**vars(build_parser().parse_args())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
