"""Import pre-target source representations into a new amended pilot run."""
import copy
import json
import shutil
from pathlib import Path

from scars.experiment.common import atomic_json
from scars.experiment.pilot_policy import POLICY, apply_fixed_families
from scars.results.provenance import environment_manifest, sha256_file


def prepare(previous_run: Path, output_run: Path, project_root: Path):
    previous_run, output_run = Path(previous_run).resolve(), Path(output_run).resolve()
    old = previous_run / "scars-pilot"
    new = output_run / "scars-pilot"
    if previous_run == output_run or new.exists() or (output_run / "preflight").exists():
        raise FileExistsError("Amendment requires a new output directory; preserve the old campaign")
    def checked(relative, expected=None):
        path = (old / relative).resolve()
        if old not in path.parents or not path.is_file():
            raise ValueError("Invalid prior artifact: " + relative)
        if expected and sha256_file(path) != expected:
            raise ValueError("Prior artifact hash mismatch: " + relative)
        return path
    campaign = json.loads(checked("source_campaign.json").read_text())
    preflight = json.loads(checked("preflight.json", campaign["input_hashes"]["preflight"]).read_text())
    if preflight.get("campaign_mode") != "pilot_two_dataset" or set(preflight["datasets"]) != set(POLICY["datasets"]):
        raise ValueError("Only a two-dataset pilot can be amended here")
    if campaign.get("target_access_authorized") or (old / "target_campaign.json").exists():
        raise PermissionError("Cannot amend a campaign after target access")
    configs = sorted((project_root / "configs").glob("*.yaml")) + sorted((project_root / "configs").glob("*.json"))
    current = environment_manifest(project_root, configs)
    old_packages = campaign["provenance"].get("packages", {})
    package_changes = {
        key: {"previous": old_packages.get(key), "current": current["packages"].get(key)}
        for key in set(old_packages) | set(current["packages"])
        if old_packages.get(key) != current["packages"].get(key)
    }
    # XGBoost is not used by source representation freezing. Permit only its
    # explicitly documented first installation; never relax other dependencies.
    allowed_install = {"xgboost": {"previous": None, "current": "2.0.3"}}
    if package_changes and package_changes != allowed_install:
        raise RuntimeError("Reuse requires original packages, except missing XGBoost -> 2.0.3")
    for key in ("platform", "hardware", "combined_config_hash"):
        if current.get(key) != campaign["provenance"].get(key):
            raise RuntimeError("Reuse requires the original environment/configuration: " + key)
    copies = []
    freezes = []
    for entry in campaign["folds"]:
        directory = entry["directory"]
        state = json.loads(checked(directory + "/run_state.json").read_text())
        if state.get("target_reads") != 0 or state.get("target_unlocks") != 0:
            raise PermissionError("Prior target ledger is not pristine")
        if (old / directory / "target-cache").exists():
            raise PermissionError("Prior target cache exists")
        path = checked(directory + "/source_freeze.json", entry["source_freeze_sha256"])
        freeze = json.loads(path.read_text())
        for record in list(freeze["candidates"].values()) + list(freeze["h2_candidates"].values()) + [freeze["canonical_tensor_artifact"]]:
            artifact = checked(directory + "/" + record["path"], record["sha256"])
            copies.append((artifact, new / directory / record["path"]))
        freezes.append((entry, freeze, state))
    for key, filename in (("recordings", "recordings_recognition.json"), ("splits", "splits.json")):
        checked(filename, campaign["input_hashes"][key])
    output_run.mkdir(parents=True, exist_ok=True)
    (output_run / "preflight").mkdir()
    new.mkdir()
    for filename in ("preflight.json", "splits.json", "recordings_recognition.json"):
        shutil.copy2(old / filename, new / filename)
        shutil.copy2(old / filename, output_run / "preflight" / filename)
    for source, dest in copies:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    revised = copy.deepcopy(campaign)
    revised.pop("model_freezes", None)
    revised.pop("model_freeze_statuses", None)
    revised.update(status="all_source_folds_frozen", target_access_authorized=False,
                   provenance=current, pilot_policy=dict(POLICY))
    revised["amendment_provenance"] = {
        "previous_campaign_sha256": sha256_file(old / "source_campaign.json"),
        "previous_environment": campaign["provenance"],
        "package_changes": package_changes,
        "authorization": "author requested complete two-dataset pilot on 2026-09-07",
        "source_waveforms_read_during_import": 0,
    }
    for entry, freeze, state in freezes:
        directory = entry["directory"]
        shutil.copy2(old / directory / "source_freeze.json", new / directory / "source_freeze_before_amendment.json")
        if freeze.get("pilot_policy") != POLICY:
            apply_fixed_families(freeze)
        digest = atomic_json(new / directory / "source_freeze.json", freeze)
        for revised_entry in revised["folds"]:
            if revised_entry["fold_id"] == entry["fold_id"]:
                revised_entry["source_freeze_sha256"] = digest
        state["source_artifact_hash"] = freeze["canonical_tensor_artifact"]["sha256"]
        state["phase"] = "PARETO_FROZEN"
        atomic_json(new / directory / "run_state.json", state)
    atomic_json(new / "source_campaign.json", revised)
    return new / "source_campaign.json"
