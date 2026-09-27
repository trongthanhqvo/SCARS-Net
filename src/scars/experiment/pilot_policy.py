"""Explicit two-dataset pilot amendment; not the confirmatory protocol."""
import math

POLICY = {
    "version": "two-dataset-fixed-wces-20260907",
    "datasets": ["DroneRFa", "DroneRFb-DIR"],
    "active_families": ["W", "C", "E", "S"],
    "global_selection": "measured_and_reported_separately_from_fixed_WCES_model",
    "channel_retention": "fixed_WCES_for_pilot; original_selection_diagnostics_retained",
    "domain_probe": "unavailable_with_one_source_dataset; not a training gate",
    "relation_cells": "train_on_observed_comparable_pairs; report_missing_cells",
    "empty_relations": "CE_only_fallback_explicitly_marked_PCRD_unavailable",
    "degenerate_shuffle": "report_untestable; never_claim_distinct_control",
    "temperature_bound": "retain_bounded_optimum_and_report_boundary",
    "claims": "H1-H4_and_M1_not_confirmatory_with_two_datasets",
    "seeds": [11, 23, 37, 53, 71],
}


THREE_POLICY = {
    **POLICY,
    "version": "three-dataset-fixed-wces-sparse-background-20260909",
    "datasets": ["DroneRFa", "DroneRFb-DIR", "DRFF-R2"],
    "domain_probe": "measured_on_source_domains; diagnostic_not_a_training_gate",
    "source_roles": "DRFF-R2_background_two_groups_hash_sorted_fit_and_validation; other_strata_unchanged",
    "claims": "exploratory_amended_protocol; H1-H4_and_M1_not_confirmatory",
}


def policy_for(preflight):
    mode = preflight.get("campaign_mode")
    if mode == "pilot_three_dataset":
        return THREE_POLICY
    if mode == "pilot_two_dataset":
        return POLICY
    raise ValueError("Not an explicitly registered pilot mode")


def is_pilot(preflight):
    return preflight.get("campaign_mode") in {"pilot_two_dataset", "pilot_three_dataset"}


def apply_fixed_families(freeze, policy=POLICY):
    freeze["pilot_policy"] = dict(policy)
    freeze["registered_active_families"] = list(freeze.get("active_families", []))
    freeze["active_families"] = list(POLICY["active_families"])
    freeze["active_indices"] = [0, 1, 2, 3]
    # channel_decisions remains the original measured source decision.


def finite_json(value):
    """Represent unavailable diagnostics as JSON null, never as invented zero."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value
