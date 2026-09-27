from __future__ import annotations

from copy import deepcopy


_FROZEN_EXTERNAL_SOTA = (
    ("open_rfnet", "frozen_ineligible", "native_training_code_and_closed_set_ontology_adapter_are_not_present_in_the_frozen_source_tree"),
    ("asa", "frozen_ineligible", "faithful_semantic_augmentation_generator_and_author_hyperparameter_contract_are_not_present_in_the_frozen_source_tree"),
    ("avoiding_shortcuts", "frozen_ineligible", "faithful_native_impairment_model_and_training_implementation_are_not_present_in_the_frozen_source_tree"),
    ("riei", "frozen_ineligible", "verified_receiver_identifiers_are_not_available_in_the_current_data_contract_and_no_faithful_adapter_is_present"),
    ("mtl_sei", "frozen_ineligible", "verified_receiver_labels_required_by_the_native_multitask_objective_are_not_available"),
    ("mcaff", "frozen_ineligible", "faithful_native_IQ_CFO_FFT_STFT_view_generator_and_training_implementation_are_not_present"),
    ("s3r", "frozen_ineligible", "signal_semantics_and_open_set_ontology_cannot_be_mapped_without_changing_the_frozen_closed_set_estimand"),
    ("rff_llm", "frozen_ineligible", "individual_emitter_identity_semantics_and_required_teacher_resources_are_not_available"),
)


def frozen_external_sota_registry() -> list[dict[str, str]]:
    records = [
        {
            "condition_id": condition_id,
            "implementation_status": status,
            "target_privilege": "source_only",
            "reason": reason,
        }
        for condition_id, status, reason in _FROZEN_EXTERNAL_SOTA
    ]
    unresolved = [item for item in records if item["implementation_status"] not in {"executable", "frozen_ineligible"}]
    if unresolved:
        raise RuntimeError("Every external SOTA comparator must be executable or frozen-ineligible")
    return deepcopy(records)
