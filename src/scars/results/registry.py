from __future__ import annotations

from copy import deepcopy


CONFIRMATORY_H2_CONFIGURATION_ORDER = (
    "raw_iq",
    "log_stft",
    "cwt",
    "wst_J3Q2",
    "wst_J4Q2",
    "wst_J5Q2",
    "cyclic_A4",
    "cyclic_A8",
    "cyclic_A16",
    "wst_cyclic_resolution_8",
    "wst_cyclic_resolution_16",
    "wst_cyclic_resolution_32",
)


RECOGNITION_CONDITIONS = [
    "raw_iq_cnn",
    "patch_transformer",
    "magnitude_phase_cnn",
    "log_stft_cnn",
    "cwt_cnn",
    "wst_only",
    "cyclic_only",
    "stft_dct_xgboost",
    "fixed_wavelet_subbands",
    "log_psd_sobel",
    "early_fusion_resnet",
    "uniform_late_fusion",
    "ordinary_gate",
    "shuffled_pcrd",
    "pcrd",
]

ABLATION_CONDITIONS = [
    "W",
    "C",
    "W+C",
    "W+C+E",
    "W+C+E+S",
    "active_minus_W",
    "active_minus_C",
    "active_minus_E",
    "active_minus_S",
    "wst_J3Q1",
    "wst_J3Q2",
    "wst_J3Q4",
    "wst_J4Q1",
    "wst_J4Q2",
    "wst_J4Q4",
    "wst_J5Q1",
    "wst_J5Q2",
    "wst_J5Q4",
    "cyclic_A4",
    "cyclic_A8",
    "cyclic_A16",
    "cyclic_permuted",
    "norm_percentile",
    "norm_zscore",
    "norm_none",
    "resolution_8",
    "resolution_16",
    "resolution_32",
    "selection_scalar",
    "selection_pareto",
]

EXTERNAL_SOTA_CONDITIONS = [
    "open_rfnet",
    "asa",
    "avoiding_shortcuts",
    "riei",
    "mtl_sei",
    "mcaff",
    "s3r",
    "rff_llm",
]

RECOGNITION_FIELDS = [
    "condition_id",
    "held_domain",
    "recording_count",
    "mean_macro_f1",
    "ci95_low",
    "ci95_high",
    "worst_domain_macro_f1",
    "balanced_accuracy",
    "parameters",
    "macs",
    "latency_ms",
    "tensor_bytes",
    "representation_latency_ms",
    "model_latency_ms",
    "peak_memory_bytes",
]

ABLATION_FIELDS = [
    "condition_id",
    "held_domain",
    "recording_count",
    "mean_macro_f1",
    "ci95_low",
    "ci95_high",
    "delta_vs_pcrd",
    "delta_ci95_low",
    "delta_ci95_high",
    "cosine",
    "nmae",
    "domain_probe_accuracy",
    "parameters",
    "macs",
    "latency_ms",
    "interpretation",
]

TABLE_ROW_FIELDS = {
    "tab_datasets": ["dataset_id", "recording_count", "group_count", "class_count", "has_background", "split_axes", "eligibility", "license", "checksum_manifest"],
    "tab_channel_contract": ["family", "dtype", "shape", "model_input"],
    "tab_baselines": ["condition_id", "source_only", "target_tuning"],
    "tab_external_sota": [
        "condition_id",
        "eligibility",
        "ineligibility_reason",
        "target_privilege",
        "native_input",
        "observation_samples",
        "recording_count",
        "mean_macro_f1",
        "ci95_low",
        "ci95_high",
        "worst_domain_macro_f1",
        "parameters",
        "macs",
        "latency_ms",
    ],
    "tab_primary_results": RECOGNITION_FIELDS,
    "tab_m1": ["component", "estimate", "interval_or_threshold", "status"],
    "tab_ablations": ABLATION_FIELDS,
    "tab_robustness": ["condition_id", "snr_auc", "sir_auc", "worst_nuisance", "worst_degradation", "recording_count", "ci95_low", "ci95_high", "interpretation"],
    "tab_detection": ["condition_id", "auroc", "auprc", "far", "miss_rate", "ci95", "recording_count", "threshold", "interpretation"],
    "tab_per_class": ["class_domain", "precision", "recall", "f1", "recording_support", "ci95_low", "ci95_high"],
    "tab_routing_calibration": ["diagnostic", "estimate", "interval_or_threshold", "recording_support", "interpretation"],
    "tab_hypotheses": ["id", "estimand", "raw_p", "holm_p", "status", "decision_reason"],
    "tab_integrity": ["audit", "status", "critical", "evidence_path_hash"],
}

FIGURE_DATA_FIELDS = {
    "fig_main_recognition": ["series_id", "panel_id", "x", "y", "ci95_low", "ci95_high", "recording_support", "held_domain"],
    "fig_m1_effects": ["series_id", "panel_id", "x", "y", "ci95_low", "ci95_high", "recording_support", "held_domain"],
    "fig_domain_heatmap": ["matrix", "x_labels", "y_labels"],
    "fig_robustness": ["series_id", "panel_id", "x", "y", "ci95_low", "ci95_high", "recording_support", "held_domain"],
    "fig_routing_diagnostics": ["panel_id", "nuisance", "severity", "family", "weights", "recording_ids", "entropy", "relation_coverage"],
    "fig_scars_selection_h2": ["panel_id", "configuration_id", "source_instability", "source_macro_f1", "cost", "feasible", "pareto", "selected", "h2_residual_instability", "h2_residual_degradation", "held_domain"],
    "fig_confusion_matrices": ["panel_id", "condition_id", "held_domain", "class_labels", "matrix", "recording_support"],
    "fig_calibration": ["panel_id", "condition_id", "held_domain", "confidence", "empirical_accuracy", "ci95_low", "ci95_high", "bin_support", "ece", "brier", "nll"],
    "fig_compute_tradeoff": ["series_id", "panel_id", "x", "y", "recording_support", "held_domain"],
}


def manuscript_registry_envelope() -> dict[str, object]:
    """Canonical static registry shared by runtime and manuscript tooling."""
    return deepcopy(
        {
            "registry": {
                "recognition_conditions": RECOGNITION_CONDITIONS,
                "recognition_row_fields": RECOGNITION_FIELDS,
                "ablation_conditions": ABLATION_CONDITIONS,
                "ablation_row_fields": ABLATION_FIELDS,
                "external_sota_conditions": EXTERNAL_SOTA_CONDITIONS,
                "figure_series_fields": [
                    "series_id", "panel_id", "x_name", "x_unit", "x",
                    "y_name", "y_unit", "y", "ci95_low", "ci95_high",
                    "recording_support", "held_domain",
                ],
                "confusion_panel_fields": FIGURE_DATA_FIELDS["fig_confusion_matrices"],
                "calibration_panel_fields": FIGURE_DATA_FIELDS["fig_calibration"],
                "routing_panel_fields": FIGURE_DATA_FIELDS["fig_routing_diagnostics"],
                "selection_h2_panel_fields": FIGURE_DATA_FIELDS["fig_scars_selection_h2"],
                "allowed_status": [
                    "open",
                    "supported",
                    "partial",
                    "negative",
                    "untestable",
                    "descriptive",
                ],
                "required_seeds": [11, 23, 37, 53, 71],
                "semantic_paths": {
                    "primary_metric": "tables.tab_primary_results.rows[condition_id=pcrd].mean_macro_f1",
                    "worst_domain_macro_f1": "tables.tab_primary_results.rows[condition_id=pcrd].worst_domain_macro_f1",
                    "detection_auroc": "detection.summary.auroc",
                    "robustness_snr_auc": "robustness.normalized_curve_auc.pcrd.snr",
                    "robustness_sir_auc": "robustness.normalized_curve_auc.pcrd.sir",
                    "m1_status": "mechanism.m1.status",
                    "hypotheses": "hypotheses.{H1,H2,H3,H4}",
                    "table_rows": {
                        table_id: f"tables.{table_id}.rows[*]"
                        for table_id in TABLE_ROW_FIELDS
                    },
                    "table_fields": {
                        table_id: {
                            "path": f"tables.{table_id}.rows[*]",
                            "fields": fields,
                        }
                        for table_id, fields in TABLE_ROW_FIELDS.items()
                    },
                    "figure_fields": {
                        figure_id: {
                            "path": f"figures.{figure_id}",
                            "fields": fields,
                        }
                        for figure_id, fields in FIGURE_DATA_FIELDS.items()
                    },
                },
            },
            "eligibility": {"eligible_domains": None, "datasets": []},
            "integrity": {"status": None, "critical_failures": None, "audits": []},
            "tables": {
                **{
                    table_id: {"rows": None, "row_fields": fields}
                    for table_id, fields in TABLE_ROW_FIELDS.items()
                },
            },
            "mechanism": {
                "relation_coverage": {"aggregate": None, "by_nuisance": None, "recording_count": None},
                "m1": {
                    "delta_gate": {"estimate": None, "ci95": None},
                    "delta_shuffle": {"estimate": None, "ci95": None},
                    "worst_domain_difference": {"estimate": None, "ci95": None},
                    "compute_gate": None,
                    "status": None,
                    "reason": None,
                },
            },
            "figures": {
                "fig_main_recognition": {"series": None},
                "fig_m1_effects": {"series": None},
                "fig_domain_heatmap": {"matrix": None},
                "fig_robustness": {"series": None},
                "fig_routing_diagnostics": {"panels": None},
                "fig_scars_selection_h2": {"panels": None},
                "fig_confusion_matrices": {"panels": None},
                "fig_calibration": {"panels": None},
                "fig_compute_tradeoff": {"series": None},
            },
        }
    )
