#!/usr/bin/env python3
"""Generate all data-bearing manuscript figures from canonical JSON, fail closed."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

FILES = {
    "fig_main_recognition": "main_recognition.pdf",
    "fig_m1_effects": "m1_effects.pdf",
    "fig_domain_heatmap": "domain_heatmap.pdf",
    "fig_robustness": "robustness_curves.pdf",
    "fig_routing_diagnostics": "routing_diagnostics.pdf",
    "fig_scars_selection_h2": "scars_selection_h2.pdf",
    "fig_confusion_matrices": "confusion_matrices.pdf",
    "fig_calibration": "calibration.pdf",
    "fig_compute_tradeoff": "compute_tradeoff.pdf",
}

def need(value, path):
    if value is None or value == []:
        raise ValueError(f"missing figure data: {path}")
    return value

def draw_series(ax, series, xlabel=None, ylabel=None):
    for s in need(series, "series"):
        x, y = need(s.get("x"), "series.x"), need(s.get("y"), "series.y")
        lo, hi = s.get("ci95_low"), s.get("ci95_high")
        label = s.get("series_id", s.get("label", "series"))
        if lo is not None and hi is not None:
            err = np.vstack([np.asarray(y)-np.asarray(lo), np.asarray(hi)-np.asarray(y)])
            ax.errorbar(x, y, yerr=err, marker="o", capsize=2, label=label)
        else:
            ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel(xlabel or "registered x")
    ax.set_ylabel(ylabel or "registered y")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7)

def render(fig_id, spec, out):
    if fig_id == "fig_confusion_matrices":
        panels = need(spec.get("panels"), f"{fig_id}.panels")
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2*len(panels), 3.8), squeeze=False, constrained_layout=True)
        for ax, p in zip(axes[0], panels):
            matrix = np.asarray(need(p.get("matrix"), f"{fig_id}.matrix"), dtype=float)
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            labels = p.get("class_labels", list(range(matrix.shape[0])))
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels)
            ax.set_title(p.get("panel_id", "confusion")); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        fig.colorbar(im, ax=axes.ravel().tolist()); fig.savefig(out, bbox_inches="tight"); plt.close(fig); return
    if fig_id == "fig_calibration":
        panels = need(spec.get("panels"), f"{fig_id}.panels")
        fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
        ax.plot([0,1], [0,1], "k--", lw=1)
        for p in panels:
            ax.plot(need(p.get("confidence"), "confidence"), need(p.get("empirical_accuracy"), "empirical_accuracy"), marker="o", label=p.get("panel_id"))
        ax.set(xlabel="confidence", ylabel="recording accuracy", xlim=(0,1), ylim=(0,1)); ax.legend(fontsize=7); fig.savefig(out); plt.close(fig); return
    if fig_id == "fig_scars_selection_h2":
        panels = need(spec.get("panels"), f"{fig_id}.panels")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
        for p in panels:
            if p.get("panel_id") == "selection":
                x, y = np.asarray(p["source_instability"]), np.asarray(p["source_macro_f1"])
                cost = np.asarray(p["cost"], dtype=float); size = cost/np.max(cost)*80+10
                feasible, pareto, selected = np.asarray(p["feasible"], bool), np.asarray(p["pareto"], bool), np.asarray(p["selected"], bool)
                axes[0].scatter(x[~feasible], y[~feasible], s=size[~feasible], c="0.75", marker="x", label="infeasible")
                axes[0].scatter(x[feasible], y[feasible], s=size[feasible], c="tab:blue", alpha=.65, label="feasible")
                axes[0].scatter(x[pareto], y[pareto], s=size[pareto]+35, facecolors="none", edgecolors="tab:orange", linewidths=1.6, label="Pareto")
                axes[0].scatter(x[selected], y[selected], s=size[selected]+90, c="tab:red", marker="*", label="source selected")
                axes[0].legend(fontsize=7)
                axes[1].scatter(p["h2_residual_instability"], p["h2_residual_degradation"], alpha=.8)
            elif p.get("panel_id") == "h2":
                axes[1].scatter(p["h2_residual_instability"], p["h2_residual_degradation"], alpha=.8)
        axes[0].set(xlabel="source instability", ylabel="source macro-F1"); axes[1].set(xlabel="residual instability", ylabel="residual degradation")
        fig.savefig(out); plt.close(fig); return
    if fig_id == "fig_compute_tradeoff":
        series = need(spec.get("series"), f"{fig_id}.series")
        axes_names = spec.get("x_names", ["latency_ms", "macs", "bytes"])
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
        for ax, x_name in zip(axes, axes_names):
            matched = [s for s in series if s.get("x_name") == x_name or s.get("panel_id") == x_name]
            if not matched:
                raise ValueError(f"{fig_id}: missing axis {x_name}")
            draw_series(ax, matched, x_name, spec.get("y_name", "recording_macro_f1"))
        fig.savefig(out, bbox_inches="tight"); plt.close(fig); return
    if fig_id == "fig_routing_diagnostics":
        panels = need(spec.get("panels"), f"{fig_id}.panels")
        labels, values = [], []
        for p in panels:
            labels.append(f"{p['family']}:{p['nuisance']}:{p['severity']}"); values.append(p["weights"])
        fig, ax = plt.subplots(figsize=(max(6, .45*len(labels)), 3.8), constrained_layout=True)
        ax.boxplot(values, labels=labels, showfliers=False); ax.tick_params(axis="x", rotation=60); ax.set_ylabel("gate weight")
        fig.savefig(out); plt.close(fig); return
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    if "matrix" in spec:
        matrix = np.asarray(need(spec["matrix"], f"{fig_id}.matrix"), dtype=float)
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(spec.get("x_name", "column")); ax.set_ylabel(spec.get("y_name", "row"))
    else:
        series = spec.get("series")
        if series is None and spec.get("panels") is not None:
            series = spec["panels"]
        draw_series(ax, series, spec.get("x_name"), spec.get("y_name"))
    ax.set_title(fig_id.replace("fig_", "").replace("_", " "))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    subprocess.run([sys.executable, str(Path(__file__).with_name("validate_results.py")), str(args.results)], check=True)
    data = json.loads(args.results.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fig_id, filename in FILES.items():
        render(fig_id, need(data["figures"].get(fig_id), f"figures.{fig_id}"), args.output_dir / filename)

if __name__ == "__main__":
    main()
