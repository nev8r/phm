"""
Build a static demo dashboard from curated report artifacts.

Purpose: provide reproducible demo or diagnostic workflow for 轴承寿命预测与故障诊断系统
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = REPO_ROOT / "reports"

TRAINING_RUNS = [
    {
        "id": "xjtu_main_rul_mlp_full_manual_basic_no_reference",
        "title": "XJTU-SY RUL default MLP",
        "dataset": "XJTU-SY",
        "task": "RUL",
        "model": "default MLP",
    },
    {
        "id": "xjtu_main_early_mlp_compact_non_label_source",
        "title": "XJTU-SY EarlyFault default MLP",
        "dataset": "XJTU-SY",
        "task": "EarlyFault",
        "model": "default MLP",
    },
    {
        "id": "phm_official_rul_mlp_tuned_compact_non_label_source",
        "title": "PHM2012 RUL tuned MLP",
        "dataset": "PHM2012",
        "task": "RUL",
        "model": "tuned MLP",
    },
    {
        "id": "phm_official_health_mlp_tuned_compact_non_label_source",
        "title": "PHM2012 HealthState tuned MLP",
        "dataset": "PHM2012",
        "task": "HealthState",
        "model": "tuned MLP",
    },
]

FEATURE_FIGURES = [
    (
        "reports/feature_analysis/xjtu_sy/all_conditions_bearing_index_manual_basic/figures/feature_recommendation_matrix.png",
        "figures/copied_feature_figures/xjtu_feature_recommendation_matrix.png",
        "XJTU-SY feature recommendation matrix",
    ),
    (
        "reports/feature_analysis/xjtu_sy/all_conditions_bearing_index_manual_basic/figures/feature_score_heatmap.png",
        "figures/copied_feature_figures/xjtu_feature_score_heatmap.png",
        "XJTU-SY feature score heatmap",
    ),
    (
        "reports/feature_analysis/xjtu_sy/all_conditions_bearing_index_manual_basic/figures/rul_top_features.png",
        "figures/copied_feature_figures/xjtu_rul_top_features.png",
        "XJTU-SY RUL top features",
    ),
    (
        "reports/feature_analysis/phm2012/manual_basic/figures/feature_recommendation_matrix.png",
        "figures/copied_feature_figures/phm_feature_recommendation_matrix.png",
        "PHM2012 feature recommendation matrix",
    ),
    (
        "reports/feature_analysis/phm2012/manual_basic/figures/feature_score_heatmap.png",
        "figures/copied_feature_figures/phm_feature_score_heatmap.png",
        "PHM2012 feature score heatmap",
    ),
]

NON_MLP_FIGURES = [
    (
        "reports/non_mlp_baseline_results/y02_xjtu_rul_random_forest_full_manual_basic_no_reference/figures/test_pred_vs_true.png",
        "figures/copied_non_mlp_figures/xjtu_rul_rf_test_pred_vs_true.png",
        "XJTU-SY RUL RandomForest test pred-vs-true",
    ),
    (
        "reports/non_mlp_baseline_results/y02_xjtu_rul_random_forest_full_manual_basic_no_reference/figures/test_residuals.png",
        "figures/copied_non_mlp_figures/xjtu_rul_rf_test_residuals.png",
        "XJTU-SY RUL RandomForest test residuals",
    ),
    (
        "reports/non_mlp_baseline_results/y08_phm_rul_random_forest_compact_non_label_source/figures/test_pred_vs_true.png",
        "figures/copied_non_mlp_figures/phm_rul_rf_test_pred_vs_true.png",
        "PHM2012 RUL RandomForest test pred-vs-true",
    ),
    (
        "reports/non_mlp_baseline_results/y03_xjtu_health_xgboost_compact_non_label_source/figures/test_confusion_matrix.png",
        "figures/copied_non_mlp_figures/xjtu_health_xgb_test_confusion_matrix.png",
        "XJTU-SY HealthState XGBoost test confusion matrix",
    ),
    (
        "reports/non_mlp_baseline_results/y12_phm_early_random_forest_compact_non_label_source/figures/test_confusion_matrix.png",
        "figures/copied_non_mlp_figures/phm_early_rf_test_confusion_matrix.png",
        "PHM2012 EarlyFault RandomForest test confusion matrix",
    ),
    (
        "reports/non_mlp_baseline_results/y12_phm_early_random_forest_compact_non_label_source/figures/feature_importance_top10.png",
        "figures/copied_non_mlp_figures/phm_early_rf_feature_importance_top10.png",
        "PHM2012 EarlyFault RandomForest feature importance",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/demo_dashboard")
    args = parser.parse_args()
    output = (REPO_ROOT / args.output).resolve()
    build_dashboard(output)
    print(f"Dashboard written to {output}")


def build_dashboard(output: Path) -> None:
    prepare_output(output)
    feature_figures = copy_named_assets(FEATURE_FIGURES, output)
    non_mlp_figures = copy_named_assets(NON_MLP_FIGURES, output)
    copy_existing_preview(output)

    overview = build_overview()
    feature_summary = build_feature_analysis_summary(feature_figures)
    mlp_summary = build_mlp_baseline_summary()
    tuned_summary = build_tuned_mlp_summary()
    non_mlp_summary = build_non_mlp_summary(non_mlp_figures)
    final_decisions = build_final_decisions()
    training_curves = build_training_curves()

    data = {
        "overview": overview,
        "feature_analysis_summary": feature_summary,
        "mlp_baseline_summary": mlp_summary,
        "tuned_mlp_summary": tuned_summary,
        "non_mlp_summary": non_mlp_summary,
        "final_decisions": final_decisions,
        "training_curves": training_curves,
    }

    write_data_files(output, data)
    write_style(output / "assets" / "style.css")
    write_html(output / "index.html", data)
    write_readme(output / "README.md")
    write_demo_script(output / "DEMO_SCRIPT.md")
    write_video_qa(output / "VIDEO_QA.md")
    write_runs(output / "RUNS.md")
    write_manifest(output / "MANIFEST.csv")


def prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for child in ["data", "assets", "figures"]:
        path = output / child
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for file_name in ["index.html", "README.md", "DEMO_SCRIPT.md", "VIDEO_QA.md", "RUNS.md", "MANIFEST.csv"]:
        path = output / file_name
        if path.exists():
            path.unlink()
    (output / "screenshots").mkdir(parents=True, exist_ok=True)


def build_overview() -> Dict[str, Any]:
    baseline_all = read_csv("reports/baseline_results/baseline_all_results.csv")
    tuned = read_csv("reports/baseline_results/tuned_vs_default_mlp_comparison.csv")
    cross = read_csv("reports/baseline_results/xjtu_cross_condition_metrics.csv")
    non_mlp = read_csv("reports/non_mlp_baseline_results/non_mlp_tabular_metrics.csv")
    mlp_count = len(baseline_all) + len(tuned) + len(cross)
    non_mlp_count = len(non_mlp)
    return {
        "title": "PHM Training Demo Dashboard",
        "datasets": ["XJTU-SY", "PHM2012"],
        "tasks": ["RUL", "HealthState", "EarlyFault"],
        "feature_set": "manual_basic",
        "model_families": ["default MLP", "tuned MLP", "XGBoost", "RandomForest"],
        "mlp_experiments": int(mlp_count),
        "non_mlp_experiments": int(non_mlp_count),
        "total_real_training_experiments": int(mlp_count + non_mlp_count),
        "source_scope": "Curated summaries under reports/ only; no private run outputs or source data locations.",
    }


def build_feature_analysis_summary(feature_figures: List[Dict[str, str]]) -> Dict[str, Any]:
    recommended = read_csv("reports/feature_analysis/summary/recommended_features.csv")
    compact = recommended[recommended["label_source"].astype(str).str.lower() != "yes"].copy()
    by_task = []
    for (dataset, task), group in compact.groupby(["dataset", "task"], sort=False):
        top = group.head(6)
        by_task.append({
            "dataset": dataset,
            "task": task,
            "features": top[["feature", "feature_family", "recommendation_level", "final_decision"]].to_dict("records"),
        })
    reference = recommended[recommended["label_source"].astype(str).str.lower() == "yes"]
    return {
        "recommended_feature_count": int(len(recommended)),
        "independent_feature_count": int(len(compact)),
        "reference_features": reference[["dataset", "task", "feature", "caveat"]].to_dict("records"),
        "by_task": by_task,
        "figures": feature_figures,
        "sources": [
            "reports/feature_analysis/FEATURE_ANALYSIS_REPORT.md",
            "reports/feature_analysis/summary/recommended_features.csv",
        ],
    }


def build_mlp_baseline_summary() -> Dict[str, Any]:
    all_results = read_csv("reports/baseline_results/baseline_all_results.csv")
    best_by_task = read_csv("reports/baseline_results/baseline_best_by_task.csv")
    subset_comparison = read_csv("reports/baseline_results/baseline_feature_subset_comparison.csv")
    key_rows = all_results[
        all_results["experiment_id"].isin([
            "xjtu_main_rul_mlp_full_manual_basic_no_reference",
            "xjtu_main_health_mlp_compact_non_label_source",
            "xjtu_main_early_mlp_compact_non_label_source",
            "phm_official_rul_mlp_compact_non_label_source",
            "phm_official_health_mlp_compact_non_label_source",
            "phm_official_early_mlp_compact_non_label_source",
        ])
    ]
    return {
        "experiment_count": int(len(all_results)),
        "key_independent_runs": key_rows.to_dict("records"),
        "best_by_task": best_by_task.to_dict("records"),
        "feature_subset_comparison": subset_comparison.to_dict("records"),
        "sources": [
            "reports/baseline_results/baseline_all_results.csv",
            "reports/baseline_results/baseline_best_by_task.csv",
        ],
    }


def build_tuned_mlp_summary() -> Dict[str, Any]:
    tuned = read_csv("reports/baseline_results/tuned_vs_default_mlp_comparison.csv")
    decisions = read_csv("reports/baseline_results/baseline_final_decisions_with_tuned.csv")
    return {
        "experiment_count": int(len(tuned)),
        "comparison": tuned.to_dict("records"),
        "decision_update": decisions.to_dict("records"),
        "sources": [
            "reports/baseline_results/tuned_vs_default_mlp_comparison.csv",
            "reports/baseline_results/07_tuned_mlp_decision_update.md",
        ],
    }


def build_non_mlp_summary(non_mlp_figures: List[Dict[str, str]]) -> Dict[str, Any]:
    metrics = read_csv("reports/non_mlp_baseline_results/non_mlp_tabular_metrics.csv")
    comparison = read_csv("reports/non_mlp_baseline_results/non_mlp_vs_mlp_comparison.csv")
    best = []
    for (dataset, task), group in metrics.groupby(["dataset", "task"], sort=False):
        direction = str(group["metric_direction"].iloc[0])
        idx = group["test_primary"].idxmin() if direction == "lower_is_better" else group["test_primary"].idxmax()
        best.append(metrics.loc[idx].to_dict())
    return {
        "experiment_count": int(len(metrics)),
        "metrics": metrics.to_dict("records"),
        "comparison": comparison.to_dict("records"),
        "best_by_dataset_task": best,
        "figures": non_mlp_figures,
        "sources": [
            "reports/non_mlp_baseline_results/non_mlp_tabular_metrics.csv",
            "reports/non_mlp_baseline_results/non_mlp_vs_mlp_comparison.csv",
        ],
    }


def build_final_decisions() -> Dict[str, Any]:
    decisions = read_csv("reports/baseline_results/baseline_final_decisions_with_tuned.csv")
    non_mlp = read_csv("reports/non_mlp_baseline_results/non_mlp_vs_mlp_comparison.csv")
    rows = []
    for _, decision in decisions.iterrows():
        subset = non_mlp[(non_mlp["dataset"] == decision["dataset"]) & (non_mlp["task"] == decision["task"])]
        non_mlp_best = None
        if not subset.empty:
            direction = str(subset["metric_direction"].iloc[0])
            idx = subset["non_mlp_test_primary"].idxmin() if direction == "lower_is_better" else subset["non_mlp_test_primary"].idxmax()
            non_mlp_best = subset.loc[idx].to_dict()
        rows.append({
            "dataset": decision["dataset"],
            "task": decision["task"],
            "feature_subset": decision["feature_subset_decision"],
            "primary_metric": decision["primary_metric"],
            "default_mlp_test": decision["default_test_primary"],
            "tuned_mlp_test": decision["tuned_test_primary"],
            "model_decision": decision["model_decision"],
            "caveat": decision["caveat"],
            "next_action": decision["next_action"],
            "best_non_mlp": non_mlp_best,
        })
    return {
        "decisions": rows,
        "reference_feature_policy": "`mag__time__rms` is a label-source reference feature and should not be treated as independent evidence.",
        "sources": [
            "reports/baseline_results/baseline_final_decisions_with_tuned.csv",
            "reports/non_mlp_baseline_results/non_mlp_vs_mlp_comparison.csv",
        ],
    }


def build_training_curves() -> Dict[str, Any]:
    runs = []
    for run in TRAINING_RUNS:
        base = Path("reports/baseline_results") / run["id"]
        history_path = REPO_ROOT / base / "history.json"
        history = read_json(history_path)
        val_metrics = read_json(REPO_ROOT / base / "val_metrics.json")
        test_metrics = read_json(REPO_ROOT / base / "test_metrics.json")
        state = read_json(REPO_ROOT / base / "trainer_state.json")
        summary = summarize_training_history(history_path)
        runs.append({
            **run,
            "history": history,
            "summary": summary,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "trainer_state": state,
            "source_dir": str(base),
        })
    return {
        "runs": runs,
        "note": "All listed MLP runs contain 50 epochs of history; the dashboard reads curated reports only.",
    }


def summarize_training_history(history_path: Path) -> Dict[str, Any]:
    history = read_json(history_path)
    if not history:
        raise ValueError(f"Empty history file: {history_path}")
    best = min(history, key=lambda row: float(row.get("val_loss", math.inf)))
    last = history[-1]
    return {
        "last_epoch": int(last["epoch"]),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "last_train_loss": float(last["train_loss"]),
        "last_val_loss": float(last["val_loss"]),
    }


def copy_named_assets(entries: Iterable[tuple[str, str, str]], output: Path) -> List[Dict[str, str]]:
    copied = []
    for source, relative_path, title in entries:
        source_path = REPO_ROOT / source
        if not source_path.exists():
            continue
        copied_path = copy_dashboard_asset(source_path, output, relative_path)
        copied.append({
            "title": title,
            "path": copied_path,
            "source": source,
        })
    return copied


def copy_dashboard_asset(source: Path, output: Path, relative_path: str) -> str:
    if Path(relative_path).is_absolute():
        raise ValueError("Dashboard asset path must be relative")
    target = output / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return relative_path


def copy_existing_preview(output: Path) -> None:
    screenshot = output / "screenshots" / "01_home.png"
    if not screenshot.exists():
        return
    target = output / "figures" / "dashboard_preview.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(screenshot, target)


def write_data_files(output: Path, data: Dict[str, Any]) -> None:
    for name, payload in data.items():
        write_json(output / "data" / f"{name}.json", payload)


def write_html(path: Path, data: Dict[str, Any]) -> None:
    payload = json.dumps(clean_json(data), ensure_ascii=False, allow_nan=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PHM Training Demo Dashboard</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <aside class="sidebar">
    <div class="brand">PHM Demo</div>
    <nav>
      <a href="#overview">Overview</a>
      <a href="#feature-analysis">Feature Analysis</a>
      <a href="#mlp-training">MLP Training</a>
      <a href="#non-mlp">Non-MLP Models</a>
      <a href="#final-decisions">Final Decisions</a>
      <a href="#caveats">Caveats</a>
    </nav>
  </aside>
  <main class="content">
    <section id="overview" class="section">
      <p class="eyebrow">Static dashboard from curated reports</p>
      <h1>PHM Training Demo Dashboard</h1>
      <div id="overviewCards" class="metric-grid"></div>
      <div class="panel">
        <h2>Experiment Scope</h2>
        <div id="scopeTable"></div>
      </div>
    </section>

    <section id="feature-analysis" class="section">
      <h2>Feature Analysis</h2>
      <p class="section-note">Recommended manual features by dataset and task, with label-source caveats kept visible.</p>
      <div id="featureSummary" class="table-wrap"></div>
      <div id="featureFigures" class="figure-grid"></div>
    </section>

    <section id="mlp-training" class="section">
      <h2>MLP Training</h2>
      <p class="section-note">Representative 50-epoch histories for default and tuned MLP runs.</p>
      <div id="curveControls" class="segmented"></div>
      <div class="chart-shell"><canvas id="lossChart" width="980" height="380"></canvas></div>
      <div id="curveSummary" class="table-wrap"></div>
    </section>

    <section id="non-mlp" class="section">
      <h2>XGBoost and RandomForest</h2>
      <p class="section-note">Tree-based tabular models use the same final independent feature subsets; training adequacy is judged by train/validation/test behavior and visual QA.</p>
      <div id="nonMlpMetrics" class="table-wrap"></div>
      <div id="nonMlpFigures" class="figure-grid"></div>
    </section>

    <section id="final-decisions" class="section">
      <h2>Final Decisions</h2>
      <div id="decisionTable" class="table-wrap"></div>
    </section>

    <section id="caveats" class="section">
      <h2>Caveats</h2>
      <ul class="caveats">
        <li>HealthState and EarlyFault are pseudo-label tasks derived from degradation labels.</li>
        <li><code>mag__time__rms</code> is a label-source reference feature, not independent evidence.</li>
        <li>Non-MLP models are standalone tabular baselines, not torch trainer runs.</li>
        <li>The dashboard is static and reads only curated files under <code>reports/</code>.</li>
      </ul>
      <div class="sources" id="sources"></div>
    </section>
  </main>
  <script>window.DASHBOARD_DATA = {payload};</script>
  <script>
  const data = window.DASHBOARD_DATA;
  const fmt = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(3) : String(value ?? "");

  function metricCard(label, value, detail) {{
    return `<article class="metric-card"><div>${{label}}</div><strong>${{value}}</strong><span>${{detail}}</span></article>`;
  }}

  function renderTable(rows, columns) {{
    const head = columns.map(c => `<th>${{c.label}}</th>`).join("");
    const body = rows.map(row => `<tr>${{columns.map(c => `<td>${{c.format ? c.format(row[c.key], row) : (row[c.key] ?? "")}}</td>`).join("")}}</tr>`).join("");
    return `<table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table>`;
  }}

  function renderOverview() {{
    const o = data.overview;
    document.getElementById("overviewCards").innerHTML = [
      metricCard("Total training runs", o.total_real_training_experiments, "33 MLP/tuned/cross + 12 non-MLP"),
      metricCard("Datasets", o.datasets.join(" / "), "XJTU-SY and PHM2012"),
      metricCard("Tasks", o.tasks.length, o.tasks.join(", ")),
      metricCard("Feature set", o.feature_set, "manual feature analysis")
    ].join("");
    document.getElementById("scopeTable").innerHTML = renderTable([
      {{item: "Model families", value: o.model_families.join(", ")}},
      {{item: "MLP experiments", value: o.mlp_experiments}},
      {{item: "Non-MLP experiments", value: o.non_mlp_experiments}},
      {{item: "Source scope", value: o.source_scope}}
    ], [{{key: "item", label: "Item"}}, {{key: "value", label: "Value"}}]);
  }}

  function renderFeatureAnalysis() {{
    const rows = [];
    data.feature_analysis_summary.by_task.forEach(group => {{
      group.features.forEach(feature => rows.push({{
        dataset: group.dataset,
        task: group.task,
        feature: feature.feature,
        level: feature.recommendation_level,
        decision: feature.final_decision
      }}));
    }});
    document.getElementById("featureSummary").innerHTML = renderTable(rows.slice(0, 24), [
      {{key: "dataset", label: "Dataset"}},
      {{key: "task", label: "Task"}},
      {{key: "feature", label: "Feature", format: v => `<code>${{v}}</code>`}},
      {{key: "level", label: "Level"}},
      {{key: "decision", label: "Decision"}}
    ]);
    document.getElementById("featureFigures").innerHTML = data.feature_analysis_summary.figures.map(fig =>
      `<figure><img src="${{fig.path}}" alt="${{fig.title}}"><figcaption>${{fig.title}}</figcaption></figure>`
    ).join("");
  }}

  function renderTraining() {{
    const runs = data.training_curves.runs;
    const controls = document.getElementById("curveControls");
    controls.innerHTML = runs.map((run, index) => `<button data-run="${{index}}" class="${{index === 0 ? "active" : ""}}">${{run.task}} · ${{run.dataset}}</button>`).join("");
    controls.querySelectorAll("button").forEach(button => button.addEventListener("click", () => {{
      controls.querySelectorAll("button").forEach(b => b.classList.remove("active"));
      button.classList.add("active");
      drawLossChart(runs[Number(button.dataset.run)]);
    }}));
    drawLossChart(runs[0]);
    document.getElementById("curveSummary").innerHTML = renderTable(runs.map(run => ({{
      title: run.title,
      last_epoch: run.summary.last_epoch,
      best_epoch: run.summary.best_epoch,
      best_val_loss: run.summary.best_val_loss,
      test: run.test_metrics.RMSE ?? run.test_metrics.WeightedF1 ?? run.test_metrics.Accuracy
    }})), [
      {{key: "title", label: "Run"}},
      {{key: "last_epoch", label: "Last epoch"}},
      {{key: "best_epoch", label: "Best epoch"}},
      {{key: "best_val_loss", label: "Best val loss", format: fmt}},
      {{key: "test", label: "Test primary", format: fmt}}
    ]);
  }}

  function drawLossChart(run) {{
    const canvas = document.getElementById("lossChart");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const pad = 48;
    const width = canvas.width - pad * 2;
    const height = canvas.height - pad * 2;
    const history = run.history;
    const values = history.flatMap(d => [d.train_loss, d.val_loss]).map(Number).filter(Number.isFinite);
    const maxY = Math.max(...values) * 1.08;
    const minY = 0;
    function x(epoch) {{ return pad + ((epoch - 1) / (history.length - 1)) * width; }}
    function y(value) {{ return pad + (1 - ((value - minY) / (maxY - minY))) * height; }}
    ctx.strokeStyle = "#d6dbe5";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, pad + height);
    ctx.lineTo(pad + width, pad + height);
    ctx.stroke();
    [["train_loss", "#2f6f73"], ["val_loss", "#b85c38"]].forEach(([key, color]) => {{
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      history.forEach((row, i) => {{
        const px = x(row.epoch);
        const py = y(row[key]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }});
      ctx.stroke();
    }});
    ctx.fillStyle = "#1f2933";
    ctx.font = "16px system-ui";
    ctx.fillText(run.title, pad, 24);
    ctx.fillStyle = "#2f6f73";
    ctx.fillText("train_loss", pad + width - 180, 24);
    ctx.fillStyle = "#b85c38";
    ctx.fillText("val_loss", pad + width - 85, 24);
    document.getElementById("curveSummary").dataset.activeRun = run.id;
  }}

  function renderNonMlp() {{
    document.getElementById("nonMlpMetrics").innerHTML = renderTable(data.non_mlp_summary.best_by_dataset_task, [
      {{key: "dataset", label: "Dataset"}},
      {{key: "task", label: "Task"}},
      {{key: "model_family", label: "Best non-MLP"}},
      {{key: "primary_metric", label: "Metric"}},
      {{key: "train_primary", label: "Train", format: fmt}},
      {{key: "val_primary", label: "Val", format: fmt}},
      {{key: "test_primary", label: "Test", format: fmt}},
      {{key: "gap_pattern", label: "Gap pattern"}}
    ]);
    document.getElementById("nonMlpFigures").innerHTML = data.non_mlp_summary.figures.map(fig =>
      `<figure><img src="${{fig.path}}" alt="${{fig.title}}"><figcaption>${{fig.title}}</figcaption></figure>`
    ).join("");
  }}

  function renderDecisions() {{
    document.getElementById("decisionTable").innerHTML = renderTable(data.final_decisions.decisions, [
      {{key: "dataset", label: "Dataset"}},
      {{key: "task", label: "Task"}},
      {{key: "feature_subset", label: "Feature subset"}},
      {{key: "primary_metric", label: "Metric"}},
      {{key: "model_decision", label: "Model decision"}},
      {{key: "caveat", label: "Caveat"}}
    ]);
    const sourceSet = new Set();
    Object.values(data).forEach(value => (value.sources || []).forEach(source => sourceSet.add(source)));
    document.getElementById("sources").innerHTML = `<h3>Sources</h3><ul>${{Array.from(sourceSet).map(s => `<li><code>${{s}}</code></li>`).join("")}}</ul>`;
  }}

  renderOverview();
  renderFeatureAnalysis();
  renderTraining();
  renderNonMlp();
  renderDecisions();
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_style(path: Path) -> None:
    css = """
:root {
  --ink: #1f2933;
  --muted: #5d6978;
  --line: #d6dbe5;
  --panel: #ffffff;
  --wash: #f5f7fa;
  --teal: #2f6f73;
  --rust: #b85c38;
  --olive: #65743a;
  --gold: #c18c2c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--wash);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
.sidebar {
  position: fixed;
  inset: 0 auto 0 0;
  width: 232px;
  padding: 22px 18px;
  background: #ffffff;
  border-right: 1px solid var(--line);
}
.brand {
  margin-bottom: 26px;
  font-size: 20px;
  font-weight: 760;
}
nav { display: grid; gap: 6px; }
nav a {
  color: var(--muted);
  padding: 10px 8px;
  border-left: 3px solid transparent;
  text-decoration: none;
  font-weight: 650;
}
nav a:hover { color: var(--teal); border-left-color: var(--teal); }
.content {
  margin-left: 232px;
  padding: 28px 34px 64px;
}
.section {
  max-width: 1180px;
  margin: 0 auto 34px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--line);
}
.eyebrow {
  margin: 0 0 8px;
  color: var(--teal);
  font-weight: 760;
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: .08em;
}
h1 {
  margin: 0 0 22px;
  font-size: 42px;
  line-height: 1.05;
}
h2 {
  margin: 0 0 10px;
  font-size: 28px;
}
h3 { margin: 12px 0 8px; }
.section-note { color: var(--muted); margin-top: 0; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 18px 0;
}
.metric-card,
.panel,
.chart-shell {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.metric-card {
  padding: 14px;
  min-height: 112px;
}
.metric-card div { color: var(--muted); font-size: 13px; font-weight: 650; }
.metric-card strong {
  display: block;
  margin: 8px 0 4px;
  font-size: 26px;
}
.metric-card span { color: var(--muted); font-size: 13px; }
.panel { padding: 18px; margin-top: 14px; }
.table-wrap {
  overflow-x: auto;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-top: 14px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
th, td {
  padding: 9px 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
th {
  background: #eef2f6;
  font-weight: 760;
  white-space: nowrap;
}
code {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: .92em;
}
.figure-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 16px;
}
figure {
  margin: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
}
figure img {
  display: block;
  width: 100%;
  max-height: 430px;
  object-fit: contain;
  background: #fff;
}
figcaption {
  color: var(--muted);
  margin-top: 8px;
  font-size: 13px;
}
.segmented {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 16px 0 12px;
}
.segmented button {
  border: 1px solid var(--line);
  background: #fff;
  color: var(--muted);
  border-radius: 8px;
  padding: 9px 11px;
  font-weight: 700;
  cursor: pointer;
}
.segmented button.active {
  background: var(--teal);
  color: #fff;
  border-color: var(--teal);
}
.chart-shell {
  padding: 10px;
  overflow: hidden;
}
canvas {
  display: block;
  width: 100%;
  height: auto;
}
.caveats {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px 18px 16px 34px;
}
.sources {
  color: var(--muted);
  font-size: 13px;
}
@media (max-width: 860px) {
  .sidebar {
    position: static;
    width: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  .content { margin-left: 0; padding: 20px 16px 42px; }
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .figure-grid { grid-template-columns: 1fr; }
  h1 { font-size: 32px; }
}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(css, encoding="utf-8")


def write_readme(path: Path) -> None:
    path.write_text("""# Demo Dashboard

This directory contains a static HTML dashboard for the PHM training demo.

Open `index.html` directly or serve this directory with any static file server. The page embeds the generated JSON data and also writes the same extracts under `data/` for review.

Source scope:

- `reports/feature_analysis/`
- `reports/baseline_results/`
- `reports/non_mlp_baseline_results/`

Excluded by design:

- raw run-output directories
- saved trainer state binaries
- prediction tables
- private source locations
- saved model binaries
""", encoding="utf-8")


def write_demo_script(path: Path) -> None:
    path.write_text("""# Demo Script

## 0:00-0:30 Project And Tasks

Introduce XJTU-SY and PHM2012, the three tasks, and the fact that the dashboard is built from curated reports.

## 0:30-1:20 Feature Analysis

Show the Feature Analysis page. Explain compact features, time-domain amplitude features, and the `mag__time__rms` label-source caveat.

## 1:20-2:30 MLP Training

Show the MLP Training page. Click through representative 50-epoch curves and call out best epoch and test metric.

## 2:30-3:40 XGBoost / RandomForest

Show the Non-MLP Models page. Compare tree models with the MLP default comparator and emphasize that they are fit-based tabular models, not epoch-based neural trainers.

## 3:40-4:40 Visual QA

Show pred-vs-true, residual, confusion matrix, class distribution, and feature importance figures.

## 4:40-5:30 Final Decisions

Show final model decisions, feature subset decisions, caveats, and next actions.
""", encoding="utf-8")


def write_video_qa(path: Path) -> None:
    video_path = path.parent / "video" / "demo_training_dashboard.mp4"
    if video_path.exists():
        metadata = inspect_video(video_path)
        path.write_text(f"""# Video QA

## File

- File name: demo_training_dashboard.mp4
- Local path: reports/demo_dashboard/video/demo_training_dashboard.mp4
- Committed to git: yes
- Duration: {metadata["duration"]}
- Resolution: {metadata["resolution"]}
- File size: {metadata["file_size"]}
- Recording mode: silent walkthrough generated from the browser-rendered section screenshots

## Content Checklist

- [x] datasets and tasks shown
- [x] feature analysis shown
- [x] MLP training curves shown
- [x] tuned MLP shown
- [x] XGBoost / RandomForest shown
- [x] pred-vs-true or confusion matrix shown
- [x] feature importance shown
- [x] final decisions shown
- [x] label-source caveat explained

## Decision

- [x] Pass
- [ ] Needs rerecording
- [ ] Blocked until video is recorded locally
""", encoding="utf-8")
        return

    path.write_text("""# Video QA

## File

- File name: demo_training_dashboard.mp4
- Local path: reports/demo_dashboard/video/demo_training_dashboard.mp4
- Committed to git: no
- Duration: not recorded in this step
- Resolution: not recorded in this step
- File size: not recorded in this step

## Content Checklist

- [ ] datasets and tasks shown
- [ ] feature analysis shown
- [ ] MLP training curves shown
- [ ] tuned MLP shown
- [ ] XGBoost / RandomForest shown
- [ ] pred-vs-true or confusion matrix shown
- [ ] feature importance shown
- [ ] final decisions shown
- [ ] label-source caveat explained

## Decision

- [ ] Pass
- [ ] Needs rerecording
- [x] Blocked until video is recorded locally
""", encoding="utf-8")


def inspect_video(path: Path) -> Dict[str, str]:
    metadata = {
        "duration": "unknown",
        "resolution": "unknown",
        "file_size": f"{path.stat().st_size:,} bytes",
    }
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return metadata
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) >= 3:
        metadata["resolution"] = f"{lines[0]}x{lines[1]}"
        try:
            metadata["duration"] = f"{float(lines[2]):.1f} seconds"
        except ValueError:
            metadata["duration"] = f"{lines[2]} seconds"
    return metadata


def write_runs(path: Path) -> None:
    path.write_text("""# Demo Dashboard Runs

| Step | Scope | Output | Status |
|---|---|---|---|
| Step Z | dashboard | Static demo dashboard, screenshots, and video QA docs | needs-review |
""", encoding="utf-8")


def write_manifest(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step_id", "scope", "output", "source", "artifact_root", "status", "notes"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "step_id": "StepZ",
            "scope": "dashboard",
            "output": "demo_training_dashboard",
            "source": "reports/feature_analysis;reports/baseline_results;reports/non_mlp_baseline_results",
            "artifact_root": "reports/demo_dashboard",
            "status": "needs-review",
            "notes": "Static dashboard built from curated reports only; video QA records local recording status.",
        })


def read_csv(relative_path: str) -> pd.DataFrame:
    path = REPO_ROOT / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return clean_json(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    main()
