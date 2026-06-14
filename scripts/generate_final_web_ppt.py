"""
Generate the final defense web deck with the Guizang Swiss PPT template.

The generated artifact is a single-file HTML slide deck. It also creates two
small evidence charts from local real bearing data when the downloaded datasets
are available.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from USTC.SSE.BearingPrediction.dataset.phm2012 import PHM2012Loader
from USTC.SSE.BearingPrediction.dataset.xjtu import XJTULoader
from USTC.SSE.BearingPrediction.feature.engineering import FeatureConfig, SignalFeatureExtractor


PROJECT_TITLE = "工业轴承设备剩余寿命预测系统的实现"
TOTAL_SLIDES = 16
SKILL_ROOT = Path(".agents/skills/guizang-ppt-skill")
SKILL_TEMPLATE = SKILL_ROOT / "assets/template-swiss.html"
SKILL_MOTION = SKILL_ROOT / "assets/motion.min.js"
OUTPUT_PATH = Path("docx/final/web-ppt/index.html")
IMAGE_DIR = OUTPUT_PATH.parent / "images"
MOTION_OUTPUT = OUTPUT_PATH.parent / "assets/motion.min.js"
XJTU_ROOT = Path("data/external/xjtu/extracted/XJTU-SY_Bearing_Datasets")
PHM2012_TRAINING_ROOT = Path("data/external/phm2012/final/Training_set")


CUSTOM_CSS = """
  /* Course-defense additions: restrained Swiss information design. */
  .course-cover-copy{flex:1;display:grid;grid-template-rows:auto 1fr auto;gap:2.8vh}
  .course-cover-title{align-self:center;font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(7.1vw,12.8vh);line-height:1.04;letter-spacing:-.025em;color:#fff;max-width:11em}
  .course-cover-title span{font-style:italic;font-weight:300}
  .course-cover-meta{display:grid;grid-template-columns:repeat(4,1fr);gap:1.2vw;border-top:1px solid rgba(255,255,255,.24);padding-top:2vh;color:rgba(255,255,255,.84)}
  .course-cover-meta div{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.92vw);line-height:1.45;font-weight:400}
  .course-cover-meta b{display:block;font-family:var(--mono);font-size:14px;letter-spacing:.16em;text-transform:uppercase;color:rgba(255,255,255,.62);font-weight:600;margin-bottom:.6vh}
  .head-stack{display:flex;flex-direction:column;gap:1.35vh}
  .page-title{font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.65vw,10vh);line-height:.98;letter-spacing:-.035em}
  .page-title.small{font-size:min(5.1vw,8.8vh)}
  .course-grid-6{display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(2,1fr);gap:1px;background:var(--border-subtle);margin-top:5vh;min-height:48vh}
  .course-cell{background:var(--paper);padding:2.1vh 1.4vw;display:flex;flex-direction:column;justify-content:space-between}
  .course-cell.accent{background:var(--accent);color:var(--accent-on)}
  .course-cell.ink{background:var(--ink);color:var(--paper)}
  .course-cell .num{font-family:var(--mono);font-size:14px;letter-spacing:.16em;opacity:.62;font-weight:600}
  .course-cell .ttl{font-family:var(--sans),var(--sans-zh);font-size:max(20px,1.42vw);line-height:1.14;font-weight:400;letter-spacing:-.012em}
  .course-cell .desc{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.94vw);line-height:1.48;font-weight:400;color:inherit;opacity:.72}
  .dataset-facts{display:grid;grid-template-columns:repeat(2,1fr);gap:1vw;margin-top:2.4vh}
  .fact-card{background:var(--paper);border-top:2px solid currentColor;padding:1.45vh 1vw}
  .fact-card .num{font-family:var(--sans);font-size:min(3.2vw,5.5vh);font-weight:200;line-height:.95;letter-spacing:-.035em}
  .fact-card .txt{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.9vw);line-height:1.45;color:var(--text-secondary);margin-top:.8vh}
  .evidence-caption{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.94vw);line-height:1.5;color:var(--text-secondary)}
  .image-hero-body.course{display:grid;grid-template-columns:1.05fr 1.35fr;gap:4vw;align-items:stretch;padding:3.4vh 5vw 7.6vh}
  .image-hero-stats.course{display:grid;grid-template-columns:repeat(3,1fr);gap:2vw}
  .course-stat{display:flex;flex-direction:column;gap:.7vh;border-top:1px solid var(--ink);padding-top:1vh}
  .course-stat .k{font-family:var(--mono);font-size:14px;letter-spacing:.14em;text-transform:uppercase;color:var(--text-helper);font-weight:600}
  .course-stat .v{font-family:var(--sans);font-weight:200;font-size:min(4.6vw,7.5vh);line-height:.96;letter-spacing:-.04em}
  .course-stat .v.accent{color:var(--accent)}
  .course-stat .d{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.9vw);line-height:1.45;color:var(--text-secondary);margin-top:auto}
  .feature-board{display:grid;grid-template-columns:1.05fr 1fr;gap:2vw;flex:1;margin-top:4.6vh}
  .feature-list{display:grid;grid-template-columns:repeat(2,1fr);gap:1vw}
  .feature-card{background:var(--grey-1);padding:2vh 1.25vw;border-top:2px solid var(--ink);display:flex;flex-direction:column}
  .feature-card.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .feature-card .ttl{font-family:var(--sans),var(--sans-zh);font-size:max(20px,1.35vw);font-weight:400;line-height:1.18}
  .feature-card .desc{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.94vw);line-height:1.48;opacity:.76;margin-top:1.1vh}
  .feature-compact{background:var(--grey-1);padding:2.3vh 1.6vw;border-top:2px solid var(--accent);display:flex;flex-direction:column}
  .feature-compact .mono{font-family:var(--mono);font-size:max(14px,.78vw);line-height:1.65;color:var(--text-secondary);word-break:break-word;font-weight:500}
  .rul-flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:1vw;align-items:stretch;margin-top:4.6vh;flex:1}
  .rul-step{background:var(--grey-1);padding:2vh 1.1vw;display:flex;flex-direction:column;min-height:0;border-top:2px solid var(--ink)}
  .rul-step.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .rul-step .num{font-family:var(--mono);font-size:14px;letter-spacing:.14em;opacity:.65;font-weight:600;margin-bottom:auto}
  .rul-step .ttl{font-family:var(--sans),var(--sans-zh);font-size:max(18px,1.18vw);line-height:1.15;font-weight:400;letter-spacing:-.012em;margin-top:3vh}
  .rul-step .txt{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.86vw);line-height:1.45;opacity:.75;margin-top:1vh}
  .architecture-board{display:grid;grid-template-columns:1.1fr .9fr;gap:3vw;align-items:stretch;margin-top:4.8vh;flex:1}
  .architecture-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border-subtle);min-height:46vh}
  .architecture-layer{background:var(--paper);padding:2.2vh 1.25vw;display:flex;flex-direction:column;justify-content:space-between;border-top:2px solid var(--ink)}
  .architecture-layer.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .architecture-layer .num{font-family:var(--mono);font-size:14px;letter-spacing:.14em;text-transform:uppercase;opacity:.62;font-weight:600}
  .architecture-layer .ttl{font-family:var(--sans),var(--sans-zh);font-size:max(21px,1.42vw);line-height:1.15;font-weight:400;letter-spacing:-.012em}
  .architecture-layer .desc{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.9vw);line-height:1.45;opacity:.76}
  .architecture-side{display:flex;flex-direction:column;gap:1.4vh}
  .architecture-note{background:var(--grey-1);border-top:2px solid var(--ink);padding:2.1vh 1.4vw}
  .architecture-note.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .architecture-note h3{font-family:var(--sans),var(--sans-zh);font-size:max(20px,1.35vw);line-height:1.2;font-weight:400;margin-bottom:.8vh}
  .architecture-note p{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.92vw);line-height:1.48;opacity:.76}
  .difficulty-grid{display:grid;grid-template-columns:1fr 1px 1fr;gap:2.8vw;margin-top:4.8vh;align-items:stretch;flex:1}
  .difficulty-col{display:flex;flex-direction:column;gap:2vh}
  .difficulty-card{background:var(--grey-1);padding:2.2vh 1.5vw;border-top:2px solid var(--ink)}
  .difficulty-card.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .difficulty-card h3{font-family:var(--sans),var(--sans-zh);font-weight:400;font-size:max(20px,1.5vw);line-height:1.18;letter-spacing:-.012em;margin-bottom:1vh}
  .difficulty-card p{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.94vw);line-height:1.5;opacity:.76}
  .sequence-table{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--border-subtle);margin-top:4.5vh;min-height:38vh}
  .sequence-cell{background:var(--paper);padding:2vh 1vw;display:flex;flex-direction:column;justify-content:space-between}
  .sequence-cell.accent{background:var(--accent);color:var(--accent-on)}
  .sequence-cell .top{font-family:var(--mono);font-size:14px;letter-spacing:.14em;opacity:.62;font-weight:600}
  .sequence-cell .main{font-family:var(--sans),var(--sans-zh);font-size:max(20px,1.36vw);line-height:1.18;font-weight:400}
  .sequence-cell .sub{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.88vw);line-height:1.45;opacity:.72}
  .spec-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.4vw;align-items:stretch;margin-top:4.6vh;flex:1}
  .spec-card{background:var(--grey-1);padding:2.4vh 1.5vw;display:flex;flex-direction:column;border-top:2px solid var(--ink)}
  .spec-card.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .spec-card .big{font-family:var(--sans);font-size:min(4.7vw,8vh);font-weight:200;line-height:.96;letter-spacing:-.04em}
  .spec-card .label{font-family:var(--mono);font-size:14px;letter-spacing:.14em;text-transform:uppercase;opacity:.65;margin-bottom:2vh;font-weight:600}
  .spec-card .desc{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.92vw);line-height:1.5;opacity:.78;margin-top:auto}
  .mini-ledger{display:grid;gap:0;margin-top:2vh;border-top:1px solid var(--border-subtle)}
  .mini-ledger div{display:grid;grid-template-columns:8em 1fr;gap:1vw;padding:1.05vh 0;border-bottom:1px solid var(--border-subtle)}
  .mini-ledger b{font-family:var(--mono);font-size:14px;letter-spacing:.12em;text-transform:uppercase;font-weight:600}
  .mini-ledger span{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.9vw);line-height:1.45;color:var(--text-secondary)}
  .result-table{display:grid;grid-template-columns:1.1fr .9fr .9fr .9fr .9fr;gap:1px;background:var(--border-subtle);margin-top:3vh}
  .result-table div{background:var(--paper);padding:1.35vh .9vw;font-family:var(--sans),var(--sans-zh);font-size:max(16px,.88vw);line-height:1.4}
  .result-table .head{font-family:var(--mono);font-size:14px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-helper);font-weight:600}
  .result-table .accent{background:var(--accent);color:var(--accent-on)}
  .figure-board{display:grid;grid-template-columns:1.15fr .85fr;gap:2.4vw;align-items:stretch;flex:1;margin-top:4.2vh}
  .figure-frame{background:var(--paper);border-top:2px solid var(--ink);padding:1.2vh 1vw;display:flex;align-items:center;justify-content:center;min-height:0}
  .figure-frame img{width:100%;height:100%;max-height:52vh;object-fit:contain}
  .figure-notes{display:grid;grid-template-rows:repeat(3,1fr);gap:1.2vh}
  .figure-note{background:var(--grey-1);border-top:2px solid var(--ink);padding:2vh 1.3vw;display:flex;flex-direction:column;justify-content:space-between}
  .figure-note.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .figure-note h3{font-family:var(--sans),var(--sans-zh);font-size:max(20px,1.35vw);line-height:1.18;font-weight:400;letter-spacing:-.012em}
  .figure-note p{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.9vw);line-height:1.45;opacity:.76}
  .paper-choice-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1.8vw;flex:1;margin-top:4.6vh}
  .paper-card{background:var(--grey-1);border-top:2px solid var(--ink);padding:2.5vh 1.7vw;display:flex;flex-direction:column;justify-content:space-between}
  .paper-card.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .paper-card .year{font-family:var(--mono);font-size:14px;letter-spacing:.16em;text-transform:uppercase;opacity:.66;font-weight:600}
  .paper-card h3{font-family:var(--sans),var(--sans-zh);font-size:max(26px,2vw);font-weight:300;line-height:1.1;letter-spacing:-.018em}
  .paper-card p{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.96vw);line-height:1.5;opacity:.78}
  .metric-board{display:grid;grid-template-columns:.9fr 1.1fr;gap:2.2vw;align-items:stretch;margin-top:3.6vh;flex:1}
  .metric-cards{display:grid;grid-template-rows:repeat(4,1fr);gap:1vh}
  .metric-card{background:var(--grey-1);border-top:2px solid var(--ink);padding:1.6vh 1.2vw}
  .metric-card.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .metric-card h3{font-family:var(--sans),var(--sans-zh);font-size:max(18px,1.22vw);font-weight:400;line-height:1.16;margin-bottom:.6vh}
  .metric-card p{font-family:var(--sans),var(--sans-zh);font-size:max(15px,.86vw);line-height:1.42;opacity:.76}
  .ledger-list{display:flex;flex-direction:column;flex:1;justify-content:center}
  .ledger-row.course{display:grid;grid-template-columns:minmax(12vw,17vw) 1fr 8vw;gap:2vw;align-items:center;padding:2vh 0;border-bottom:1px solid var(--border-subtle)}
  .ledger-row.course .ledger-num{font-family:var(--sans);font-weight:200;font-size:min(7.4vw,11.6vh);line-height:.92;letter-spacing:-.04em}
  .closing-list{display:flex;flex-direction:column;gap:0}
  .closing-item{display:grid;grid-template-columns:auto 1fr;gap:2vw;align-items:start;padding:2.1vh 0;border-top:1px solid var(--border-subtle)}
  .closing-item:last-child{border-bottom:2px solid var(--accent)}
  .closing-item .n{font-family:var(--sans);font-weight:200;font-size:min(3.8vw,6.8vh);line-height:.9;color:var(--text-primary)}
  .closing-item.accent .n,.closing-item.accent h3{color:var(--accent)}
  .closing-item h3{font-family:var(--sans),var(--sans-zh);font-weight:400;font-size:max(20px,1.42vw);line-height:1.2;letter-spacing:-.012em;margin-bottom:.8vh}
  .closing-item p{font-family:var(--sans),var(--sans-zh);font-size:max(16px,.92vw);line-height:1.5;color:var(--text-secondary)}
"""


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values) + 1e-8)


def _extract_feature_summary(entity, *, channel_name: str) -> pd.DataFrame:
    extractor = SignalFeatureExtractor(FeatureConfig(sample_rate=entity.sample_rate))
    features = extractor.extract(list(entity.samples[channel_name]))
    summary = pd.DataFrame(
        {
            "elapsed_seconds": entity.samples["elapsed_seconds"].astype(float).to_numpy(),
            "rms": features["rms"].to_numpy(dtype=float),
            "peak": features["peak"].to_numpy(dtype=float),
            "kurtosis": features["kurtosis"].to_numpy(dtype=float),
            "health": extractor.build_health_indicator(features),
        }
    )
    return summary


def _plot_normalized_curve(summary: pd.DataFrame, *, time_unit: str, output_path: Path, source_note: str) -> None:
    if time_unit == "min":
        x_values = summary["elapsed_seconds"].to_numpy(dtype=float) / 60.0
        xlabel = "Elapsed time (min)"
    elif time_unit == "hour":
        x_values = summary["elapsed_seconds"].to_numpy(dtype=float) / 3600.0
        xlabel = "Elapsed time (hour)"
    else:
        x_values = summary["elapsed_seconds"].to_numpy(dtype=float)
        xlabel = "Elapsed time (s)"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#0a0a0a",
            "axes.labelcolor": "#0a0a0a",
            "xtick.color": "#525252",
            "ytick.color": "#525252",
        }
    )
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#fafaf8")
    ax.set_facecolor("#fafaf8")
    ax.plot(x_values, _normalize(summary["rms"]), color="#002FA7", linewidth=2.6, label="RMS")
    ax.plot(x_values, _normalize(summary["peak"]), color="#0a0a0a", linewidth=2.0, label="Peak")
    ax.plot(x_values, summary["health"], color="#737373", linewidth=2.0, linestyle="--", label="Health indicator")
    ax.fill_between(x_values, _normalize(summary["rms"]), color="#002FA7", alpha=0.08)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Normalized value", fontsize=11)
    ax.grid(True, axis="y", color="#d4d4d2", linewidth=0.8, alpha=0.75)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=10)
    ax.text(
        0.995,
        0.03,
        source_note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#737373",
    )
    fig.tight_layout(pad=1.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_fallback_chart(output_path: Path, message: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#fafaf8")
    ax.set_facecolor("#fafaf8")
    ax.axis("off")
    ax.text(0.05, 0.56, "Local evidence chart", fontsize=30, fontweight=200, color="#0a0a0a")
    ax.text(0.05, 0.42, message, fontsize=14, color="#525252")
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_evidence_assets() -> None:
    """
    Generate data evidence images used by the final deck.
    """
    owner_asset_dir = Path("docs/project-owner/assets")
    for source_name, deck_name in {
        "multi-bearing-feature-summary.png": "07-multi-bearing-feature-summary.png",
        "end-to-end-rul-architecture.png": "09-end-to-end-rul-architecture.png",
    }.items():
        source_path = owner_asset_dir / source_name
        target_path = IMAGE_DIR / deck_name
        if source_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)

    try:
        xjtu_entity = XJTULoader(XJTU_ROOT).load_entity("Bearing1_1", max_samples=80)
        xjtu_summary = _extract_feature_summary(xjtu_entity, channel_name="Horizontal Vibration")
        _plot_normalized_curve(
            xjtu_summary,
            time_unit="min",
            output_path=IMAGE_DIR / "05-xjtu-bearing1-1-rms-health.png",
            source_note="XJTU-SY Bearing1_1 · 35Hz12kN · Horizontal · 80 / 123 snapshots",
        )
    except Exception as exc:  # pragma: no cover - local data fallback
        _write_fallback_chart(IMAGE_DIR / "05-xjtu-bearing1-1-rms-health.png", f"XJTU-SY local data was not available: {exc}")

    try:
        phm_entity = PHM2012Loader(PHM2012_TRAINING_ROOT).load_entity("Bearing1_1", max_samples=80)
        phm_summary = _extract_feature_summary(phm_entity, channel_name="Horizontal Vibration")
        _plot_normalized_curve(
            phm_summary,
            time_unit="hour",
            output_path=IMAGE_DIR / "06-phm2012-bearing1-1-rms-health.png",
            source_note="PHM2012 Bearing1_1 · Condition 1 · Horizontal · 80 / 2803 snapshots",
        )
    except Exception as exc:  # pragma: no cover - local data fallback
        _write_fallback_chart(IMAGE_DIR / "06-phm2012-bearing1-1-rms-health.png", f"PHM2012 local data was not available: {exc}")


SLIDES = f"""
<section class="slide accent" data-layout="S01" data-animate="hero">
  <div class="canvas-card">
    <canvas class="ascii-bg" aria-hidden="true"></canvas>
    <div class="chrome-min">
      <div class="l">中国科学技术大学软件学院 · 软件工程课程</div>
      <div class="r">结题答辩 · 01 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="course-cover-copy" data-anim="up">
      <div class="t-meta" style="color:rgba(255,255,255,.72);letter-spacing:.22em">剩余寿命预测 · 结题答辩</div>
      <h1 class="course-cover-title">工业轴承设备<br/>剩余寿命预测系统<br/>的<span>实现</span></h1>
      <div class="course-cover-meta">
        <div><b>课程</b>软件工程</div>
        <div><b>指导老师</b>zjf</div>
        <div><b>小组成员</b>zyj / cyy / zdh / zy</div>
        <div><b>日期</b>2026 年 6 月</div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S04" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">项目目标与交付物</div>
      <div class="r">02 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="head-stack" data-anim="line">
      <div class="t-meta">本次答辩重点放在数据理解、工程实现和验收证据</div>
      <h2 class="page-title">把真实轴承振动数据转成可复现的 RUL 预测流程。</h2>
    </div>
    <div class="course-grid-6" data-anim="up">
      <article class="course-cell accent"><div class="num">01</div><div class="ttl">任务定义</div><p class="desc">估计设备还能稳定运行多久，输出连续剩余寿命。</p></article>
      <article class="course-cell"><div class="num">02</div><div class="ttl">两个数据集</div><p class="desc">XJTU-SY 与 PHM2012，覆盖不同工况、快照长度和时间间隔。</p></article>
      <article class="course-cell"><div class="num">03</div><div class="ttl">特征分析</div><p class="desc">19 维时频域特征，解释振动强度、冲击变化和频谱变化。</p></article>
      <article class="course-cell"><div class="num">04</div><div class="ttl">系统实现</div><p class="desc">数据、特征、标签、模型、训练、评价分层，核心逻辑在 src 包内。</p></article>
      <article class="course-cell"><div class="num">05</div><div class="ttl">复现实验</div><p class="desc">CNN-LSTM-AM 与 xLSTM-Transformer 均读取真实数据训练。</p></article>
      <article class="course-cell ink"><div class="num">06</div><div class="ttl">测试交付</div><p class="desc">31 个自动化测试、8 个 notebook 示例和结题文档材料。</p></article>
    </div>
  </div>
</section>

<section class="slide split" data-layout="S03" data-animate="split-statement">
  <div class="canvas-card">
    <div class="split-half">
      <div class="half b-accent" style="justify-content:space-between;position:relative;overflow:hidden">
        <div class="chrome-min" style="position:relative;z-index:1">
          <div class="l">任务定义</div>
          <div class="r">03 / {TOTAL_SLIDES:02d}</div>
        </div>
        <h2 data-anim="manifesto" style="position:relative;z-index:1;font-family:var(--sans),var(--sans-zh);font-size:min(7.2vw,13vh);line-height:1;letter-spacing:-.025em;font-weight:200;color:#fff">为什么是 RUL？</h2>
        <div class="t-meta" style="position:relative;z-index:1;color:rgba(255,255,255,.72)">Remaining Useful Life</div>
      </div>
      <div class="half b-grey" style="justify-content:center">
        <div data-anim="rules" style="display:flex;flex-direction:column;gap:3vh">
          <p class="lead" style="font-weight:300;color:var(--text-primary);max-width:38ch">预测性维护最关心的是设备还能稳定运行多久，而不是只看某一个瞬间的状态。</p>
          <div class="mini-ledger">
            <div><b>输入</b><span>轴承运行过程中的水平、垂直振动快照。</span></div>
            <div><b>输出</b><span>连续 RUL 数值、健康趋势曲线和误差指标。</span></div>
            <div><b>边界</b><span>本次答辩主线是 RUL 回归预测；生存分析和失效概率保留为扩展接口。</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S08" data-animate="duo-mirror">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">数据来源与时间语义</div>
      <div class="r">04 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="head-stack" data-anim="line">
      <div class="t-meta">两个真实退化数据集；一个偏全寿命连续采样，一个带竞赛 Learning / Test 语义</div>
      <h2 class="page-title">数据不是文件列表，而是带时间顺序的退化过程。</h2>
    </div>
    <div class="duo-compare" data-anim="up" style="margin-top:5vh">
      <div class="col accent">
        <div class="col-tag"><span class="num">A</span> XJTU-SY</div>
        <div class="col-ttl">三工况十五轴承</div>
        <p class="col-desc">每个轴承沿寿命周期持续采样，水平、垂直两个振动通道用于构建退化特征序列。</p>
        <div class="dataset-facts">
          <div class="fact-card"><div class="num">25.6 kHz</div><div class="txt">采样频率</div></div>
          <div class="fact-card"><div class="num">32768</div><div class="txt">每个快照点数</div></div>
          <div class="fact-card"><div class="num">1.28 s</div><div class="txt">单个快照覆盖时长</div></div>
          <div class="fact-card"><div class="num">1 min</div><div class="txt">相邻快照间隔</div></div>
        </div>
      </div>
      <div class="vrule"></div>
      <div class="col">
        <div class="col-tag"><span class="num">B</span> PHM2012 / FEMTO</div>
        <div class="col-ttl">三工况竞赛基准</div>
        <p class="col-desc">快照更短、间隔更密，并包含温度文件；本次主线使用振动特征，温度信息保留在接口里。</p>
        <div class="dataset-facts">
          <div class="fact-card"><div class="num">25.6 kHz</div><div class="txt">采样频率</div></div>
          <div class="fact-card"><div class="num">2560</div><div class="txt">每个加速度文件点数</div></div>
          <div class="fact-card"><div class="num">0.1 s</div><div class="txt">单个快照覆盖时长</div></div>
          <div class="fact-card"><div class="num">约 10 s</div><div class="txt">相邻快照间隔；Test_set 终止 RUL 按官方条目补充</div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S22" data-animate="image-hero">
  <div class="canvas-card" style="padding:0;display:flex;flex-direction:column;overflow:hidden">
    <div data-anim="img" style="position:relative;flex:0 0 58%;overflow:hidden;background:var(--paper)">
      <img src="images/05-xjtu-bearing1-1-rms-health.png" data-image-slot="s22-hero-21x9" alt="XJTU-SY Bearing1_1 RMS and health trend" loading="eager" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;object-position:center center">
      <div class="chrome-min" style="position:absolute;top:0;left:0;right:0;padding:5.6vh 5vw 0">
        <div class="l">XJTU-SY 数据特点与曲线分析</div>
        <div class="r">05 / {TOTAL_SLIDES:02d}</div>
      </div>
    </div>
    <div data-anim="kpi" class="image-hero-body course">
      <p class="evidence-caption">以 Bearing1_1 为例，前期 RMS 和峰值较平稳，后期明显抬升。Health indicator 由 RMS、峭度、峰值因子、谱能量标准化后合成，用于辅助观察整体退化趋势。</p>
      <div class="image-hero-stats course">
        <div class="course-stat"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">Bearing</div><div class="kpi-hero" style="font-size:min(4.6vw,7.5vh);font-weight:200;line-height:.96;letter-spacing:-.04em">1_1</div><div style="height:1px;background:var(--border-subtle);margin-top:auto"></div><p class="body-sm">35Hz12kN 工况</p></div>
        <div class="course-stat"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">Snapshots</div><div class="kpi-hero" style="font-size:min(4.6vw,7.5vh);font-weight:200;line-height:.96;letter-spacing:-.04em">80</div><div style="height:1px;background:var(--border-subtle);margin-top:auto"></div><p class="body-sm">从 123 个快照均匀抽样</p></div>
        <div class="course-stat"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">RMS</div><div class="kpi-hero" style="font-size:min(4.6vw,7.5vh);font-weight:200;line-height:.96;letter-spacing:-.04em;color:var(--accent)">0.617→4.080</div><div style="height:1px;background:var(--border-subtle);margin-top:auto"></div><p class="body-sm">水平通道，80 个抽样快照首末值</p></div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S22" data-animate="image-hero">
  <div class="canvas-card" style="padding:0;display:flex;flex-direction:column;overflow:hidden">
    <div data-anim="img" style="position:relative;flex:0 0 58%;overflow:hidden;background:var(--paper)">
      <img src="images/06-phm2012-bearing1-1-rms-health.png" data-image-slot="s22-hero-21x9" alt="PHM2012 Bearing1_1 RMS and health trend" loading="eager" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;object-position:center center">
      <div class="chrome-min" style="position:absolute;top:0;left:0;right:0;padding:5.6vh 5vw 0">
        <div class="l">PHM2012 数据特点与差异</div>
        <div class="r">06 / {TOTAL_SLIDES:02d}</div>
      </div>
    </div>
    <div data-anim="kpi" class="image-hero-body course">
      <p class="evidence-caption">PHM2012 的单个快照只有 0.1 秒，但采样间隔约 10 秒，文件数量更密。Health indicator 同样由强度、冲击和频谱相关特征合成，温度文件由 loader 对齐并保留。</p>
      <div class="image-hero-stats course">
        <div class="course-stat"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">Bearing</div><div class="kpi-hero" style="font-size:min(4.6vw,7.5vh);font-weight:200;line-height:.96;letter-spacing:-.04em">1_1</div><div style="height:1px;background:var(--border-subtle);margin-top:auto"></div><p class="body-sm">Condition 1，Learning_set</p></div>
        <div class="course-stat"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">Snapshots</div><div class="kpi-hero" style="font-size:min(4.6vw,7.5vh);font-weight:200;line-height:.96;letter-spacing:-.04em">80</div><div style="height:1px;background:var(--border-subtle);margin-top:auto"></div><p class="body-sm">从 2803 个快照均匀抽样</p></div>
        <div class="course-stat"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">RMS</div><div class="kpi-hero" style="font-size:min(4.6vw,7.5vh);font-weight:200;line-height:.96;letter-spacing:-.04em;color:var(--accent)">0.428→2.068</div><div style="height:1px;background:var(--border-subtle);margin-top:auto"></div><p class="body-sm">水平通道，80 个抽样快照首末值</p></div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S17" data-animate="system-diagram">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">多轴承特征摘要</div>
      <div class="r">07 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="head-stack" data-anim="line">
      <div class="t-meta">同一指标在不同工况和不同轴承上的变化幅度并不相同</div>
      <h2 class="page-title small">特征分析不能只依赖一条代表曲线。</h2>
    </div>
    <div class="figure-board" data-anim="up">
      <div class="figure-frame">
        <img src="images/07-multi-bearing-feature-summary.png" data-image-slot="s22-hero-21x9" alt="Multi-bearing RMS summary" loading="eager">
      </div>
      <div class="figure-notes">
        <div class="figure-note accent"><h3>XJTU-SY 趋势更明显</h3><p>四个代表轴承后期 RMS 均高于早期，放大倍数约 4.47 到 6.61。</p></div>
        <div class="figure-note"><h3>PHM2012 差异更大</h3><p>同一数据集内部也存在不同寿命长度和不同退化速度，短序列不一定呈单调变化。</p></div>
        <div class="figure-note"><h3>对实验设计的影响</h3><p>训练划分按轴承和工况组织，避免相邻片段泄漏；指标解释必须保留数据范围说明。</p></div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S16" data-animate="field-notes">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">特征工程与 RUL 标签构造</div>
      <div class="r">08 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="head-stack" data-anim="line">
      <div class="t-meta">14 个时域特征 + 5 个频域特征</div>
      <h2 class="page-title">19 维特征让长振动片段变得可解释、可训练。</h2>
    </div>
    <div class="feature-board" data-anim="up">
      <div class="feature-list">
        <article class="feature-card accent"><div class="ttl">强度特征</div><p class="desc">RMS、峰值、谱能量，描述振动强度是否随寿命推进而增强。</p></article>
        <article class="feature-card"><div class="ttl">冲击特征</div><p class="desc">峭度、峰值因子、脉冲因子，对后期尖峰和波动变化更敏感。</p></article>
        <article class="feature-card"><div class="ttl">稳定性特征</div><p class="desc">均值、方差、标准差，用于观察总体偏移和波动强度。</p></article>
        <article class="feature-card"><div class="ttl">频谱特征</div><p class="desc">主频、谱质心、谱熵，补充频率分布和能量重心变化。</p></article>
      </div>
      <div class="feature-compact">
        <div class="t-meta" style="color:var(--accent);margin-bottom:2vh">Labeling</div>
        <div class="mono">snapshot → 19-dim feature<br/>feature[t:t+sequence_length] → sequence<br/>rul = max(elapsed_seconds) - elapsed_seconds<br/>PHM2012 Test_set terminal RUL → official entries only</div>
        <p class="body-sm" style="margin-top:auto;color:var(--text-secondary)">讲解时不逐个背特征名，重点说明它们分别对应强度、冲击和频谱变化。</p>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S17" data-animate="system-diagram">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">系统总体架构</div>
      <div class="r">09 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="head-stack" data-anim="line">
      <div class="t-meta">从数据读取到结果输出，按真实数据流组织模块</div>
      <h2 class="page-title small">架构图对应源码模块，不只是概念分层。</h2>
    </div>
    <div class="figure-board" data-anim="up">
      <div class="figure-frame">
        <img src="images/09-end-to-end-rul-architecture.png" data-image-slot="s22-hero-21x9" alt="End-to-end RUL architecture" loading="eager">
      </div>
      <div class="figure-notes">
        <div class="architecture-note accent"><h3>边界原则</h3><p>数据读取、特征构造、模型训练和评价输出分开实现，避免互相依赖。</p></div>
        <div class="architecture-note"><h3>复用方式</h3><p>两个数据集进入同一套特征序列接口，后续替换模型时不重写 loader。</p></div>
        <div class="architecture-note"><h3>展示入口</h3><p>核心逻辑在 src 包内；notebook 主要用于示例运行和论文复现实验。</p></div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S08" data-animate="duo-mirror">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">难点一：异构数据集统一</div>
      <div class="r">10 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="head-stack" data-anim="line">
      <div class="t-meta">同一个模型训练流程，不能依赖某个数据集的目录习惯</div>
      <h2 class="page-title">解决办法是把差异收在 loader 层。</h2>
    </div>
    <div class="difficulty-grid" data-anim="up">
      <div class="difficulty-col">
        <div class="difficulty-card accent"><h3>数据差异</h3><p>XJTU-SY 与 PHM2012 的快照长度、时间间隔、split 语义和终止 RUL 信息都不同。</p></div>
        <div class="difficulty-card"><h3>时间顺序</h3><p>RUL 是随寿命推进变化的目标，训练划分必须保留退化顺序，不能把时间窗口随意打乱。</p></div>
      </div>
      <div class="vrule"></div>
      <div class="difficulty-col">
        <div class="difficulty-card"><h3>统一实体</h3><p>所有样本进入后续流程前，都拥有 sample frame、elapsed_seconds、metadata 和 rul_unit。</p></div>
        <div class="difficulty-card"><h3>工程权衡</h3><p>max_samples 是读前均匀抽样，用于控制课程训练时间，同时保留早期、中期和后期片段。</p></div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S11" data-animate="timeline-walk">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">难点二：信号到序列特征</div>
      <div class="r">11 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="head-stack" data-anim="line">
      <div class="t-meta">原始振动快照太长，直接输入不适合课程范围内的稳定复现</div>
      <h2 class="page-title">模型看到的是一段连续变化，而不是孤立的一个点。</h2>
    </div>
    <div class="sequence-table" data-anim="up">
      <div class="sequence-cell"><div class="top">01</div><div class="main">Raw snapshot</div><div class="sub">32768 或 2560 个振动点</div></div>
      <div class="sequence-cell"><div class="top">02</div><div class="main">Feature vector</div><div class="sub">压缩为 19 维时频域描述</div></div>
      <div class="sequence-cell accent"><div class="top">03</div><div class="main">Feature sequence</div><div class="sub">5 或 10 个时间步组成输入序列</div></div>
      <div class="sequence-cell"><div class="top">04</div><div class="main">RUL label</div><div class="sub">按 elapsed_seconds 生成剩余寿命</div></div>
      <div class="sequence-cell"><div class="top">05</div><div class="main">Prediction</div><div class="sub">输出连续寿命值与误差指标</div></div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S21" data-animate="tech-spec">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">为什么选这两篇论文</div>
      <div class="r">12 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="head-stack" data-anim="line">
      <div class="t-meta">选择标准：数据集匹配、结构可实现、指标可复查</div>
      <h2 class="page-title small">两篇论文分别覆盖 attention 路线和跨工况 Transformer 路线。</h2>
    </div>
    <div class="paper-choice-grid" data-anim="up">
      <article class="paper-card accent">
        <div class="year">Huang et al. · 2024</div>
        <h3>CNN-LSTM-AM</h3>
        <p>论文使用时域和频域特征，模型结构清晰，给出 Score 公式，适合验证本项目的 19 维特征、attention 模型和指标实现。</p>
        <p>本项目对齐：CNN-LSTM-AM 主模型、CNN-LSTM baseline、XJTU-SY 与 PHM2012 两个数据集。</p>
      </article>
      <article class="paper-card">
        <div class="year">Jiang et al. · 2026</div>
        <h3>xLSTM-Transformer</h3>
        <p>论文同时覆盖 XJTU-SY 和 PHM2012，公开工况划分、序列长度、训练参数和 RMSE/R2/Score，适合做第二篇同规格复现。</p>
        <p>本项目对齐：XLSTM-Transformer、Feature-Transformer baseline、LSTM-Transformer baseline 和六个工况输出。</p>
      </article>
    </div>
  </div>
</section>

<section class="slide" data-layout="S11" data-animate="timeline-walk">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">复现实验怎么做</div>
      <div class="r">13 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="head-stack" data-anim="line">
      <div class="t-meta">对齐论文输入组织、模型结构、baseline、划分和指标口径</div>
      <h2 class="page-title small">复现定义为工程 workflow 复现，不声明作者源码级数值重现。</h2>
    </div>
    <div class="rul-flow" data-anim="up">
      <div class="rul-step"><div class="num">01</div><div class="ttl">读取真实数据</div><div class="txt">优先从 data/external 读取 XJTU-SY 与 PHM2012。</div></div>
      <div class="rul-step"><div class="num">02</div><div class="ttl">构造特征序列</div><div class="txt">每个快照提取 19 维特征，再按时间顺序组成序列。</div></div>
      <div class="rul-step accent"><div class="num">03</div><div class="ttl">训练主模型</div><div class="txt">CNN-LSTM-AM 与 XLSTM-Transformer 均真实训练 8 epoch。</div></div>
      <div class="rul-step"><div class="num">04</div><div class="ttl">训练 baseline</div><div class="txt">保留 CNN-LSTM、Feature-Transformer、LSTM-Transformer 对比。</div></div>
      <div class="rul-step"><div class="num">05</div><div class="ttl">输出指标</div><div class="txt">RMSE、NormalizedRMSE、R2、Huang Score、PHM Score。</div></div>
      <div class="rul-step"><div class="num">06</div><div class="ttl">保存证据</div><div class="txt">history、metrics、predictions、comparison_metrics 全部落盘。</div></div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S20" data-animate="stacked-ledger">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">评价指标与实验结果解释</div>
      <div class="r">14 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="head-stack" data-anim="line">
      <div class="t-meta">8 epoch；CNN-LSTM-AM 每数据集抽样 48 快照，xLSTM-Transformer 每轴承抽样 16 快照</div>
      <h2 class="page-title small">指标已复现输出；单次短训练不当作最终性能结论。</h2>
    </div>
    <div class="result-table" data-anim="up">
      <div class="head">实验</div><div class="head">数据集</div><div class="head">模型</div><div class="head">RMSE</div><div class="head">Huang Score</div>
      <div>CNN-LSTM-AM</div><div>XJTU-SY</div><div class="accent">Attention</div><div>406.30</div><div>29.00</div>
      <div>CNN-LSTM-AM</div><div>PHM2012</div><div class="accent">Attention</div><div>651.04</div><div>29.45</div>
      <div>xLSTM-Transformer</div><div>XJTU-SY condition 1</div><div>XLSTM</div><div>2280.63</div><div>27.31</div>
      <div>xLSTM-Transformer</div><div>PHM2012 condition 3</div><div>XLSTM</div><div>1594.74</div><div>31.89</div>
    </div>
    <div data-anim="foot" style="display:grid;grid-template-columns:repeat(3,1fr);gap:1.5vw;margin-top:3.2vh">
      <div class="difficulty-card"><h3>误差大小</h3><p>RMSE、MAE、SMAPE、R2 用于普通回归解释；PHM2012 上负 R2 已作为限制保留。</p></div>
      <div class="difficulty-card accent"><h3>RUL Score</h3><p>Huang 原版 Score 与 PHM/RUL 惩罚 Score 分开输出，不混同解释。</p></div>
      <div class="difficulty-card"><h3>方向倾向</h3><p>over / under prediction rate 用于判断偏早或偏晚预测，辅助解释维护风险。</p></div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S20" data-animate="stacked-ledger">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">测试方案与验收证据</div>
      <div class="r">15 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="ledger" class="ledger-list">
      <div class="ledger-row course">
        <div class="ledger-num">31</div>
        <div class="ledger-label"><div class="t-meta">自动化测试</div><div class="lead" style="font-weight:300">全量 pytest 通过，覆盖 loader、特征、标签、指标、训练 workflow 和 notebook smoke。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">pytest</div>
      </div>
      <div class="ledger-row course">
        <div class="ledger-num" style="color:var(--accent)">8</div>
        <div class="ledger-label"><div class="t-meta">notebook 示例</div><div class="lead" style="font-weight:300">examples 下 notebook 统一执行，不只是检查文件存在。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">ipynb</div>
      </div>
      <div class="ledger-row course">
        <div class="ledger-num">2</div>
        <div class="ledger-label"><div class="t-meta">论文复现 workflow</div><div class="lead" style="font-weight:300">两篇复现均读取 data/external 真实数据，并输出 history、metrics、predictions 与 comparison 表。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">CSV</div>
      </div>
      <div class="ledger-row course">
        <div class="ledger-num">3</div>
        <div class="ledger-label"><div class="t-meta">测试层次</div><div class="lead" style="font-weight:300">单元测试、集成测试、确认测试报告均已补齐，结题材料可归档。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">docs</div>
      </div>
    </div>
  </div>
</section>

<section class="slide split" data-layout="S10" data-animate="split-statement">
  <div class="canvas-card">
    <div class="split-half">
      <div class="half b-accent" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between;position:relative;overflow:hidden">
        <canvas class="ascii-bg" aria-hidden="true"></canvas>
        <div class="chrome-min" style="margin-bottom:0;position:relative;z-index:1">
          <div class="l">成员分工与总结</div>
          <div class="r">16 / {TOTAL_SLIDES:02d}</div>
        </div>
        <div data-anim="manifesto" style="display:flex;flex-direction:column;gap:2vh;position:relative;z-index:1">
          <div class="t-meta" style="color:rgba(255,255,255,.78);letter-spacing:.18em;margin-bottom:1.6vh">结题结论</div>
          <h2 style="font-family:var(--sans),var(--sans-zh);font-size:min(6.5vw,11.3vh);line-height:1;letter-spacing:-.025em;font-weight:200;color:#fff">从数据到结果，流程实际跑通了。</h2>
          <div style="font-family:var(--sans),var(--sans-zh);font-size:max(16px,.98vw);line-height:1.6;color:rgba(255,255,255,.84);font-weight:400;max-width:39ch;margin-top:1.4vh">真实数据读取、特征提取、RUL 建模、评价输出和课程文档均可追溯。</div>
        </div>
        <div data-anim="signature" style="display:flex;justify-content:space-between;align-items:end;border-top:1px solid rgba(255,255,255,.22);padding-top:2vh;position:relative;z-index:1">
          <div class="t-meta" style="color:rgba(255,255,255,.62)">谢谢老师和同学</div>
          <div class="t-meta" style="color:rgba(255,255,255,.62)">欢迎提问</div>
        </div>
      </div>
      <div class="half" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between">
        <div class="chrome-min">
          <div class="l">分工 / 不足</div>
          <div class="r">后续工作</div>
        </div>
        <div data-anim="rules" class="closing-list">
          <div class="closing-item"><div class="n">35%</div><div><h3>zyj：架构与 RUL 模型</h3><p>项目负责人、训练 workflow、论文复现和主要实验集成。</p></div></div>
          <div class="closing-item"><div class="n">25%</div><div><h3>cyy：数据处理与特征工程</h3><p>两个数据集 loader、时频域特征、样本组织和数据文档。</p></div></div>
          <div class="closing-item"><div class="n">20%</div><div><h3>zdh：分析与评价支持</h3><p>概率分析基础能力、评价支持、测试报告和确认测试材料。</p></div></div>
          <div class="closing-item"><div class="n">20%</div><div><h3>zy：可视化与文档</h3><p>可视化页面、用户手册、安装手册和答辩材料。</p></div></div>
          <div class="closing-item accent"><div class="n">Next</div><div><h3>不足与后续</h3><p>扩大训练规模，补充多随机种子和更严格跨工况验证；温度信息后续可进一步融合。</p></div></div>
        </div>
        <div data-anim="foot" class="t-meta" style="color:var(--text-helper);text-align:right">主线：剩余寿命预测与预测性维护</div>
      </div>
    </div>
  </div>
</section>
"""


def build_deck() -> str:
    """
    Build the final web PPT by replacing the template slide sample block.

    Returns
    -------
    str
        Complete HTML slide deck.
    """
    template = SKILL_TEMPLATE.read_text(encoding="utf-8")
    start = template.index("<!-- ============================================================\n     SLIDES 插入区")
    end = template.index("</div>\n\n<div id=\"nav\"></div>", start)
    replacement = f"<!-- Generated final defense slides. -->\n{SLIDES.strip()}\n"
    html = template[:start] + replacement + template[end:]
    template_title = "<title>" + "[\u5fc5\u586b] 替换为 PPT 标题 · Deck Title</title>"
    html = html.replace(template_title, f"<title>{PROJECT_TITLE} · 结题答辩</title>")
    html = html.replace("</style>", CUSTOM_CSS.rstrip() + "\n</style>", 1)
    html = html.replace("guizang-ppt-low-power", "bearing-rul-ppt-low-power")
    html = html.replace("<!-- Motion One 动效引擎 (与原模板一致) -->", "<!-- Motion One animation runtime -->")
    html = html.replace("fat skills", "fat layer")
    html = html.replace("P15 skill 矩阵", "P15 matrix")
    return html


def main() -> None:
    """
    Generate the final deck file and companion evidence images.
    """
    if not SKILL_TEMPLATE.exists():
        raise FileNotFoundError(
            "Guizang PPT skill template was not found. Expected "
            f"{SKILL_TEMPLATE}. Clone op7418/guizang-ppt-skill.git into .agents/skills first."
        )

    generate_evidence_assets()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH.parent / "assets").mkdir(parents=True, exist_ok=True)
    if SKILL_MOTION.exists():
        shutil.copyfile(SKILL_MOTION, MOTION_OUTPUT)
    OUTPUT_PATH.write_text(build_deck(), encoding="utf-8")
    print(f"generated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
