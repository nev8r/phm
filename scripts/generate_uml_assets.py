#!/usr/bin/env python3
"""
Generate PNG UML-style diagrams for the course UML document.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docx" / "mid-term" / "assets" / "uml"


def setup_canvas(width: float = 12, height: float = 7):
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    return fig, ax


def box(ax, xy, wh, text, face="#F7F7F7", edge="#222222", fontsize=10):
    x, y = xy
    w, h = wh
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, start, end, color="#333333"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def save(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def use_case_diagram():
    fig, ax = setup_canvas()
    box(ax, (0.3, 5.2), (1.6, 0.7), "实验使用者", "#E8F1FF")
    box(ax, (0.3, 3.2), (1.6, 0.7), "课程评审人员", "#E8F1FF")
    box(ax, (0.3, 1.2), (1.6, 0.7), "项目维护者", "#E8F1FF")
    cases = [
        (3.0, 5.7, "导入轴承数据"),
        (6.0, 5.7, "查看特征趋势"),
        (9.0, 5.7, "构造 RUL 标签"),
        (3.0, 3.7, "训练 RUL 模型"),
        (6.0, 3.7, "执行论文复现"),
        (9.0, 3.7, "查看指标与曲线"),
        (4.5, 1.5, "运行自动化测试"),
        (7.5, 1.5, "导出课程文档"),
    ]
    for x, y, text in cases:
        box(ax, (x, y), (2.0, 0.75), text, "#FFF8E6")
    for target in [(3.0, 6.05), (6.0, 6.05), (9.0, 6.05), (3.0, 4.05), (6.0, 4.05), (9.0, 4.05)]:
        arrow(ax, (1.9, 5.55), target)
    arrow(ax, (1.9, 3.55), (9.0, 4.05))
    arrow(ax, (1.9, 3.55), (7.5, 1.85))
    arrow(ax, (1.9, 1.55), (4.5, 1.85))
    arrow(ax, (1.9, 1.55), (7.5, 1.85))
    ax.set_title("UML 用例图：课程项目主要参与者与用例", fontsize=14, pad=12)
    save(fig, "use-case.png")


def class_diagram():
    fig, ax = setup_canvas(13.8, 8.0)
    nodes = {
        "BaseBearingLoader\n+ list_entities()\n+ load_entity()": (0.35, 6.25),
        "XJTULoader\n+ parse condition\n+ load vibration CSV": (0.35, 4.95),
        "PHM2012Loader\n+ align accel/temp\n+ parse split": (0.35, 3.65),
        "BearingEntity\nentity_id, samples\nsample_rate, metadata": (3.15, 5.15),
        "SignalFeatureExtractor\n+ extract()\n+ build_health_indicator()": (6.0, 5.15),
        "FeatureSequenceRulLabeler\nsequence_length\n+ label()": (8.95, 5.15),
        "BearingWindowDataset\nwindows, targets\nfeature_frame": (11.35, 5.15),
        "BaseBearingModel\n+ forward()\n+ count_parameters()": (6.0, 2.9),
        "CNNLSTMAttention\nCNN + LSTM + AM": (3.15, 1.3),
        "XLSTMTransformer\nxLSTM + Transformer": (8.95, 1.3),
        "BaseTrainer\n+ train()\nExperimentTracker": (0.35, 1.3),
        "BaseTester\n+ test()\nPredictionResult": (3.15, 0.1),
        "Evaluator\n+ add()\n+ evaluate()": (6.0, 0.1),
    }
    for name, (x, y) in nodes.items():
        width = 2.45 if x < 11 else 2.2
        box(ax, (x, y), (width, 0.92), name, "#F5F5F5", fontsize=8)

    relations = [
        ("BaseBearingLoader\n+ list_entities()\n+ load_entity()", "XJTULoader\n+ parse condition\n+ load vibration CSV", "extends"),
        ("BaseBearingLoader\n+ list_entities()\n+ load_entity()", "PHM2012Loader\n+ align accel/temp\n+ parse split", "extends"),
        ("XJTULoader\n+ parse condition\n+ load vibration CSV", "BearingEntity\nentity_id, samples\nsample_rate, metadata", "creates"),
        ("PHM2012Loader\n+ align accel/temp\n+ parse split", "BearingEntity\nentity_id, samples\nsample_rate, metadata", "creates"),
        ("BearingEntity\nentity_id, samples\nsample_rate, metadata", "SignalFeatureExtractor\n+ extract()\n+ build_health_indicator()", "uses samples"),
        ("SignalFeatureExtractor\n+ extract()\n+ build_health_indicator()", "FeatureSequenceRulLabeler\nsequence_length\n+ label()", "features"),
        ("FeatureSequenceRulLabeler\nsequence_length\n+ label()", "BearingWindowDataset\nwindows, targets\nfeature_frame", "builds"),
        ("BearingWindowDataset\nwindows, targets\nfeature_frame", "BaseTrainer\n+ train()\nExperimentTracker", "train/test data"),
        ("BaseTrainer\n+ train()\nExperimentTracker", "BaseBearingModel\n+ forward()\n+ count_parameters()", "optimizes"),
        ("BaseBearingModel\n+ forward()\n+ count_parameters()", "CNNLSTMAttention\nCNN + LSTM + AM", "implemented by"),
        ("BaseBearingModel\n+ forward()\n+ count_parameters()", "XLSTMTransformer\nxLSTM + Transformer", "implemented by"),
        ("BaseTrainer\n+ train()\nExperimentTracker", "BaseTester\n+ test()\nPredictionResult", "trained model"),
        ("BaseTester\n+ test()\nPredictionResult", "Evaluator\n+ add()\n+ evaluate()", "targets/predictions"),
    ]
    for source, target, label in relations:
        x1, y1 = nodes[source]
        x2, y2 = nodes[target]
        start = (x1 + 1.15, y1 + 0.46)
        end = (x2 + 1.15, y2 + 0.46)
        arrow(ax, start, end)
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label, ha="center", fontsize=7, color="#555555")
    ax.set_title("UML 类图：核心类、方法与依赖关系", fontsize=14, pad=12)
    save(fig, "class-diagram.png")


def sequence_diagram():
    fig, ax = setup_canvas(13, 7)
    parts = ["Notebook/API", "Loader", "Labeler", "Trainer", "Tester", "Evaluator", "Output"]
    xs = [0.8, 2.7, 4.6, 6.5, 8.4, 10.1, 11.8]
    for x, p in zip(xs, parts):
        box(ax, (x - 0.65, 6.1), (1.3, 0.55), p, "#EAF7EA", fontsize=8)
        ax.plot([x, x], [0.5, 6.1], color="#BBBBBB", linestyle="--", linewidth=0.9)
    steps = [
        (0, 1, 5.4, "load(data_path)"),
        (1, 2, 4.8, "BearingEntity"),
        (2, 3, 4.2, "build RUL dataset"),
        (3, 6, 3.6, "history.csv"),
        (3, 4, 3.0, "predict(test_dataset)"),
        (4, 5, 2.4, "evaluate(target, prediction)"),
        (5, 6, 1.8, "metrics / comparison"),
    ]
    for src, dst, y, label in steps:
        arrow(ax, (xs[src], y), (xs[dst], y))
        ax.text((xs[src] + xs[dst]) / 2, y + 0.12, label, ha="center", fontsize=8)
    ax.set_title("UML 顺序图：RUL 训练与评估调用顺序", fontsize=14, pad=12)
    save(fig, "sequence-diagram.png")


def component_diagram():
    fig, ax = setup_canvas()
    box(ax, (0.7, 4.8), (2.0, 0.8), "data/external\n真实数据", "#E8F1FF")
    box(ax, (0.7, 2.0), (2.0, 0.8), "examples\nnotebook", "#E8F1FF")
    components = [
        (4.0, 5.4, "dataset / data"),
        (4.0, 4.0, "feature / labeling"),
        (4.0, 2.6, "models / training"),
        (4.0, 1.2, "evaluation / visualization"),
    ]
    for x, y, text in components:
        box(ax, (x, y), (2.5, 0.8), text, "#FFF8E6")
    box(ax, (8.6, 4.5), (2.2, 0.8), "tests", "#F5F5F5")
    box(ax, (8.6, 2.8), (2.2, 0.8), "outputs / tmp", "#F5F5F5")
    box(ax, (8.6, 1.1), (2.2, 0.8), "docx / docs", "#F5F5F5")
    for start, end in [((2.7, 5.2), (4.0, 5.8)), ((5.25, 5.4), (5.25, 4.8)), ((5.25, 4.0), (5.25, 3.4)), ((5.25, 2.6), (5.25, 2.0)), ((6.5, 1.6), (8.6, 3.2)), ((2.7, 2.4), (4.0, 3.0)), ((6.5, 5.8), (8.6, 4.9)), ((6.5, 1.6), (8.6, 1.5))]:
        arrow(ax, start, end)
    ax.set_title("UML 组件图：工程模块与交付物边界", fontsize=14, pad=12)
    save(fig, "component-diagram.png")


def deployment_diagram():
    fig, ax = setup_canvas()
    nodes = [
        (0.8, 4.8, "开发者本地机器"),
        (3.4, 4.8, "uv / Python 3.11+"),
        (6.0, 4.8, "项目仓库"),
        (8.8, 5.7, "data/external"),
        (8.8, 4.2, "data/generated"),
        (8.8, 2.7, "tmp / outputs"),
        (6.0, 2.0, "docx PDF/DOCX"),
    ]
    for x, y, text in nodes:
        box(ax, (x, y), (2.0, 0.75), text, "#F5F5F5")
    for a, b in [((2.8, 5.18), (3.4, 5.18)), ((5.4, 5.18), (6.0, 5.18)), ((8.0, 5.18), (8.8, 6.05)), ((8.0, 5.18), (8.8, 4.55)), ((8.0, 5.18), (8.8, 3.05)), ((7.0, 4.8), (7.0, 2.75))]:
        arrow(ax, a, b)
    ax.set_title("UML 部署图：本地课程实验运行环境", fontsize=14, pad=12)
    save(fig, "deployment-diagram.png")


def main() -> None:
    use_case_diagram()
    class_diagram()
    sequence_diagram()
    component_diagram()
    deployment_diagram()
    print(f"generated UML assets in {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
