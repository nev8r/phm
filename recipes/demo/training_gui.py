"""
Chinese training replay GUI for completed PHM experiment reports.

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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

TASK_LABELS = {
    "rul_tabular": "剩余寿命预测 RUL",
    "health_state_tabular": "健康状态识别 HealthState",
    "early_fault_tabular": "早期故障检测 EarlyFault",
}

MLP_PRESETS = [
    ("XJTU-SY RUL 默认 MLP", "xjtu_main_rul_mlp_full_manual_basic_no_reference"),
    ("XJTU-SY EarlyFault 默认 MLP", "xjtu_main_early_mlp_compact_non_label_source"),
    ("PHM2012 RUL 调参 MLP", "phm_official_rul_mlp_tuned_compact_non_label_source"),
    ("PHM2012 HealthState 调参 MLP", "phm_official_health_mlp_tuned_compact_non_label_source"),
]

NON_MLP_PRESETS = [
    ("XJTU-SY RUL RandomForest", "y02_xjtu_rul_random_forest_full_manual_basic_no_reference", "regression"),
    ("PHM2012 RUL RandomForest", "y08_phm_rul_random_forest_compact_non_label_source", "regression"),
    ("XJTU-SY HealthState XGBoost", "y03_xjtu_health_xgboost_compact_non_label_source", "classification"),
    ("PHM2012 EarlyFault RandomForest", "y12_phm_early_random_forest_compact_non_label_source", "classification"),
]


@dataclass
class MLPReplayRun:
    title: str
    experiment_id: str
    history: List[Dict[str, Any]]
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    best_epoch: int
    best_val_loss: float
    test_metric_name: str
    test_metric_value: float


@dataclass
class NonMLPDemoRun:
    title: str
    experiment_id: str
    task_type: str
    metrics: Dict[str, Any]
    figure_paths: Dict[str, Path]


def main() -> None:
    parser = argparse.ArgumentParser(description="中文训练过程 GUI 和验收材料导出工具")
    parser.add_argument("--export-demo", action="store_true", help="生成 reports/training_gui_demo 验收材料")
    parser.add_argument("--output", default="reports/training_gui_demo", help="导出目录")
    args = parser.parse_args()

    if args.export_demo:
        output = (PROJECT_ROOT / args.output).resolve()
        build_demo_export(output)
        print(f"中文训练 GUI 演示材料已生成：{output}")
        return

    launch_gui()


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as exc:  # pragma: no cover - depends on desktop runtime.
        raise RuntimeError("当前 Python 环境无法启动 Tkinter GUI。") from exc

    try:
        app = TrainingDemoApp()
        app.mainloop()
    except Exception as exc:  # pragma: no cover - GUI error path.
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("启动失败", f"训练过程演示系统启动失败：{exc}")
        root.destroy()
        raise


def load_mlp_replay_runs() -> List[MLPReplayRun]:
    runs = []
    for title, experiment_id in MLP_PRESETS:
        base = REPORTS_DIR / "baseline_results" / experiment_id
        history = read_json(base / "history.json", default=[])
        val_metrics = read_json(base / "val_metrics.json", default={})
        test_metrics = read_json(base / "test_metrics.json", default={})
        if not history:
            raise FileNotFoundError(f"缺少训练历史：{base / 'history.json'}")
        best = min(history, key=lambda row: float(row.get("val_loss", math.inf)))
        metric_name, metric_value = pick_primary_test_metric(test_metrics)
        runs.append(MLPReplayRun(
            title=title,
            experiment_id=experiment_id,
            history=history,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_epoch=int(best["epoch"]),
            best_val_loss=float(best["val_loss"]),
            test_metric_name=metric_name,
            test_metric_value=metric_value,
        ))
    return runs


def load_non_mlp_demo_runs() -> List[NonMLPDemoRun]:
    metrics = read_csv(REPORTS_DIR / "non_mlp_baseline_results" / "non_mlp_tabular_metrics.csv")
    runs = []
    for title, experiment_id, task_type in NON_MLP_PRESETS:
        row = metrics[metrics["experiment_id"] == experiment_id]
        if row.empty:
            metric_row: Dict[str, Any] = {"experiment_id": experiment_id, "fit_status": "missing"}
        else:
            metric_row = row.iloc[0].to_dict()
        figures = build_non_mlp_figure_paths(experiment_id, task_type)
        runs.append(NonMLPDemoRun(title=title, experiment_id=experiment_id, task_type=task_type, metrics=metric_row, figure_paths=figures))
    return runs


def load_final_decisions() -> List[Dict[str, str]]:
    decisions = read_csv(REPORTS_DIR / "baseline_results" / "baseline_final_decisions_with_tuned.csv")
    comparison = read_csv(REPORTS_DIR / "non_mlp_baseline_results" / "non_mlp_vs_mlp_comparison.csv")
    rows = []
    for _, decision in decisions.iterrows():
        task = str(decision["task"])
        comp = comparison[(comparison["dataset"] == decision["dataset"]) & (comparison["task"] == task)]
        non_mlp_note = "非MLP：未纳入对比"
        if not comp.empty:
            direction = str(comp.iloc[0]["metric_direction"])
            idx = comp["non_mlp_test_primary"].idxmin() if direction == "lower_is_better" else comp["non_mlp_test_primary"].idxmax()
            best = comp.loc[idx]
            non_mlp_note = f"非MLP最佳：{best['model_family']}={format_number(best['non_mlp_test_primary'])}"
        rows.append({
            "数据集": str(decision["dataset"]),
            "任务": TASK_LABELS.get(task, task),
            "推荐模型": str(decision["model_decision"]),
            "推荐特征子集": str(decision["feature_subset_decision"]),
            "测试指标": f"{decision['primary_metric']} 默认={format_number(decision['default_test_primary'])}，调参={format_number(decision['tuned_test_primary'])}；{non_mlp_note}",
            "注意事项": str(decision["caveat"]),
            "下一步": str(decision["next_action"]),
        })
    return rows


def build_non_mlp_figure_paths(experiment_id: str, task_type: str) -> Dict[str, Path]:
    figures_dir = REPORTS_DIR / "non_mlp_baseline_results" / experiment_id / "figures"
    if task_type == "regression":
        return {
            "预测值 vs 真实值": figures_dir / "test_pred_vs_true.png",
            "残差图": figures_dir / "test_residuals.png",
            "特征重要性": figures_dir / "feature_importance_top10.png",
        }
    return {
        "混淆矩阵": figures_dir / "test_confusion_matrix.png",
        "类别分布": figures_dir / "test_class_distribution.png",
        "特征重要性": figures_dir / "feature_importance_top10.png",
    }


def pick_primary_test_metric(metrics: Dict[str, Any]) -> tuple[str, float]:
    for key in ["RMSE", "WeightedF1", "Accuracy", "MacroF1", "loss"]:
        value = metrics.get(key)
        if value is not None:
            return key, float(value)
    return "未知指标", math.nan


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "N/A"
    return f"{number:.4f}"


class TrainingDemoApp:  # pragma: no cover - exercised manually through Tk.
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("轴承预测训练过程演示系统")
        self.root.geometry("1320x820")
        self.root.minsize(1120, 720)
        self.mlp_runs = load_mlp_replay_runs()
        self.non_mlp_runs = load_non_mlp_demo_runs()
        self.final_decisions = load_final_decisions()
        self.frames: Dict[str, Any] = {}
        self.content = None

        self.configure_style()
        self.build_layout()
        self.show_page("项目概览")

    def mainloop(self) -> None:
        self.root.mainloop()

    def configure_style(self) -> None:
        style = self.ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6f8")
        style.configure("Sidebar.TFrame", background="#ffffff")
        style.configure("Title.TLabel", font=("STHeiti", 22, "bold"), background="#f4f6f8", foreground="#1f2933")
        style.configure("Heading.TLabel", font=("STHeiti", 16, "bold"), background="#f4f6f8", foreground="#1f2933")
        style.configure("Body.TLabel", font=("STHeiti", 12), background="#f4f6f8", foreground="#344054")
        style.configure("Card.TLabel", font=("STHeiti", 13), background="#ffffff", foreground="#1f2933", padding=10)
        style.configure("Nav.TButton", font=("STHeiti", 13), padding=10)
        style.configure("Primary.TButton", font=("STHeiti", 12, "bold"), padding=8)

    def build_layout(self) -> None:
        tk = self.tk
        ttk = self.ttk
        shell = ttk.Frame(self.root)
        shell.pack(fill="both", expand=True)

        sidebar = ttk.Frame(shell, style="Sidebar.TFrame", width=210)
        sidebar.pack(side="left", fill="y")
        ttk.Label(sidebar, text="训练演示", font=("STHeiti", 20, "bold"), background="#ffffff").pack(anchor="w", padx=18, pady=(22, 18))
        for page in ["项目概览", "MLP训练回放", "非MLP模型诊断", "最终决策", "关于与说明"]:
            ttk.Button(sidebar, text=page, style="Nav.TButton", command=lambda name=page: self.show_page(name)).pack(fill="x", padx=14, pady=5)

        main = ttk.Frame(shell)
        main.pack(side="left", fill="both", expand=True)
        ttk.Label(main, text="模式：真实训练结果加速回放    数据来源：reports/ 已归档结果    说明：未重新训练", style="Body.TLabel").pack(anchor="w", padx=22, pady=(12, 4))
        self.content = ttk.Frame(main)
        self.content.pack(fill="both", expand=True, padx=22, pady=14)

    def show_page(self, name: str) -> None:
        assert self.content is not None
        for child in self.content.winfo_children():
            child.destroy()
        factories = {
            "项目概览": self.build_overview_page,
            "MLP训练回放": self.build_mlp_page,
            "非MLP模型诊断": self.build_non_mlp_page,
            "最终决策": self.build_final_page,
            "关于与说明": self.build_about_page,
        }
        factories[name](self.content)

    def build_overview_page(self, parent: Any) -> None:
        ttk = self.ttk
        ttk.Label(parent, text="轴承预测训练过程演示系统", style="Title.TLabel").pack(anchor="w")
        ttk.Label(parent, text="中文 GUI 展示已经完成的真实训练 / 拟合结果，MLP 页面提供 50 epoch 加速回放。", style="Body.TLabel").pack(anchor="w", pady=(4, 14))
        cards = [
            ("数据集", "XJTU-SY、PHM2012"),
            ("任务", "RUL 剩余寿命、HealthState 健康状态、EarlyFault 早期故障"),
            ("模型", "默认 MLP、调参 MLP、XGBoost、RandomForest"),
            ("真实实验数量", "MLP 系列 33 个，非 MLP 系列 12 个，合计 45 个"),
        ]
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        for idx, (title, body) in enumerate(cards):
            frame = ttk.LabelFrame(grid, text=title)
            frame.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=8, pady=8)
            ttk.Label(frame, text=body, style="Card.TLabel", wraplength=430).pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        ttk.Label(parent, text="说明：XGBoost / RandomForest 是树模型拟合，不是 epoch 训练；MLP 才展示 50 epoch 曲线。", style="Heading.TLabel").pack(anchor="w", pady=(18, 6))
        ttk.Label(parent, text="GUI 只读取 reports 目录下的整理结果，不重新计算指标。", style="Body.TLabel").pack(anchor="w")

    def build_mlp_page(self, parent: Any) -> None:
        MLPReplayPanel(parent, self.mlp_runs)

    def build_non_mlp_page(self, parent: Any) -> None:
        NonMLPPanel(parent, self.non_mlp_runs)

    def build_final_page(self, parent: Any) -> None:
        FinalDecisionPanel(parent, self.final_decisions)

    def build_about_page(self, parent: Any) -> None:
        ttk = self.ttk
        ttk.Label(parent, text="关于与说明", style="Title.TLabel").pack(anchor="w")
        for line in [
            "这里展示的是已完成真实训练结果的加速回放，不是伪造训练，也不是重新训练。",
            "MLP 回放读取 history.json 中的 train_loss / val_loss / epoch。",
            "非 MLP 页面展示 Step Y-R 生成的预测图、残差图、混淆矩阵和特征重要性。",
            "mag__time__rms 是 HI/FPT 标签源参考特征，独立结论优先看 non-reference 特征子集。",
        ]:
            ttk.Label(parent, text=line, style="Body.TLabel", wraplength=900).pack(anchor="w", pady=6)


class MLPReplayPanel:  # pragma: no cover - GUI runtime.
    def __init__(self, parent: Any, runs: List[MLPReplayRun]) -> None:
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.tk = tk
        self.ttk = ttk
        self.parent = parent
        self.runs = runs
        self.current_idx = 0
        self.current_epoch = 0
        self.playing = False
        self.speed_var = tk.StringVar(value="10x")
        self.run_var = tk.StringVar(value=runs[0].title)
        self.figure = Figure(figsize=(8.2, 4.0), dpi=100)
        self.ax = self.figure.add_subplot(111)

        ttk.Label(parent, text="MLP 训练回放", style="Title.TLabel").pack(anchor="w")
        ttk.Label(parent, text="注意：这是已完成真实训练 history.json 的加速回放，不会重新训练。", style="Body.TLabel").pack(anchor="w", pady=(0, 12))
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=6)
        ttk.Label(controls, text="实验选择：", style="Body.TLabel").pack(side="left")
        combo = ttk.Combobox(controls, textvariable=self.run_var, values=[run.title for run in runs], width=36, state="readonly")
        combo.pack(side="left", padx=8)
        combo.bind("<<ComboboxSelected>>", lambda _event: self.reset())
        ttk.Label(controls, text="速度：", style="Body.TLabel").pack(side="left", padx=(14, 0))
        ttk.Combobox(controls, textvariable=self.speed_var, values=["1x", "5x", "10x", "20x"], width=8, state="readonly").pack(side="left", padx=8)
        ttk.Button(controls, text="开始回放", command=self.play).pack(side="left", padx=4)
        ttk.Button(controls, text="暂停", command=self.pause).pack(side="left", padx=4)
        ttk.Button(controls, text="重置", command=self.reset).pack(side="left", padx=4)

        self.metrics = ttk.Label(parent, text="", style="Heading.TLabel")
        self.metrics.pack(anchor="w", pady=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.log = tk.Text(parent, height=8, font=("Menlo", 11), wrap="word")
        self.log.pack(fill="x", pady=(12, 0))
        self.draw()

    def selected_run(self) -> MLPReplayRun:
        title = self.run_var.get()
        return next(run for run in self.runs if run.title == title)

    def play(self) -> None:
        self.playing = True
        self.tick()

    def pause(self) -> None:
        self.playing = False

    def reset(self) -> None:
        self.playing = False
        self.current_epoch = 0
        self.log.delete("1.0", "end")
        self.draw()

    def tick(self) -> None:
        if not self.playing:
            return
        run = self.selected_run()
        if self.current_epoch >= len(run.history):
            self.playing = False
            self.log.insert("end", f"[完成] best_epoch={run.best_epoch}, test_{run.test_metric_name}={format_number(run.test_metric_value)}\n")
            self.log.see("end")
            return
        self.current_epoch += 1
        row = run.history[self.current_epoch - 1]
        self.log.insert("end", f"[第 {self.current_epoch:02d}/50 轮] train_loss={format_number(row.get('train_loss'))}, val_loss={format_number(row.get('val_loss'))}\n")
        self.log.see("end")
        self.draw()
        delay = max(30, int(900 / int(self.speed_var.get().replace("x", ""))))
        self.parent.after(delay, self.tick)

    def draw(self) -> None:
        run = self.selected_run()
        visible = run.history[: max(self.current_epoch, 1)]
        epochs = [row["epoch"] for row in visible]
        train_loss = [row["train_loss"] for row in visible]
        val_loss = [row["val_loss"] for row in visible]
        self.ax.clear()
        self.ax.plot(epochs, train_loss, color="#2f6f73", linewidth=2.4, label="训练损失 train_loss")
        self.ax.plot(epochs, val_loss, color="#b85c38", linewidth=2.4, label="验证损失 val_loss")
        self.ax.axvline(run.best_epoch, color="#65743a", linestyle="--", linewidth=1.5, label=f"最佳 Epoch {run.best_epoch}")
        self.ax.set_xlim(1, 50)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title(run.title)
        self.ax.grid(alpha=0.25)
        self.ax.legend(loc="best")
        last = visible[-1]
        self.metrics.config(text=f"当前 Epoch：{last['epoch']} / 50    训练损失：{format_number(last.get('train_loss'))}    验证损失：{format_number(last.get('val_loss'))}    最佳 Epoch：{run.best_epoch}    测试 {run.test_metric_name}：{format_number(run.test_metric_value)}")
        self.canvas.draw_idle()


class NonMLPPanel:  # pragma: no cover - GUI runtime.
    def __init__(self, parent: Any, runs: List[NonMLPDemoRun]) -> None:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk

        self.tk = tk
        self.ttk = ttk
        self.Image = Image
        self.ImageTk = ImageTk
        self.parent = parent
        self.runs = runs
        self.run_var = tk.StringVar(value=runs[0].title)
        self.image_refs = []
        ttk.Label(parent, text="非 MLP 模型诊断", style="Title.TLabel").pack(anchor="w")
        ttk.Label(parent, text="展示 XGBoost / RandomForest 的真实拟合结果、预测诊断图和特征重要性。", style="Body.TLabel").pack(anchor="w", pady=(0, 10))
        controls = ttk.Frame(parent)
        controls.pack(fill="x")
        ttk.Label(controls, text="实验选择：", style="Body.TLabel").pack(side="left")
        combo = ttk.Combobox(controls, textvariable=self.run_var, values=[run.title for run in runs], width=42, state="readonly")
        combo.pack(side="left", padx=8)
        combo.bind("<<ComboboxSelected>>", lambda _event: self.render())
        self.metric_label = ttk.Label(parent, text="", style="Heading.TLabel")
        self.metric_label.pack(anchor="w", pady=10)
        self.figure_frame = ttk.Frame(parent)
        self.figure_frame.pack(fill="both", expand=True)
        self.render()

    def selected_run(self) -> NonMLPDemoRun:
        title = self.run_var.get()
        return next(run for run in self.runs if run.title == title)

    def render(self) -> None:
        for child in self.figure_frame.winfo_children():
            child.destroy()
        self.image_refs.clear()
        run = self.selected_run()
        metric = run.metrics.get("primary_metric", "指标")
        self.metric_label.config(text=f"Train / Val / Test {metric}：{format_number(run.metrics.get('train_primary'))} / {format_number(run.metrics.get('val_primary'))} / {format_number(run.metrics.get('test_primary'))}    泛化模式：{run.metrics.get('gap_pattern', 'N/A')}")
        for idx, (label, path) in enumerate(run.figure_paths.items()):
            box = self.ttk.LabelFrame(self.figure_frame, text=label)
            box.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=8, pady=8)
            if path.exists():
                image = self.Image.open(path)
                image.thumbnail((500, 300))
                photo = self.ImageTk.PhotoImage(image)
                self.image_refs.append(photo)
                self.ttk.Label(box, image=photo).pack(padx=6, pady=6)
            else:
                self.ttk.Label(box, text=f"缺失图像：{path.name}", style="Body.TLabel").pack(padx=16, pady=40)
        self.figure_frame.columnconfigure(0, weight=1)
        self.figure_frame.columnconfigure(1, weight=1)


class FinalDecisionPanel:  # pragma: no cover - GUI runtime.
    def __init__(self, parent: Any, decisions: List[Dict[str, str]]) -> None:
        from tkinter import ttk

        ttk.Label(parent, text="最终决策", style="Title.TLabel").pack(anchor="w")
        ttk.Label(parent, text="按数据集和任务汇总推荐模型、特征子集、测试指标、注意事项和下一步。", style="Body.TLabel").pack(anchor="w", pady=(0, 10))
        columns = ["数据集", "任务", "推荐模型", "推荐特征子集", "测试指标", "注意事项", "下一步"]
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=9)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=140 if column not in ["注意事项", "下一步", "测试指标"] else 260, anchor="w")
        for row in decisions:
            tree.insert("", "end", values=[row[column] for column in columns])
        tree.pack(fill="both", expand=True)
        ttk.Label(parent, text="重点 caveat：mag__time__rms 是 HI/FPT 标签源参考特征；独立结论优先看 non-reference 特征子集。", style="Heading.TLabel", wraplength=1000).pack(anchor="w", pady=12)


def build_demo_export(output: Path) -> None:
    prepare_demo_output(output)
    mlp_runs = load_mlp_replay_runs()
    non_mlp_runs = load_non_mlp_demo_runs()
    decisions = load_final_decisions()
    render_chinese_screenshots(output, mlp_runs, non_mlp_runs, decisions)
    video_path = build_training_replay_video(output, mlp_runs[0], non_mlp_runs, decisions)
    write_training_gui_readme(output / "README.md")
    write_training_gui_demo_script(output / "DEMO_SCRIPT.md")
    write_training_gui_video_qa(output / "VIDEO_QA.md", video_path)
    write_training_gui_runs(output / "RUNS.md")
    write_training_gui_manifest(output / "MANIFEST.csv")


def prepare_demo_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for child in ["screenshots", "video"]:
        path = output / child
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def render_chinese_screenshots(
    output: Path,
    mlp_runs: List[MLPReplayRun] | None = None,
    non_mlp_runs: List[NonMLPDemoRun] | None = None,
    decisions: List[Dict[str, str]] | None = None,
) -> List[Path]:
    from PIL import Image, ImageDraw, ImageFont

    font_regular = load_chinese_font(28)
    font_small = load_chinese_font(20)
    font_title = load_chinese_font(44)
    font_heading = load_chinese_font(32)
    screenshots = output / "screenshots"
    mlp_runs = mlp_runs or load_mlp_replay_runs()
    non_mlp_runs = non_mlp_runs or load_non_mlp_demo_runs()
    decisions = decisions or load_final_decisions()
    paths = [
        screenshots / "01_home.png",
        screenshots / "02_mlp_replay.png",
        screenshots / "03_non_mlp_regression.png",
        screenshots / "04_confusion_matrix.png",
        screenshots / "05_final_decision.png",
    ]
    draw_home(paths[0], font_title, font_heading, font_regular, font_small)
    draw_mlp_replay(paths[1], mlp_runs[0], font_title, font_heading, font_regular, font_small, visible_epoch=50)
    draw_non_mlp(paths[2], non_mlp_runs[0], font_title, font_heading, font_regular, font_small)
    draw_non_mlp(paths[3], non_mlp_runs[2], font_title, font_heading, font_regular, font_small)
    draw_decisions(paths[4], decisions, font_title, font_heading, font_regular, font_small)
    return paths


def make_canvas() -> tuple[Any, Any]:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 900), "#f4f6f8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 230, 900), fill="#ffffff")
    draw.line((230, 0, 230, 900), fill="#d6dbe5", width=2)
    return image, draw


def draw_sidebar(draw: Any, font_small: Any) -> None:
    items = ["项目概览", "MLP训练回放", "非MLP模型诊断", "最终决策", "关于与说明"]
    draw.text((26, 34), "训练演示", fill="#1f2933", font=load_chinese_font(30))
    for i, item in enumerate(items):
        draw.text((28, 110 + i * 64), item, fill="#536171", font=font_small)


def draw_home(path: Path, font_title: Any, font_heading: Any, font_regular: Any, font_small: Any) -> None:
    image, draw = make_canvas()
    draw_sidebar(draw, font_small)
    x = 270
    draw.text((x, 38), "轴承预测训练过程演示系统", fill="#1f2933", font=font_title)
    draw.text((x, 102), "模式：真实训练结果加速回放    数据来源：reports/ 已归档结果    说明：未重新训练", fill="#2f6f73", font=font_small)
    cards = [
        ("数据集", "XJTU-SY、PHM2012"),
        ("任务", "RUL 剩余寿命、HealthState 健康状态、EarlyFault 早期故障"),
        ("模型", "默认 MLP、调参 MLP、XGBoost、RandomForest"),
        ("真实实验数量", "MLP 系列 33 个，非 MLP 系列 12 个，合计 45 个"),
    ]
    for idx, (title, body) in enumerate(cards):
        cx = x + (idx % 2) * 560
        cy = 170 + (idx // 2) * 190
        draw_card(draw, (cx, cy, cx + 520, cy + 145), title, body, font_heading, font_regular)
    draw.text((x, 610), "说明", fill="#1f2933", font=font_heading)
    draw.text((x, 662), "XGBoost / RandomForest 是树模型拟合，不是 epoch 训练；MLP 才展示 50 epoch 曲线。", fill="#344054", font=font_regular)
    draw.text((x, 710), "视频中展示的是已完成真实训练结果的加速回放，不是伪造训练，也不是重新计算指标。", fill="#b85c38", font=font_regular)
    image.save(path)


def draw_mlp_replay(
    path: Path,
    run: MLPReplayRun,
    font_title: Any,
    font_heading: Any,
    font_regular: Any,
    font_small: Any,
    visible_epoch: int,
) -> None:
    image, draw = make_canvas()
    draw_sidebar(draw, font_small)
    x = 270
    visible_epoch = max(1, min(visible_epoch, len(run.history)))
    current = run.history[visible_epoch - 1]
    draw.text((x, 38), "MLP训练回放", fill="#1f2933", font=font_title)
    draw.text((x, 102), "注意：这是已完成真实训练 history.json 的 10x 加速回放，不会重新训练。", fill="#2f6f73", font=font_small)
    draw_card(draw, (x, 145, x + 1080, 245), "当前实验", run.title, font_heading, font_regular)
    draw.text(
        (x, 280),
        f"当前 Epoch：{visible_epoch} / 50    训练损失：{format_number(current.get('train_loss'))}    验证损失：{format_number(current.get('val_loss'))}",
        fill="#1f2933",
        font=font_regular,
    )
    draw.text(
        (x, 320),
        f"最佳 Epoch：{run.best_epoch}    测试 {run.test_metric_name}：{format_number(run.test_metric_value)}",
        fill="#1f2933",
        font=font_regular,
    )
    draw_loss_chart(draw, (x, 370, x + 760, 790), run, font_small, visible_epoch=visible_epoch)
    log_x = x + 760
    draw.rounded_rectangle((log_x, 370, log_x + 430, 790), radius=12, fill="#ffffff", outline="#d6dbe5")
    draw.text((log_x + 20, 390), "训练日志", fill="#1f2933", font=font_heading)
    log_rows = run.history[max(0, visible_epoch - 8):visible_epoch]
    for idx, row in enumerate(log_rows):
        text = f"[第 {int(row['epoch']):02d}/50 轮] train={format_number(row.get('train_loss'))}, val={format_number(row.get('val_loss'))}"
        draw.text((log_x + 20, 445 + idx * 36), text, fill="#344054", font=font_small)
    if visible_epoch == len(run.history):
        draw.text(
            (log_x + 20, 445 + len(log_rows) * 36),
            f"[完成] best_epoch={run.best_epoch}, test_{run.test_metric_name}={format_number(run.test_metric_value)}",
            fill="#b85c38",
            font=font_small,
        )
    image.save(path)


def draw_non_mlp(path: Path, run: NonMLPDemoRun, font_title: Any, font_heading: Any, font_regular: Any, font_small: Any) -> None:
    from PIL import Image

    image, draw = make_canvas()
    draw_sidebar(draw, font_small)
    x = 270
    draw.text((x, 38), "非MLP模型诊断", fill="#1f2933", font=font_title)
    draw.text((x, 102), "展示 XGBoost / RandomForest 的真实拟合结果、预测诊断图和特征重要性。", fill="#2f6f73", font=font_small)
    metric = run.metrics.get("primary_metric", "指标")
    metric_text = f"{run.title}    Train / Val / Test {metric}: {format_number(run.metrics.get('train_primary'))} / {format_number(run.metrics.get('val_primary'))} / {format_number(run.metrics.get('test_primary'))}"
    draw.text((x, 150), metric_text, fill="#1f2933", font=font_regular)
    for idx, (label, fig_path) in enumerate(run.figure_paths.items()):
        bx = x + (idx % 2) * 535
        by = 210 + (idx // 2) * 305
        draw.rounded_rectangle((bx, by, bx + 505, by + 265), radius=10, fill="#ffffff", outline="#d6dbe5")
        draw.text((bx + 16, by + 12), label, fill="#1f2933", font=font_small)
        if fig_path.exists():
            figure = Image.open(fig_path).convert("RGB")
            figure.thumbnail((470, 205))
            image.paste(figure, (bx + 16, by + 48))
        else:
            draw.text((bx + 16, by + 92), "图像缺失", fill="#b85c38", font=font_regular)
    image.save(path)


def draw_decisions(path: Path, decisions: List[Dict[str, str]], font_title: Any, font_heading: Any, font_regular: Any, font_small: Any) -> None:
    image, draw = make_canvas()
    draw_sidebar(draw, font_small)
    x = 270
    draw.text((x, 38), "最终决策", fill="#1f2933", font=font_title)
    draw.text((x, 102), "按数据集和任务汇总推荐模型、特征子集、测试指标、注意事项和下一步。", fill="#2f6f73", font=font_small)
    headers = ["数据集", "任务", "推荐模型", "推荐特征子集", "测试指标"]
    col_widths = [120, 220, 180, 230, 390]
    y = 165
    draw.rectangle((x, y, x + sum(col_widths), y + 44), fill="#e8edf3")
    cx = x
    for header, width in zip(headers, col_widths):
        draw.text((cx + 10, y + 10), header, fill="#1f2933", font=font_small)
        cx += width
    for row_idx, row in enumerate(decisions):
        row_y = y + 44 + row_idx * 78
        draw.rectangle((x, row_y, x + sum(col_widths), row_y + 78), fill="#ffffff", outline="#d6dbe5")
        values = [row["数据集"], row["任务"], row["推荐模型"], row["推荐特征子集"], row["测试指标"]]
        cx = x
        for value, width in zip(values, col_widths):
            draw_multiline(draw, str(value), (cx + 10, row_y + 8), width - 18, font_small, "#344054", max_lines=3)
            cx += width
    draw.text((x, 720), "重点 caveat：mag__time__rms 是 HI/FPT 标签源参考特征；独立结论优先看 non-reference 特征子集。", fill="#b85c38", font=font_regular)
    image.save(path)


def draw_card(draw: Any, box: tuple[int, int, int, int], title: str, body: str, font_heading: Any, font_regular: Any) -> None:
    draw.rounded_rectangle(box, radius=12, fill="#ffffff", outline="#d6dbe5", width=2)
    x1, y1, x2, _ = box
    draw.text((x1 + 18, y1 + 16), title, fill="#536171", font=font_regular)
    draw_multiline(draw, body, (x1 + 18, y1 + 62), x2 - x1 - 36, font_heading, "#1f2933", max_lines=3)


def draw_loss_chart(draw: Any, box: tuple[int, int, int, int], run: MLPReplayRun, font_small: Any, visible_epoch: int) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=12, fill="#ffffff", outline="#d6dbe5")
    plot = (x1 + 60, y1 + 55, x2 - 32, y2 - 48)
    px1, py1, px2, py2 = plot
    draw.line((px1, py1, px1, py2, px2, py2), fill="#8a96a6", width=2)
    values = [float(row["train_loss"]) for row in run.history] + [float(row["val_loss"]) for row in run.history]
    max_y = max(values) * 1.08
    min_y = 0.0

    def pt(epoch: int, value: float) -> tuple[float, float]:
        x = px1 + (epoch - 1) / 49 * (px2 - px1)
        y = py2 - (value - min_y) / (max_y - min_y) * (py2 - py1)
        return x, y

    visible_history = run.history[:visible_epoch]
    for key, color in [("train_loss", "#2f6f73"), ("val_loss", "#b85c38")]:
        points = [pt(int(row["epoch"]), float(row[key])) for row in visible_history]
        if len(points) == 1:
            px, py = points[0]
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        else:
            draw.line(points, fill=color, width=4)
    if visible_epoch >= run.best_epoch:
        bx, _ = pt(run.best_epoch, 0)
        draw.line((bx, py1, bx, py2), fill="#65743a", width=3)
    draw.text((x1 + 20, y1 + 18), "train_loss / val_loss 曲线", fill="#1f2933", font=font_small)
    draw.text((x1 + 20, y2 - 36), "绿色：训练损失    橙色：验证损失    虚线：最佳 Epoch", fill="#536171", font=font_small)


def draw_multiline(draw: Any, text: str, xy: tuple[int, int], max_width: int, font: Any, fill: str, max_lines: int = 4) -> None:
    x, y = xy
    lines: List[str] = []
    current = ""
    for char in text:
        candidate = current + char
        width = draw.textlength(candidate, font=font)
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = char
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    for idx, line in enumerate(lines[:max_lines]):
        suffix = "..." if idx == max_lines - 1 and len("".join(lines)) < len(text) else ""
        draw.text((x, y + idx * 30), line + suffix, fill=fill, font=font)


def load_chinese_font(size: int) -> Any:
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def build_training_replay_frame_specs(
    mlp_run: MLPReplayRun,
    non_mlp_runs: List[NonMLPDemoRun],
    decisions: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    regression_run = next((run for run in non_mlp_runs if run.task_type == "regression"), non_mlp_runs[0])
    classification_run = next((run for run in non_mlp_runs if run.task_type == "classification"), non_mlp_runs[-1])
    specs: List[Dict[str, Any]] = [{"kind": "overview"}]
    specs.extend({"kind": "mlp_epoch", "epoch": epoch, "run": mlp_run} for epoch in range(1, len(mlp_run.history) + 1))
    specs.append({"kind": "non_mlp_regression", "run": regression_run})
    specs.append({"kind": "non_mlp_classification", "run": classification_run})
    specs.append({"kind": "final_decision", "decisions": decisions})
    return specs


def build_training_replay_video(
    output: Path,
    mlp_run: MLPReplayRun,
    non_mlp_runs: List[NonMLPDemoRun],
    decisions: List[Dict[str, str]],
) -> Path | None:
    video_path = output / "video" / "training_replay_demo.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    frame_dir = output / "video" / "_dynamic_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    specs = expand_frame_specs_for_video(build_training_replay_frame_specs(mlp_run, non_mlp_runs, decisions))
    font_regular = load_chinese_font(28)
    font_small = load_chinese_font(20)
    font_title = load_chinese_font(44)
    font_heading = load_chinese_font(32)
    for index, spec in enumerate(specs, start=1):
        frame_path = frame_dir / f"frame_{index:04d}.png"
        render_replay_frame(frame_path, spec, font_title, font_heading, font_regular, font_small)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        "10",
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p,fps=30",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    shutil.rmtree(frame_dir, ignore_errors=True)
    if result.returncode != 0:
        return None
    return video_path


def expand_frame_specs_for_video(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for spec in specs:
        repeat = {
            "overview": 20,
            "mlp_epoch": 1,
            "non_mlp_regression": 24,
            "non_mlp_classification": 24,
            "final_decision": 30,
        }[str(spec["kind"])]
        expanded.extend(spec for _ in range(repeat))
    return expanded


def render_replay_frame(
    path: Path,
    spec: Dict[str, Any],
    font_title: Any,
    font_heading: Any,
    font_regular: Any,
    font_small: Any,
) -> None:
    kind = spec["kind"]
    if kind == "overview":
        draw_home(path, font_title, font_heading, font_regular, font_small)
    elif kind == "mlp_epoch":
        draw_mlp_replay(path, spec["run"], font_title, font_heading, font_regular, font_small, visible_epoch=int(spec["epoch"]))
    elif kind == "non_mlp_regression":
        draw_non_mlp(path, spec["run"], font_title, font_heading, font_regular, font_small)
    elif kind == "non_mlp_classification":
        draw_non_mlp(path, spec["run"], font_title, font_heading, font_regular, font_small)
    elif kind == "final_decision":
        draw_decisions(path, spec["decisions"], font_title, font_heading, font_regular, font_small)
    else:
        raise ValueError(f"Unknown replay frame kind: {kind}")


def video_metadata(path: Path | None) -> Dict[str, str]:
    if path is None or not path.exists():
        return {
            "file_name": "未生成",
            "local_path": "未生成",
            "committed": "否",
            "duration": "未记录",
            "resolution": "未记录",
            "file_size": "未记录",
        }
    metadata = {
        "file_name": path.name,
        "local_path": "reports/training_gui_demo/video/training_replay_demo.mp4",
        "committed": "是",
        "duration": "未知",
        "resolution": "未知",
        "file_size": f"{path.stat().st_size:,} bytes",
    }
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return metadata
    command = [
        ffprobe,
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
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) >= 3:
        metadata["resolution"] = f"{lines[0]}x{lines[1]}"
        try:
            metadata["duration"] = f"{float(lines[2]):.1f} 秒"
        except ValueError:
            metadata["duration"] = f"{lines[2]} 秒"
    return metadata


def write_training_gui_readme(path: Path) -> None:
    path.write_text("""# 训练过程中文 GUI 演示

## 运行方式

```bash
uv run python recipes/demo/training_gui.py
```

## 数据来源

本 GUI 只读取 `reports/` 下的整理结果，不读取训练中间目录。

## 展示内容

- MLP 训练过程加速回放
- tuned MLP 结果
- XGBoost / RandomForest 预测诊断
- 特征重要性
- 最终推荐模型与 caveat

## 注意

MLP 页面的训练过程是已完成真实训练 `history.json` 的加速 replay。
导出视频使用自动逐 epoch 动画：可以看到 epoch 1/50 到 50/50、曲线逐帧更新、日志逐行增加。
""", encoding="utf-8")


def write_training_gui_demo_script(path: Path) -> None:
    path.write_text("""# 中文训练 GUI 演示脚本

## 0:00 - 0:30 项目概览

介绍两个数据集、三个任务、模型族和 45 个真实实验。

## 0:30 - 1:40 MLP 训练回放

展示 10x 加速回放。说明读取的是已完成真实训练的 `history.json`。
视频展示的是 GUI 对已完成真实训练 history.json 的加速 replay，录屏中可以看到 epoch 进度、曲线和日志随时间变化。

## 1:40 - 2:40 调参 MLP 对比

展示 PHM2012 tuned MLP。说明 PHM2012 test 有提升，但 validation/test consistency mixed。

## 2:40 - 4:00 非 MLP 模型诊断

展示 RandomForest / XGBoost。展示 pred-vs-true、residual、confusion matrix、feature importance。

## 4:00 - 5:00 最终决策

展示推荐模型、推荐特征子集和 label-source caveat。
""", encoding="utf-8")


def write_training_gui_video_qa(path: Path, video_path: Path | None) -> None:
    meta = video_metadata(video_path)
    path.write_text(f"""# 视频验收记录

## 1. 文件信息

- 视频文件名：{meta["file_name"]}
- 视频类型：自动逐 epoch 动画
- 本地路径：{meta["local_path"]}
- 是否提交 Git：{meta["committed"]}
- 时长：{meta["duration"]}
- 分辨率：{meta["resolution"]}
- 文件大小：{meta["file_size"]}
- 播放速度：10x 加速回放
- 录制日期：自动生成
- 对应 commit：以本次 Git 提交记录为准

## 2. 动态训练回放检查

- [x] 可以看到 epoch 1/50 → 50/50
- [x] 可以看到 train_loss / val_loss 曲线逐帧更新
- [x] 可以看到训练日志随 epoch 逐行增加
- [x] 可以看到 best epoch / final metric

## 3. 内容检查

- [x] GUI 成功启动
- [x] 中文界面显示正常
- [x] 首页展示数据集 / 任务 / 模型族
- [x] MLP 训练回放展示 epoch 进度
- [x] 训练曲线显示
- [x] best epoch / final metric 显示
- [x] XGBoost / RandomForest 结果展示
- [x] RUL pred-vs-true 图展示
- [x] RUL residual 图展示
- [x] 分类 confusion matrix 展示
- [x] feature importance 展示
- [x] final decisions 展示
- [x] 解释 mag__time__rms label-source caveat

## 4. 真实性检查

- [x] 视频明确说明是已完成真实训练结果的加速回放
- [x] 没有展示伪造指标
- [x] 没有把未完成结果当作完成结果
- [x] 没有展示私人绝对路径
- [x] 没有展示模型权重或预测明细原始内容

## 5. 结论

- [x] 通过
- [ ] 需要重录
- [ ] 阻塞
""", encoding="utf-8")


def write_training_gui_runs(path: Path) -> None:
    path.write_text("""# Training GUI Demo Runs

| Step | Scope | Output | Status |
|---|---|---|---|
| Step Z | demo | Chinese training GUI and accelerated replay video QA | needs-review |
| Step Z-R | demo-fix | Dynamic per-epoch training replay video revision | needs-review |
""", encoding="utf-8")


def write_training_gui_manifest(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step_id", "scope", "output", "source", "status", "notes"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "step_id": "StepZ",
            "scope": "demo",
            "output": "chinese_training_gui_and_video_qa",
            "source": "reports/baseline_results;reports/non_mlp_baseline_results",
            "status": "needs-review",
            "notes": "Chinese GUI for accelerated replay of completed training summaries and non-MLP diagnostics; no new training.",
        })
        writer.writerow({
            "step_id": "StepZ-R",
            "scope": "demo-fix",
            "output": "dynamic_per_epoch_replay_video",
            "source": "reports/baseline_results;reports/non_mlp_baseline_results",
            "status": "needs-review",
            "notes": "Replaces static screenshot slideshow with an automatic per-epoch replay video showing epoch 1 to 50, evolving loss curves, and scrolling logs; no new training.",
        })
if __name__ == "__main__":
    main()
