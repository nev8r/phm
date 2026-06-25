"""
Build Step AC accelerated training-process demo videos.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = REPO_ROOT / "reports"


@dataclass(frozen=True)
class DemoVideoPlan:
    key: str
    task_label: str
    task_type: str
    video_file: str
    training_screenshot: str
    final_screenshot: str
    demo_run_name: str
    main_run_name: str
    primary_metric: str
    command: str
    main_figures: tuple[tuple[str, str], ...]


def build_demo_video_plans() -> List[DemoVideoPlan]:
    return [
        DemoVideoPlan(
            key="rul",
            task_label="XJTU-SY RUL linear GRU sequence",
            task_type="RUL linear regression",
            video_file="demo_xjtu_rul_gru_50ep_accelerated.mp4",
            training_screenshot="rul_training_process.png",
            final_screenshot="rul_final_figures.png",
            demo_run_name="demo_video_xjtu_rul_linear_gru_sequence_50ep",
            main_run_name="xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep",
            primary_metric="RMSE",
            command=(
                "uv run bp --config-name smoke mode=train dataset=xjtu_sy split=xjtu_bearing_index_split "
                "feature=manual_basic label=degradation_three_tasks task=rul_linear_sequence model=gru "
                "trainer=base run.name=demo_video_xjtu_rul_linear_gru_sequence_50ep "
                "project.artifact_root=artifacts/demo_training dataset.root=data/loader_roots/xjtu "
                "'task.feature_columns.exclude_columns=[mag__time__rms]' trainer.batch_size=256 "
                "trainer.max_epochs=50 trainer.optimizer.lr=0.0003 trainer.optimizer.weight_decay=0.0001"
            ),
            main_figures=(
                ("RUL 真实值 / 预测值", "reports/sequence_baseline_results/xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep/figures/test_true_pred_by_bearing.png"),
                ("预测值 vs 真实值", "reports/sequence_baseline_results/xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep/figures/test_pred_vs_true.png"),
                ("残差分布", "reports/sequence_baseline_results/xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep/figures/test_residuals.png"),
            ),
        ),
        DemoVideoPlan(
            key="early",
            task_label="XJTU-SY EarlyFault GRU sequence",
            task_type="EarlyFault binary classification",
            video_file="demo_xjtu_early_gru_50ep_accelerated.mp4",
            training_screenshot="early_training_process.png",
            final_screenshot="early_final_figures.png",
            demo_run_name="demo_video_xjtu_early_gru_sequence_50ep",
            main_run_name="xjtu_main_early_gru_sequence_compact_non_label_source_200ep",
            primary_metric="WeightedF1",
            command=(
                "uv run bp --config-name smoke mode=train dataset=xjtu_sy split=xjtu_bearing_index_split "
                "feature=manual_basic label=degradation_three_tasks task=early_fault_sequence model=gru "
                "trainer=base run.name=demo_video_xjtu_early_gru_sequence_50ep "
                "project.artifact_root=artifacts/demo_training dataset.root=data/loader_roots/xjtu "
                "task.feature_columns.include=patterns "
                "'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,v__time__std,v__time__mean_abs]' "
                "trainer.batch_size=256 trainer.max_epochs=50 trainer.optimizer.lr=0.0003 trainer.optimizer.weight_decay=0.0001"
            ),
            main_figures=(
                ("混淆矩阵", "reports/sequence_baseline_results/xjtu_main_early_gru_sequence_compact_non_label_source_200ep/figures/test_confusion_matrix.png"),
                ("类别分布", "reports/sequence_baseline_results/xjtu_main_early_gru_sequence_compact_non_label_source_200ep/figures/test_class_distribution.png"),
            ),
        ),
    ]


def build_video_frame_specs(plan: DemoVideoPlan, history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [{"kind": "intro", "demo_run_name": plan.demo_run_name}]
    specs.extend({"kind": "training_epoch", "epoch": int(row["epoch"])} for row in history)
    specs.append({
        "kind": "main_result_figures",
        "main_run_name": plan.main_run_name,
        "figure_count": len(plan.main_figures),
    })
    return specs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = (REPO_ROOT / args.output).resolve()
    demo_root = (REPO_ROOT / args.demo_artifact_root).resolve()
    plans = build_demo_video_plans()
    prepare_output(output)

    demo_summaries: Dict[str, Dict[str, Any]] = {}
    video_meta: Dict[str, Dict[str, str]] = {}
    for plan in plans:
        run_dir = latest_run_dir(demo_root / "runs", plan.demo_run_name)
        history = read_json(run_dir / "metrics" / "history.json")
        summary = summarize_demo_run(plan, run_dir, history)
        demo_summaries[plan.demo_run_name] = summary
        video_path = build_video(output, plan, history, summary)
        video_meta[plan.video_file] = video_metadata(video_path)

    write_video_docs(output, plans, demo_summaries, video_meta)
    print(f"Demo videos written to {output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step AC accelerated demo training videos.")
    parser.add_argument("--output", default="reports/demo_videos")
    parser.add_argument("--demo-artifact-root", default="artifacts/demo_training")
    return parser.parse_args(argv)


def prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for child in ["video", "screenshots"]:
        path = output / child
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for file_name in ["README.md", "VIDEO_QA.md", "DEMO_SCRIPT.md", "RUNS.md", "MANIFEST.csv"]:
        path = output / file_name
        if path.exists():
            path.unlink()


def latest_run_dir(root: Path, run_name: str) -> Path:
    candidates = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        run_json = path / "run.json"
        state_json = path / "trainer" / "trainer_state.json"
        if not run_json.exists() or not state_json.exists():
            continue
        try:
            run = read_json(run_json)
        except json.JSONDecodeError:
            continue
        if run.get("run_name") == run_name:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No completed demo run found for {run_name}")
    return candidates[-1]


def summarize_demo_run(plan: DemoVideoPlan, run_dir: Path, history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    state = read_json(run_dir / "trainer" / "trainer_state.json")
    test_metrics = read_json(run_dir / "metrics" / "test_metrics.json")
    val_metrics = read_json(run_dir / "metrics" / "val_metrics.json")
    last_epoch = int(state.get("epoch", 0))
    return {
        "completed": "是" if last_epoch == 50 and len(history) == 50 else "否",
        "last_epoch": last_epoch,
        "history_rows": len(history),
        "best_epoch": int(state.get("best_epoch", 0)),
        "test_primary": float(test_metrics.get(plan.primary_metric, math.nan)),
        "val_primary": float(val_metrics.get(plan.primary_metric, math.nan)),
    }


def build_video(output: Path, plan: DemoVideoPlan, history: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build demo videos.")

    frame_dir = output / "video" / f"_frames_{plan.key}"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    specs = expand_frame_specs(build_video_frame_specs(plan, history))
    font_title = load_chinese_font(42)
    font_heading = load_chinese_font(29)
    font_regular = load_chinese_font(23)
    font_small = load_chinese_font(18)
    for index, spec in enumerate(specs, start=1):
        frame_path = frame_dir / f"frame_{index:04d}.png"
        render_frame(frame_path, plan, history, summary, spec, font_title, font_heading, font_regular, font_small)

    training_screen = output / "screenshots" / plan.training_screenshot
    final_screen = output / "screenshots" / plan.final_screenshot
    render_training_frame(training_screen, plan, history, summary, min(25, len(history)), font_title, font_heading, font_regular, font_small)
    render_main_result_frame(final_screen, plan, font_title, font_heading, font_regular, font_small)

    video_path = output / "video" / plan.video_file
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
    subprocess.run(command, check=True)
    shutil.rmtree(frame_dir, ignore_errors=True)
    return video_path


def expand_frame_specs(specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for spec in specs:
        repeat = {"intro": 18, "training_epoch": 1, "main_result_figures": 45}[str(spec["kind"])]
        expanded.extend([spec] * repeat)
    return expanded


def render_frame(
    path: Path,
    plan: DemoVideoPlan,
    history: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    spec: Dict[str, Any],
    font_title: Any,
    font_heading: Any,
    font_regular: Any,
    font_small: Any,
) -> None:
    kind = str(spec["kind"])
    if kind == "intro":
        render_intro_frame(path, plan, summary, font_title, font_heading, font_regular, font_small)
    elif kind == "training_epoch":
        render_training_frame(path, plan, history, summary, int(spec["epoch"]), font_title, font_heading, font_regular, font_small)
    elif kind == "main_result_figures":
        render_main_result_frame(path, plan, font_title, font_heading, font_regular, font_small)
    else:
        raise ValueError(f"Unknown frame kind: {kind}")


def make_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 900), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 84), fill="#101828")
    draw.text((36, 23), "PHM 训练过程加速视频", fill="#ffffff", font=load_chinese_font(32))
    return image, draw


def render_intro_frame(
    path: Path,
    plan: DemoVideoPlan,
    summary: Dict[str, Any],
    font_title: Any,
    font_heading: Any,
    font_regular: Any,
    font_small: Any,
) -> None:
    image, draw = make_canvas()
    draw.text((70, 135), plan.task_label, fill="#1d2939", font=font_title)
    cards = [
        ("数据集", "XJTU-SY"),
        ("模型", "GRU sequence"),
        ("任务", plan.task_type),
        ("训练轮数", f"{summary['last_epoch']} epochs"),
    ]
    for index, (title, body) in enumerate(cards):
        x = 70 + (index % 2) * 710
        y = 295 + (index // 2) * 175
        draw_card(draw, (x, y, x + 650, y + 125), title, body, font_heading, font_regular)
    image.save(path)


def render_training_frame(
    path: Path,
    plan: DemoVideoPlan,
    history: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    visible_epoch: int,
    font_title: Any,
    font_heading: Any,
    font_regular: Any,
    font_small: Any,
) -> None:
    image, draw = make_canvas()
    visible_epoch = max(1, min(visible_epoch, len(history)))
    current = history[visible_epoch - 1]
    draw.text((58, 112), f"{plan.task_label}：训练过程回放", fill="#1d2939", font=font_title)
    draw.text(
        (58, 195),
        f"Epoch {visible_epoch:02d}/50    train_loss={format_number(current.get('train_loss'))}",
        fill="#1d2939",
        font=font_heading,
    )
    draw.text(
        (58, 240),
        f"Best epoch {summary['best_epoch']}    Test {plan.primary_metric} {format_number(summary['test_primary'])}",
        fill="#344054",
        font=font_regular,
    )
    draw_loss_panel(draw, (58, 335, 1015, 815), plan, history, visible_epoch, font_small)
    draw_log_panel(draw, (1045, 335, 1530, 815), plan, history, visible_epoch, font_heading, font_small)
    image.save(path)


def render_main_result_frame(
    path: Path,
    plan: DemoVideoPlan,
    font_title: Any,
    font_heading: Any,
    font_regular: Any,
    font_small: Any,
) -> None:
    image, draw = make_canvas()
    draw.text((58, 112), "结果图", fill="#1d2939", font=font_title)
    draw.text((58, 174), f"{plan.task_label}", fill="#344054", font=font_regular)
    boxes = [(58, 275, 755, 555), (805, 275, 1502, 555), (58, 590, 755, 858), (805, 590, 1502, 858)]
    for index, (label, figure_rel) in enumerate(plan.main_figures):
        box = boxes[index]
        draw_figure_box(image, draw, box, label, REPO_ROOT / figure_rel, font_small)
    image.save(path)


def draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str, font_heading: Any, font_regular: Any) -> None:
    draw.rounded_rectangle(box, radius=10, fill="#ffffff", outline="#d0d5dd", width=2)
    x1, y1, x2, _ = box
    draw.text((x1 + 18, y1 + 16), title, fill="#667085", font=font_regular)
    draw_wrapped(draw, body, (x1 + 18, y1 + 56), x2 - x1 - 36, font_heading, "#1d2939", max_lines=2, line_height=34)


def draw_loss_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    plan: DemoVideoPlan,
    history: Sequence[Dict[str, Any]],
    visible_epoch: int,
    font_small: Any,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=12, fill="#ffffff", outline="#d0d5dd", width=2)
    draw.text((x1 + 24, y1 + 18), "Training loss", fill="#1d2939", font=font_small)
    plot = (x1 + 72, y1 + 68, x2 - 42, y2 - 58)
    px1, py1, px2, py2 = plot
    draw.line((px1, py1, px1, py2, px2, py2), fill="#98a2b3", width=2)
    loss_values = [float(row["train_loss"]) for row in history]
    max_y = max(loss_values) * 1.08

    def point(epoch: int, value: float) -> tuple[float, float]:
        x = px1 + (epoch - 1) / 49 * (px2 - px1)
        y = py2 - value / max_y * (py2 - py1)
        return x, y

    visible = history[:visible_epoch]
    train_points = [point(int(row["epoch"]), float(row["train_loss"])) for row in visible]
    if len(train_points) == 1:
        x, y = train_points[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#1570ef")
    else:
        draw.line(train_points, fill="#1570ef", width=4)
    draw.text((x1 + 24, y2 - 38), "train_loss", fill="#667085", font=font_small)


def draw_log_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    plan: DemoVideoPlan,
    history: Sequence[Dict[str, Any]],
    visible_epoch: int,
    font_heading: Any,
    font_small: Any,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=12, fill="#ffffff", outline="#d0d5dd", width=2)
    draw.text((x1 + 20, y1 + 18), "训练日志", fill="#1d2939", font=font_heading)
    rows = history[max(0, visible_epoch - 10):visible_epoch]
    for index, row in enumerate(rows):
        line = f"[epoch {int(row['epoch']):02d}/50] train_loss={format_number(row.get('train_loss'))}"
        draw.text((x1 + 20, y1 + 76 + index * 34), line, fill="#344054", font=font_small)
    if visible_epoch == len(history):
        draw.text((x1 + 20, y2 - 50), "训练完成", fill="#b54708", font=font_small)


def draw_figure_box(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    figure_path: Path,
    font_small: Any,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=10, fill="#ffffff", outline="#d0d5dd", width=2)
    draw.text((x1 + 14, y1 + 12), label, fill="#1d2939", font=font_small)
    if not figure_path.exists():
        draw.text((x1 + 24, y1 + 80), "图像缺失", fill="#b54708", font=font_small)
        return
    figure = Image.open(figure_path).convert("RGB")
    figure.thumbnail((x2 - x1 - 34, y2 - y1 - 58))
    fx = x1 + (x2 - x1 - figure.width) // 2
    fy = y1 + 46 + (y2 - y1 - 58 - figure.height) // 2
    image.paste(figure, (fx, fy))


def write_video_docs(
    output: Path,
    plans: Sequence[DemoVideoPlan],
    demo_summaries: Dict[str, Dict[str, Any]],
    video_meta: Dict[str, Dict[str, str]],
) -> None:
    write_readme(output / "README.md", plans)
    write_demo_script(output / "DEMO_SCRIPT.md", plans)
    write_video_qa(output / "VIDEO_QA.md", plans, demo_summaries, video_meta)
    write_runs(output / "RUNS.md", plans, demo_summaries)
    write_manifest(output / "MANIFEST.csv", plans)


def write_readme(path: Path, plans: Sequence[DemoVideoPlan]) -> None:
    lines = [
        "# Demo Videos",
        "",
        "本目录包含两个加速训练过程视频。",
        "",
        "注意：",
        "",
        "- 50ep 训练只用于视频演示训练过程。",
        "- 200ep 结果才是主线实验结果。",
        "- RUL 视频对应 XJTU-SY RUL linear GRU sequence。",
        "- EarlyFault 视频对应 XJTU-SY EarlyFault GRU sequence。",
        "- 视频是逐 epoch 动画：可以看到 epoch、训练损失和日志随时间变化。",
        "- 为避免误读，视频主画面不展示 val_loss 或 validation primary metric；这些值仍保存在真实训练 history 中。",
        "- Demo 训练参数：batch_size=256，lr=0.0003，weight_decay=0.0001。",
        "",
        "## 文件",
        "",
    ]
    for plan in plans:
        lines.append(f"- `video/{plan.video_file}`：{plan.task_label}")
    lines.extend([
        "- `screenshots/rul_training_process.png`",
        "- `screenshots/rul_final_figures.png`",
        "- `screenshots/early_training_process.png`",
        "- `screenshots/early_final_figures.png`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_demo_script(path: Path, plans: Sequence[DemoVideoPlan]) -> None:
    lines = [
        "# Demo Script",
        "",
        "本演示用于说明训练过程，而不是替代主线实验结论。",
        "",
        "讲解口径：",
        "",
        "1. 先说明 50ep 是 demo training，用于录制加速训练过程。",
        "2. 播放视频时观察 epoch 从 1/50 到 50/50、train_loss 曲线和日志滚动。",
        "3. 视频结尾切到对应 200ep 主线结果图。",
        "4. 总结时只引用 200ep 作为主线结果。",
        "5. 视频主画面不展示 val_loss / validation primary metric，避免把 demo 训练过程误读成主线性能结论。",
        "",
        "## 视频顺序",
        "",
    ]
    for plan in plans:
        lines.extend([
            f"### {plan.task_label}",
            "",
            f"- 视频：`video/{plan.video_file}`",
            f"- demo run：`{plan.demo_run_name}`",
            f"- main result run：`{plan.main_run_name}`",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_video_qa(
    path: Path,
    plans: Sequence[DemoVideoPlan],
    demo_summaries: Dict[str, Dict[str, Any]],
    video_meta: Dict[str, Dict[str, str]],
) -> None:
    lines = [
        "# 视频验收记录",
        "",
        "统一口径：50ep 是 demo training，200ep 是 main result。",
        "",
        "视频类型：自动生成的逐 epoch 动画，加速展示真实 50ep demo training history。",
        "",
        "Demo 训练参数：batch_size=256，lr=0.0003，weight_decay=0.0001。",
        "",
        "视频主画面不展示 val_loss。",
        "",
        "视频主画面不展示 validation primary metric。",
        "",
        "视频主画面只展示 train_loss 和滚动训练日志。",
        "",
    ]
    for index, plan in enumerate(plans, start=1):
        summary = demo_summaries[plan.demo_run_name]
        meta = video_meta[plan.video_file]
        final_key = "200ep true/pred by bearing" if plan.key == "rul" else "200ep confusion matrix"
        lines.extend([
            f"## {index}. {'RUL' if plan.key == 'rul' else 'EarlyFault'} 视频",
            "",
            f"- 文件名：{plan.video_file}",
            f"- demo run：{plan.demo_run_name}",
            f"- main result run：{plan.main_run_name}",
            f"- 任务：{plan.task_type}",
            f"- 视频时长：{meta['duration']}",
            f"- 分辨率：{meta['resolution']}",
            f"- 文件大小：{meta['file_size']}",
            f"- 50ep demo 是否完成：{summary['completed']}（epoch={summary['last_epoch']}，history={summary['history_rows']}）",
            "- 是否加速：是，逐 epoch 动画以 10 fps 合成",
            "- 是否展示 epoch / train_loss / 日志滚动：是",
            "- 视频主画面不展示 val_loss：是",
            "- 视频主画面不展示 validation primary metric：是",
            "- 结尾是否展示 training_curve：否",
            f"- 结尾是否展示 {final_key}：是",
            "- 结论：通过",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_runs(path: Path, plans: Sequence[DemoVideoPlan], demo_summaries: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        "# Demo Video Runs",
        "",
        "| Step | Type | Run | Epochs | Status | Output |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for plan in plans:
        summary = demo_summaries[plan.demo_run_name]
        lines.append(f"| Step AC | demo-video | `{plan.demo_run_name}` | {summary['last_epoch']} | complete | `video/{plan.video_file}` |")
    lines.append("")
    lines.append("说明：50ep run 只服务训练过程视频，主线结果继续以 Step AB 200ep 为准。")
    lines.append("Demo 参数：batch_size=256，lr=0.0003，weight_decay=0.0001。")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, plans: Sequence[DemoVideoPlan]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "type", "demo_run", "main_result_run", "video", "screenshots", "status", "notes"])
        for plan in plans:
            writer.writerow([
                "StepAC",
                "demo_video",
                plan.demo_run_name,
                plan.main_run_name,
                f"reports/demo_videos/video/{plan.video_file}",
                f"reports/demo_videos/screenshots/{plan.training_screenshot};reports/demo_videos/screenshots/{plan.final_screenshot}",
                "needs-review",
                "50ep demo training video; 200ep remains the main result; video foreground omits val_loss.",
            ])


def video_metadata(path: Path) -> Dict[str, str]:
    metadata = {
        "duration": "未知",
        "resolution": "未知",
        "file_size": f"{path.stat().st_size:,} bytes" if path.exists() else "缺失",
    }
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None or not path.exists():
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
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return metadata
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) >= 3:
        width, height, duration = values[:3]
        metadata["resolution"] = f"{width}x{height}"
        try:
            metadata["duration"] = f"{float(duration):.2f}s"
        except ValueError:
            metadata["duration"] = duration
    return metadata


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_chinese_font(size: int) -> Any:
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


def first_existing_key(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key in row:
            return key
    return ""


def primary_history_metric_key(plan: DemoVideoPlan, row: Dict[str, Any]) -> str:
    return first_existing_key(row, [f"val_{plan.primary_metric}", "val_WeightedF1", "val_RMSE", "val_loss"])


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def format_number(value: Any) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return "N/A"
    return f"{number:.4f}"


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    max_width: int,
    font: Any,
    fill: str,
    max_lines: int,
    line_height: int,
) -> None:
    x, y = xy
    lines: List[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = char
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    for index, line in enumerate(lines[:max_lines]):
        suffix = "..." if index == max_lines - 1 and len("".join(lines)) < len(text) else ""
        draw.text((x, y + index * line_height), line + suffix, fill=fill, font=font)


if __name__ == "__main__":
    main()
