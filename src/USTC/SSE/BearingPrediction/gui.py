"""
Streamlit GUI support module

this file is for serving classroom demo helpers for bearing PHM workflows

created by zy

copyright USTC

2026
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.analysis import (
    build_dataset_cards,
    render_model_architecture_diagrams,
    task_relationship_summary,
)
from USTC.SSE.BearingPrediction.workflow import (
    evaluate_saved_training_run,
    predict_feature_csv_with_run,
    run_benchmark,
    run_paper_training,
)


DEFAULT_BENCHMARK_RUN = Path("outputs/runs/20260618_153347_benchmark_all")
DEFAULT_TRAIN_RUNS = {
    "rul": Path("outputs/runs/20260618_143026_train_rul"),
    "fault": Path("outputs/runs/20260618_153012_train_fault"),
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _format_float(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _new_run_dir(output_root: Path | str, command: str, task: str) -> Path:
    root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{timestamp}_{command}_{task}"
    counter = 1
    while run_dir.exists():
        run_dir = root / f"{timestamp}_{command}_{task}_{counter}"
        counter += 1
    (run_dir / "figures").mkdir(parents=True, exist_ok=False)
    return run_dir


def list_run_directories(
    output_root: Path | str = Path("outputs/runs"),
    *,
    command: str | None = None,
    task: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    if not root.exists():
        return []
    runs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        config = _load_json(child / "config.json")
        if not config:
            continue
        if command is not None and config.get("command") != command:
            continue
        if task is not None and config.get("task") != task:
            continue
        runs.append(
            {
                "name": child.name,
                "path": child,
                "command": config.get("command", ""),
                "task": config.get("task", ""),
                "sample": bool(config.get("sample", False)),
                "mtime": child.stat().st_mtime,
            }
        )
    return sorted(runs, key=lambda item: item["mtime"], reverse=True)


def summarize_run(run_dir: Path | str) -> dict[str, Any]:
    path = Path(run_dir)
    config = _load_json(path / "config.json")
    metrics = _load_json(path / "metrics.json")
    model_summary = _load_json(path / "model_summary.json")
    task = str(config.get("task", metrics.get("task", "")))
    command = str(config.get("command", metrics.get("command", "")))
    figure_dir = path / "figures"
    summary: dict[str, Any] = {
        "path": path,
        "name": path.name,
        "command": command,
        "task": task,
        "sample": bool(config.get("sample", False)),
        "metrics": {},
        "figures": {},
        "raw_config": config,
        "raw_metrics": metrics,
        "model_summary": model_summary,
    }
    if command == "train":
        test = metrics.get("test", {})
        if task == "rul":
            summary["metrics"] = {
                "MSE": _format_float(test.get("mse")),
                "RMSE": _format_float(test.get("rmse")),
                "MAE": _format_float(test.get("mae")),
                "R2": _format_float(test.get("r2")),
                "PHM Score": _format_float(test.get("phm2012_score")),
            }
            summary["figures"] = {
                "训练曲线": figure_dir / "training_curve.png",
                "RUL 预测曲线": figure_dir / "rul_prediction_curve.png",
                "分轴承 RUL 曲线": figure_dir / "rul_prediction_by_bearing.png",
            }
        elif task == "fault":
            summary["metrics"] = {
                "Accuracy": _format_float(test.get("accuracy")),
                "Macro F1": _format_float(test.get("macro_f1")),
                "Weighted F1": _format_float(test.get("weighted_f1")),
                "Fault F1": _format_float(test.get("fault_f1")),
            }
            summary["figures"] = {
                "训练曲线": figure_dir / "training_curve.png",
                "混淆矩阵": figure_dir / "fault_confusion_matrix.png",
            }
    elif command == "benchmark":
        csv_path = path / "benchmark_results.csv"
        rows = []
        if csv_path.exists():
            rows = pd.read_csv(csv_path).to_dict(orient="records")
        summary["benchmark_rows"] = rows
        summary["figures"] = {
            "RUL baseline": figure_dir / "rul_benchmark.png",
            "Fault baseline": figure_dir / "fault_benchmark.png",
        }
    return summary


def _dataset_detection_from_names(names: list[str]) -> dict[str, Any]:
    normalized = [name.replace("\\", "/").strip("/") for name in names if name and not name.endswith("/")]
    lower = [name.lower() for name in normalized]
    phm_checks = {
        "Learning_set": any("learning_set/" in name for name in lower),
        "Full_Test_Set": any("full_test_set/" in name for name in lower),
        "acc_*.csv": any(Path(name).name.startswith("acc_") and name.endswith(".csv") for name in lower),
        "Bearing folders": any("bearing1_" in name or "bearing2_" in name or "bearing3_" in name for name in lower),
    }
    xjtu_conditions = ("35hz12kn", "37.5hz11kn", "40hz10kn")
    xjtu_checks = {
        "Condition folders": any(condition in name for condition in xjtu_conditions for name in lower),
        "Bearing folders": any("bearing1_" in name or "bearing2_" in name or "bearing3_" in name for name in lower),
        "*.csv": any(name.endswith(".csv") for name in lower),
    }
    if all(phm_checks.values()):
        return {
            "valid": True,
            "dataset": "PHM2012",
            "message": "识别为 PHM2012 标准目录结构，可用于 RUL 特征缓存与训练。",
            "evidence": [key for key, ok in phm_checks.items() if ok],
            "missing": [],
            "file_count": len(normalized),
        }
    if all(xjtu_checks.values()):
        return {
            "valid": True,
            "dataset": "XJTU-SY",
            "message": "识别为 XJTU-SY 标准目录结构，可用于故障诊断特征缓存与训练。",
            "evidence": [key for key, ok in xjtu_checks.items() if ok],
            "missing": [],
            "file_count": len(normalized),
        }
    missing = [f"PHM2012:{key}" for key, ok in phm_checks.items() if not ok]
    missing.extend([f"XJTU-SY:{key}" for key, ok in xjtu_checks.items() if not ok])
    return {
        "valid": False,
        "dataset": "unknown",
        "message": "未识别为当前支持的数据集结构；可上传 PHM2012/XJTU-SY zip，或上传特征 CSV 做演示预测。",
        "evidence": [key for key, ok in {**phm_checks, **xjtu_checks}.items() if ok],
        "missing": missing,
        "file_count": len(normalized),
    }


def _inspect_single_csv(path: Path) -> dict[str, Any]:
    try:
        preview = pd.read_csv(path, nrows=256)
    except Exception as exc:
        return {
            "valid": False,
            "dataset": "single_csv",
            "message": f"CSV 读取失败：{exc}",
            "evidence": [],
            "missing": ["readable CSV"],
            "numeric_columns": 0,
            "row_preview": 0,
        }
    numeric = preview.select_dtypes(include=[np.number])
    valid = numeric.shape[1] > 0 and len(preview) > 0
    return {
        "valid": valid,
        "dataset": "single_csv",
        "message": (
            "识别为特征 CSV：数值列可用于演示级预测；若要直接加载模型，数值列数量需等于模型输入维度。"
            if valid
            else "CSV 中没有可用的数值特征列。"
        ),
        "evidence": [f"{numeric.shape[1]} numeric columns", f"{len(preview)} preview rows"] if valid else [],
        "missing": [] if valid else ["numeric feature columns"],
        "numeric_columns": int(numeric.shape[1]),
        "row_preview": int(len(preview)),
    }


def inspect_uploaded_dataset(path: Path | str) -> dict[str, Any]:
    upload_path = Path(path)
    if upload_path.is_dir():
        names = [str(item.relative_to(upload_path)) for item in upload_path.rglob("*") if item.is_file()]
        return _dataset_detection_from_names(names)
    if upload_path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(upload_path) as archive:
                return _dataset_detection_from_names(archive.namelist())
        except zipfile.BadZipFile:
            return {
                "valid": False,
                "dataset": "unknown",
                "message": "zip 文件无法解压或格式损坏。",
                "evidence": [],
                "missing": ["valid zip archive"],
                "file_count": 0,
            }
    if upload_path.suffix.lower() == ".csv":
        return _inspect_single_csv(upload_path)
    return {
        "valid": False,
        "dataset": "unknown",
        "message": "当前仅支持 zip、目录或特征 CSV。",
        "evidence": [],
        "missing": ["zip", "directory", "csv"],
        "file_count": 0,
    }


def _default_train_run(task: str) -> Path | None:
    preferred = DEFAULT_TRAIN_RUNS.get(task)
    if preferred and preferred.exists():
        return preferred
    runs = list_run_directories(command="train", task=task)
    return runs[0]["path"] if runs else None


def _default_benchmark_run() -> Path | None:
    if DEFAULT_BENCHMARK_RUN.exists():
        return DEFAULT_BENCHMARK_RUN
    runs = list_run_directories(command="benchmark")
    return runs[0]["path"] if runs else None


def _selectbox_runs(st, label: str, runs: list[dict[str, Any]], preferred: Path | None = None) -> Path | None:
    if not runs:
        st.info("还没有可用 run。")
        return None
    index = 0
    if preferred is not None:
        for idx, item in enumerate(runs):
            if Path(item["path"]) == preferred:
                index = idx
                break
    selected = st.selectbox(
        label,
        options=runs,
        index=index,
        format_func=lambda item: f"{item['name']}  {'sample' if item['sample'] else 'full'}",
    )
    return selected["path"] if selected else None


def _show_metrics(st, metrics: dict[str, str]) -> None:
    if not metrics:
        st.info("暂无指标。")
        return
    columns = st.columns(min(len(metrics), 5))
    for column, (name, value) in zip(columns, metrics.items()):
        column.metric(name, value)


def _show_figures(st, figures: dict[str, Path | str]) -> None:
    visible = [(name, Path(path)) for name, path in figures.items() if Path(path).exists()]
    if not visible:
        st.info("暂无可展示图像。")
        return
    columns = st.columns(2)
    for index, (name, path) in enumerate(visible):
        columns[index % 2].image(str(path), caption=name, use_container_width=True)


def _show_benchmark(st, run_dir: Path | None) -> None:
    if run_dir is None:
        st.info("还没有 benchmark run。")
        return
    summary = summarize_run(run_dir)
    rows = summary.get("benchmark_rows", [])
    st.caption(str(run_dir))
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    _show_figures(st, summary.get("figures", {}))


def _render_overview(st) -> None:
    asset_dir = Path("outputs/gui_demo/assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    figures = render_model_architecture_diagrams(asset_dir)
    cards = build_dataset_cards()
    relation = task_relationship_summary()
    st.subheader("系统概览")
    st.write("系统围绕轴承振动信号，完成数据加载、特征工程、任务建模、训练评估和结果可视化。")
    dataset_columns = st.columns(2)
    for column, (_, card) in zip(dataset_columns, cards.items()):
        with column:
            st.markdown(f"**{card['name']}**")
            st.write(f"采样频率：{card['sampling_rate_hz']} Hz")
            st.write(f"工况：{'；'.join(card['operating_conditions'])}")
            st.write(f"任务：{', '.join(card['tasks'])}")
            st.write(f"标签：{card['label_source']}")
    st.markdown("**任务关系**")
    st.write(relation["shared_pipeline"])
    st.write(relation["relationship"])
    _show_figures(st, {key: Path(value) for key, value in figures.items()})


def _append_log(st, message: str) -> None:
    st.session_state.setdefault("gui_logs", [])
    st.session_state["gui_logs"].append(f"{datetime.now().strftime('%H:%M:%S')}  {message}")


def _run_reload_action(st, task: str, run_dir: Path | None) -> None:
    if run_dir is None:
        st.warning("没有可加载的训练 run。")
        return
    with st.spinner(f"加载 {task.upper()} 模型并复推理..."):
        result = evaluate_saved_training_run(run_dir, device_name="auto")
    st.session_state["last_result"] = result
    _append_log(st, f"复推理完成：{result['output_dir']}")


def _run_train_demo_action(st, task: str) -> None:
    run_dir = _new_run_dir("outputs/runs", "train", task)
    with st.spinner(f"运行 {task.upper()} smoke 训练 demo..."):
        run_paper_training(task=task, preset="smoke", sample=True, device_name="cpu", run_dir=run_dir)
    st.session_state["selected_run_override"] = run_dir
    _append_log(st, f"训练 demo 完成：{run_dir}")


def _run_benchmark_demo_action(st) -> None:
    run_dir = _new_run_dir("outputs/runs", "benchmark", "all")
    with st.spinner("运行 sample benchmark demo..."):
        run_benchmark(task="all", baselines="linear,forest", sample=True, run_dir=run_dir)
    st.session_state["benchmark_run_override"] = run_dir
    _append_log(st, f"benchmark demo 完成：{run_dir}")


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="轴承寿命预测与故障诊断系统演示台",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1.4rem; }
        section[data-testid="stSidebar"] { min-width: 320px; }
        div[data-testid="stMetric"] { background: #f8fafc; border: 1px solid #d9e2ec; padding: 0.7rem; border-radius: 6px; }
        .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("轴承寿命预测与故障诊断系统演示台")
    st.caption("课堂展示入口：训练 demo、benchmark、模型复推理、上传结构检测。")

    task = st.sidebar.radio("任务选择", ["rul", "fault"], format_func=lambda value: "RUL 寿命预测" if value == "rul" else "Fault 故障诊断")
    train_runs = list_run_directories(command="train", task=task)
    preferred = st.session_state.get("selected_run_override") or _default_train_run(task)
    selected_run = _selectbox_runs(st.sidebar, "选择训练 run", train_runs, preferred=Path(preferred) if preferred else None)
    benchmark_override = st.session_state.get("benchmark_run_override")
    benchmark_run = Path(benchmark_override) if benchmark_override else _default_benchmark_run()

    if st.sidebar.button("加载 RUL 模型复推理", use_container_width=True):
        _run_reload_action(st, "rul", _default_train_run("rul"))
    if st.sidebar.button("运行 RUL 训练 Demo", use_container_width=True):
        _run_train_demo_action(st, "rul")
    if st.sidebar.button("运行 Benchmark Demo", use_container_width=True):
        _run_benchmark_demo_action(st)
    if st.sidebar.button("加载 Fault 模型复推理", use_container_width=True):
        _run_reload_action(st, "fault", _default_train_run("fault"))

    local_dir = st.sidebar.text_input("本地数据目录检测", value="")
    uploaded = st.sidebar.file_uploader("上传 zip 或特征 CSV", type=["zip", "csv"])

    overview_tab, train_tab, benchmark_tab, reload_tab, upload_tab = st.tabs(
        ["系统概览", "训练 Demo", "Benchmark", "加载训练好的模型", "上传数据集"]
    )

    with overview_tab:
        _render_overview(st)

    with train_tab:
        st.subheader("训练 Demo")
        st.write("现场演示使用 sample/smoke 训练，完整结果从已保存 full run 加载。")
        if selected_run is not None:
            selected_summary = summarize_run(selected_run)
            _show_metrics(st, selected_summary.get("metrics", {}))
            _show_figures(st, selected_summary.get("figures", {}))
        if st.button("运行当前任务训练 Demo", use_container_width=True):
            _run_train_demo_action(st, task)
            st.rerun()

    with benchmark_tab:
        st.subheader("Benchmark")
        st.write("默认展示 full benchmark；现场按钮运行 sample baseline 方便演示。")
        _show_benchmark(st, benchmark_run)
        if st.button("运行 Sample Benchmark", use_container_width=True):
            _run_benchmark_demo_action(st)
            st.rerun()

    with reload_tab:
        st.subheader("加载训练好的模型")
        st.write("加载 checkpoint 和 standardizer，在固定测试集重新推理并生成图表。")
        if selected_run is not None:
            st.caption(str(selected_run))
            if st.button("加载当前选择 run 复推理", use_container_width=True):
                _run_reload_action(st, task, selected_run)
        result = st.session_state.get("last_result")
        if result:
            _show_metrics(
                st,
                {
                    name.upper() if len(name) <= 4 else name: _format_float(value)
                    for name, value in result.get("metrics", {}).items()
                    if isinstance(value, (int, float))
                },
            )
            _show_figures(st, {key: Path(value) for key, value in result.get("figures", {}).items()})
            st.caption(result.get("output_dir", ""))

    with upload_tab:
        st.subheader("上传数据集")
        st.write("支持 PHM2012/XJTU-SY 标准 zip 或目录检测；特征 CSV 可在列数匹配时做演示级预测。")
        inspection: dict[str, Any] | None = None
        upload_path: Path | None = None
        if local_dir:
            candidate = Path(local_dir).expanduser()
            if candidate.exists():
                inspection = inspect_uploaded_dataset(candidate)
                upload_path = candidate
            else:
                st.error("本地目录不存在。")
        if uploaded is not None:
            temp_root = Path(tempfile.mkdtemp(prefix="phm_gui_upload_"))
            upload_path = temp_root / uploaded.name
            upload_path.write_bytes(uploaded.getbuffer())
            inspection = inspect_uploaded_dataset(upload_path)
        if inspection:
            st.json(inspection)
            if (
                inspection.get("dataset") == "single_csv"
                and inspection.get("valid")
                and selected_run is not None
                and upload_path is not None
                and st.button("对特征 CSV 做演示预测", use_container_width=True)
            ):
                try:
                    prediction = predict_feature_csv_with_run(selected_run, upload_path, device_name="auto")
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"预测完成：{prediction['output_dir']}")
                    _show_figures(st, {key: Path(value) for key, value in prediction.get("figures", {}).items()})

    st.divider()
    st.subheader("运行日志与输出目录")
    logs = st.session_state.get("gui_logs", [])
    st.code("\n".join(logs[-12:]) if logs else "等待操作。", language="text")


if __name__ == "__main__":
    main()
