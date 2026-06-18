"""
Streamlit GUI support module

this file is for serving an operational bearing PHM workbench

created by zy

copyright USTC

2026
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.gui_jobs import (
    DEFAULT_JOBS_ROOT,
    list_jobs,
    poll_job,
    read_job_log,
    start_cli_job,
)


DEFAULT_PHM_ROOT = Path("data/loader_roots/phm2012")
DEFAULT_XJTU_ROOT = Path("data/loader_roots/xjtu")
DEFAULT_BENCHMARK_RUN = Path("outputs/runs/20260618_153347_benchmark_all")
DEFAULT_TRAIN_RUNS = {
    "rul": Path("outputs/runs/20260618_143026_train_rul"),
    "fault": Path("outputs/runs/20260618_153012_train_fault"),
}
GUI_OUTPUT_ROOT = Path("outputs/gui")


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
    return run_dir


def _mtime(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _cache_status(cache_dir: Path | str, task: str) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    filename = {
        "rul": "phm2012_rul_fft256_full.npz",
        "fault": "xjtu_binary_fault_diagnosis_fft256_full.npz",
    }[task]
    path = cache_dir / filename
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "mtime": _mtime(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "feature_shape": [],
        "target_shape": [],
    }
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as cache:
                status["feature_shape"] = list(cache["features"].shape) if "features" in cache.files else []
                targets_key = "targets" if "targets" in cache.files else "labels"
                status["target_shape"] = list(cache[targets_key].shape) if targets_key in cache.files else []
        except (OSError, KeyError, ValueError):
            status["read_error"] = True
    return status


def _inspect_phm2012_root(root: Path, cache_dir: Path | str) -> dict[str, Any]:
    learning_dir = root / "Learning_set"
    test_dir = root / "Full_Test_Set"
    files = list(root.glob("Learning_set/Bearing*/acc_*.csv")) + list(root.glob("Full_Test_Set/Bearing*/acc_*.csv"))
    bearing_dirs = {item.parent.name for item in files}
    checks = {
        "root_exists": root.exists(),
        "Learning_set": learning_dir.exists(),
        "Full_Test_Set": test_dir.exists(),
        "acc_files": bool(files),
        "bearing_folders": bool(bearing_dirs),
    }
    return {
        "dataset": "PHM2012",
        "task": "rul",
        "root": str(root),
        "valid": all(checks.values()),
        "checks": checks,
        "missing": [key for key, ok in checks.items() if not ok],
        "bearing_count": len(bearing_dirs),
        "file_count": len(files),
        "split_count": int(learning_dir.exists()) + int(test_dir.exists()),
        "cache": _cache_status(cache_dir, "rul"),
    }


def _inspect_xjtu_root(root: Path, cache_dir: Path | str) -> dict[str, Any]:
    known_conditions = ("35Hz12kN", "37.5Hz11kN", "40Hz10kN")
    condition_dirs = [root / name for name in known_conditions if (root / name).exists()]
    files = []
    for condition_dir in condition_dirs:
        files.extend(condition_dir.glob("Bearing*/*.csv"))
    bearing_dirs = {item.parent.name for item in files}
    checks = {
        "root_exists": root.exists(),
        "condition_folders": bool(condition_dirs),
        "bearing_folders": bool(bearing_dirs),
        "csv_files": bool(files),
    }
    return {
        "dataset": "XJTU-SY",
        "task": "fault",
        "root": str(root),
        "valid": all(checks.values()),
        "checks": checks,
        "missing": [key for key, ok in checks.items() if not ok],
        "bearing_count": len(bearing_dirs),
        "file_count": len(files),
        "condition_count": len(condition_dirs),
        "cache": _cache_status(cache_dir, "fault"),
    }


def inspect_dataset_roots(
    phm_root: Path | str = DEFAULT_PHM_ROOT,
    xjtu_root: Path | str = DEFAULT_XJTU_ROOT,
    *,
    cache_dir: Path | str = Path("cache/paper_features"),
) -> dict[str, dict[str, Any]]:
    return {
        "PHM2012": _inspect_phm2012_root(Path(phm_root), cache_dir),
        "XJTU-SY": _inspect_xjtu_root(Path(xjtu_root), cache_dir),
    }


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
        "message": "未识别为当前支持的数据集结构；可上传 PHM2012/XJTU-SY zip，或上传特征 CSV 做推理。",
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
        "message": "识别为特征 CSV；数值列数量需等于所选模型输入维度。" if valid else "CSV 中没有可用的数值特征列。",
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


def validate_training_run(run_dir: Path | str, expected_task: str | None = None) -> dict[str, Any]:
    path = Path(run_dir)
    required = ["config.json", "metrics.json", "model_summary.json", "model_state.pt", "standardizer.npz"]
    missing = [filename for filename in required if not (path / filename).exists()]
    config = _load_json(path / "config.json")
    metrics = _load_json(path / "metrics.json")
    summary = _load_json(path / "model_summary.json")
    task = str(config.get("task", summary.get("task", "")))
    task_mismatch = bool(expected_task and task and expected_task != task)
    return {
        "path": str(path),
        "valid": not missing and not task_mismatch and config.get("command") == "train",
        "task": task,
        "expected_task": expected_task or "",
        "task_mismatch": task_mismatch,
        "missing": missing,
        "command": config.get("command", ""),
        "model": summary.get("model", ""),
        "input_dim": summary.get("input_dim", ""),
        "sequence_length": summary.get("sequence_length", ""),
        "parameter_count": summary.get("parameter_count", ""),
        "metrics": metrics.get("test", {}),
        "summary": summary,
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


def _phm_command(*args: str | Path) -> list[str]:
    return [sys.executable, "-m", "USTC.SSE.BearingPrediction.cli", *[str(arg) for arg in args]]


def _has_running_job(job: dict[str, Any] | None) -> bool:
    return bool(job and job.get("status") in {"queued", "running"})


def _active_job(st) -> dict[str, Any] | None:
    job_dir = st.session_state.get("active_job_dir")
    if not job_dir:
        return None
    try:
        job = poll_job(job_dir)
    except FileNotFoundError:
        st.session_state.pop("active_job_dir", None)
        return None
    if job.get("status") in {"succeeded", "failed"}:
        st.session_state["last_job_dir"] = job_dir
    return {**job, "job_dir": job_dir}


def _start_job(st, command: list[str], *, kind: str, task: str | None, run_dir: Path | str | None) -> None:
    active = _active_job(st)
    if _has_running_job(active):
        st.warning("已有任务正在运行，请等待结束后再启动新任务。")
        return
    if kind == "train" and run_dir is not None:
        st.session_state["selected_model_run"] = str(run_dir)
    job = start_cli_job(command, kind=kind, task=task, run_dir=run_dir)
    st.session_state["active_job_dir"] = str(job["job_dir"])
    st.session_state["last_job_dir"] = str(job["job_dir"])
    st.rerun()


def _show_metrics(st, metrics: dict[str, Any]) -> None:
    numeric = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
    if not numeric:
        st.info("暂无指标。")
        return
    columns = st.columns(min(len(numeric), 5))
    for column, (name, value) in zip(columns, numeric.items()):
        column.metric(name, _format_float(value))


def _show_figures(st, figures: dict[str, Path | str]) -> None:
    visible = [(name, Path(path)) for name, path in figures.items() if Path(path).exists()]
    if not visible:
        st.info("暂无图表。")
        return
    columns = st.columns(2)
    for index, (name, path) in enumerate(visible):
        columns[index % 2].image(str(path), caption=name, width="stretch")


def _render_run_summary(st, run_dir: Path | str) -> None:
    path = Path(run_dir)
    summary = summarize_run(path)
    st.caption(str(path))
    _show_metrics(st, summary.get("raw_metrics", {}).get("test", {}))
    if summary.get("command") == "benchmark" and summary.get("benchmark_rows"):
        st.dataframe(pd.DataFrame(summary["benchmark_rows"]), width="stretch", hide_index=True)
    _show_figures(st, summary.get("figures", {}))


def _render_artifact_dir(st, artifact_dir: Path | str) -> None:
    path = Path(artifact_dir)
    metrics = _load_json(path / "metrics.json")
    if not metrics:
        st.info("输出目录还没有 metrics.json。")
        return
    st.caption(str(path))
    _show_metrics(st, metrics.get("metrics", metrics.get("test", {})))
    figures = metrics.get("figures", {})
    if figures:
        _show_figures(st, {key: Path(value) for key, value in figures.items()})


def _render_job_panel(st, job: dict[str, Any] | None) -> None:
    st.subheader("任务状态")
    if job is None:
        st.info("当前没有运行中的后台任务。")
        return
    status = str(job.get("status", "unknown"))
    st.write(f"状态：**{status}**")
    st.write(f"类型：**{job.get('kind', '')}**")
    st.write(f"任务：**{job.get('task', '') or '-'}**")
    st.write(f"退出码：**{'-' if job.get('exit_code') is None else job.get('exit_code')}**")
    if job.get("run_dir"):
        st.caption(f"输出目录：{job['run_dir']}")
    st.code(read_job_log(job["job_dir"], tail_bytes=8000) or "等待日志输出。", language="text")
    if status in {"queued", "running"}:
        time.sleep(1.0)
        st.rerun()


def _render_sidebar_summary(st, job: dict[str, Any] | None) -> None:
    st.markdown("### 全局状态")
    status = str(job.get("status", "idle")) if job else "idle"
    kind = str(job.get("kind", "-")) if job else "-"
    task = str(job.get("task", "-") or "-") if job else "-"
    st.metric("后台任务", status)
    st.caption(f"{kind} / {task}")
    if st.button("刷新", width="stretch"):
        st.rerun()
    selected_run = st.session_state.get("selected_model_run")
    if selected_run:
        st.caption(f"模型 run：{Path(selected_run).name}")


def _render_last_output(st, job_dir: str | Path | None) -> None:
    if not job_dir:
        st.info("暂无最近输出。")
        return
    job = poll_job(job_dir)
    run_dir = job.get("run_dir")
    if not run_dir or not Path(run_dir).exists():
        st.info("最近任务还没有可读输出目录。")
        return
    if job.get("status") != "succeeded":
        st.info("最近任务尚未成功结束。")
        return
    st.markdown("**最近输出**")
    if (Path(run_dir) / "config.json").exists():
        _render_run_summary(st, run_dir)
    elif (Path(run_dir) / "metrics.json").exists():
        _render_artifact_dir(st, run_dir)
    else:
        st.caption(str(run_dir))


def _render_main_status_area(st, active: dict[str, Any] | None) -> None:
    st.subheader("运行状态")
    _render_job_panel(st, active)
    st.divider()
    _render_last_output(st, st.session_state.get("last_job_dir"))


def _dataframe_status(st, status: dict[str, dict[str, Any]]) -> None:
    rows = []
    for name, item in status.items():
        rows.append(
            {
                "dataset": name,
                "valid": item["valid"],
                "root": item["root"],
                "bearings": item.get("bearing_count", 0),
                "files": item.get("file_count", 0),
                "conditions/splits": item.get("condition_count", item.get("split_count", 0)),
                "cache": "ready" if item["cache"]["exists"] else "missing",
                "cache_shape": str(item["cache"].get("feature_shape", [])),
                "missing": ", ".join(item.get("missing", [])),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _select_training_run(st, task: str | None = None, *, key_prefix: str = "training_run") -> Path | None:
    runs = list_run_directories(command="train", task=task)
    if not runs:
        st.info("暂无训练 run。")
        return None
    current = st.session_state.get("selected_model_run")
    index = 0
    found_current = False
    if current:
        for idx, item in enumerate(runs):
            if str(item["path"]) == str(current):
                index = idx
                found_current = True
                break
    current_key = Path(current).name if current else "none"
    selected = st.selectbox(
        "训练 run",
        runs,
        index=index,
        format_func=lambda item: f"{item['name']}  {'sample' if item['sample'] else 'full'}",
        key=f"{key_prefix}_select_{task or 'all'}_{current_key}",
    )
    if current and not found_current:
        return Path(current)
    st.session_state["selected_model_run"] = str(selected["path"])
    return Path(selected["path"])


def _render_data_tab(st, active: dict[str, Any] | None) -> None:
    st.subheader("数据加载")
    phm_root = Path(st.text_input("PHM2012 根目录", value=str(st.session_state.get("phm_root", DEFAULT_PHM_ROOT))))
    xjtu_root = Path(st.text_input("XJTU-SY 根目录", value=str(st.session_state.get("xjtu_root", DEFAULT_XJTU_ROOT))))
    st.session_state["phm_root"] = str(phm_root)
    st.session_state["xjtu_root"] = str(xjtu_root)
    status = inspect_dataset_roots(phm_root, xjtu_root)
    _dataframe_status(st, status)
    force = st.checkbox("强制刷新缓存", value=False)
    disabled = _has_running_job(active)
    columns = st.columns(3)
    cache_specs = [("rul", "构建 RUL 特征缓存"), ("fault", "构建 Fault 特征缓存"), ("all", "构建全部特征缓存")]
    for column, (task, label) in zip(columns, cache_specs):
        if column.button(label, disabled=disabled, width="stretch"):
            run_dir = _new_run_dir(GUI_OUTPUT_ROOT / "cache", "cache", task)
            command = _phm_command(
                "cache",
                "--task",
                task,
                "--phm-root",
                phm_root,
                "--xjtu-root",
                xjtu_root,
                "--run-dir",
                run_dir,
                *(["--force"] if force else []),
            )
            _start_job(st, command, kind="cache", task=task, run_dir=run_dir)


def _render_training_tab(st, active: dict[str, Any] | None) -> None:
    st.subheader("训练")
    with st.form("train_form"):
        task = st.selectbox("任务", ["rul", "fault"], format_func=lambda value: "RUL 寿命预测" if value == "rul" else "Fault 故障诊断")
        data_mode = st.radio("数据规模", ["sample", "full"], horizontal=True, format_func=lambda value: "快速样本" if value == "sample" else "全量数据")
        preset = st.selectbox("训练预设", ["smoke", "paper"], index=0 if data_mode == "sample" else 1)
        device = st.selectbox("设备", ["auto", "mps", "cuda", "cpu"])
        submitted = st.form_submit_button("开始训练", disabled=_has_running_job(active), width="stretch")
    if submitted:
        run_dir = _new_run_dir("outputs/runs", "train", task)
        command = _phm_command(
            "train",
            "--task",
            task,
            "--preset",
            preset,
            "--device",
            device,
            "--run-dir",
            run_dir,
            "--sample" if data_mode == "sample" else "--full",
        )
        _start_job(st, command, kind="train", task=task, run_dir=run_dir)
    latest = _default_train_run("rul") or _default_train_run("fault")
    if latest:
        st.markdown("**最近可用训练结果**")
        _render_run_summary(st, latest)


def _render_model_tab(st) -> None:
    st.subheader("模型加载")
    task_filter = st.radio("任务过滤", ["all", "rul", "fault"], horizontal=True, format_func=lambda value: "全部" if value == "all" else value.upper())
    selected = _select_training_run(st, None if task_filter == "all" else task_filter, key_prefix="model_run")
    if selected is None:
        return
    expected = None if task_filter == "all" else task_filter
    validation = validate_training_run(selected, expected_task=expected)
    columns = st.columns(5)
    columns[0].metric("可用", "yes" if validation["valid"] else "no")
    columns[1].metric("任务", validation["task"] or "-")
    columns[2].metric("模型", validation["model"] or "-")
    columns[3].metric("输入维度", str(validation["input_dim"] or "-"))
    columns[4].metric("序列长度", str(validation["sequence_length"] or "-"))
    if validation["missing"]:
        st.error(f"缺少文件：{', '.join(validation['missing'])}")
    if validation["task_mismatch"]:
        st.error(f"任务不匹配：期望 {validation['expected_task']}，实际 {validation['task']}")
    _render_run_summary(st, selected)


def _save_uploaded_csv(uploaded) -> Path:
    upload_dir = GUI_OUTPUT_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded.name}"
    target.write_bytes(uploaded.getbuffer())
    return target


def _render_inference_tab(st, active: dict[str, Any] | None) -> None:
    st.subheader("推理与评测")
    selected = _select_training_run(st, key_prefix="inference_run")
    if selected is None:
        return
    validation = validate_training_run(selected)
    if not validation["valid"]:
        st.error("当前 run 不完整，不能用于评测或推理。")
        return
    device = st.selectbox("推理设备", ["auto", "mps", "cuda", "cpu"], key="infer_device")
    disabled = _has_running_job(active)
    if st.button("运行固定测试集评测", disabled=disabled, width="stretch"):
        output_dir = _new_run_dir(GUI_OUTPUT_ROOT / "evaluations", "evaluate", validation["task"])
        command = _phm_command("evaluate", "--run", selected, "--device", device, "--output-dir", output_dir)
        _start_job(st, command, kind="evaluate", task=validation["task"], run_dir=output_dir)
    st.divider()
    csv_path_text = st.text_input("特征 CSV 路径", value="")
    uploaded = st.file_uploader("或上传特征 CSV", type=["csv"])
    csv_path: Path | None = Path(csv_path_text).expanduser() if csv_path_text else None
    if uploaded is not None:
        csv_path = _save_uploaded_csv(uploaded)
        st.caption(f"已保存上传文件：{csv_path}")
    if csv_path is not None and csv_path.exists():
        inspection = inspect_uploaded_dataset(csv_path)
        st.json(inspection)
        if st.button("运行特征 CSV 推理", disabled=disabled, width="stretch"):
            output_dir = _new_run_dir(GUI_OUTPUT_ROOT / "predictions", "predict", validation["task"])
            command = _phm_command("predict", "--run", selected, "--csv", csv_path, "--device", device, "--output-dir", output_dir)
            _start_job(st, command, kind="predict", task=validation["task"], run_dir=output_dir)
    elif csv_path_text:
        st.error("特征 CSV 路径不存在。")


def _render_benchmark_tab(st, active: dict[str, Any] | None) -> None:
    st.subheader("Benchmark 与运行记录")
    with st.form("benchmark_form"):
        task = st.selectbox("Benchmark 任务", ["all", "rul", "fault"], format_func=lambda value: "全部" if value == "all" else value.upper())
        data_mode = st.radio("Benchmark 数据规模", ["sample", "full"], horizontal=True, format_func=lambda value: "快速样本" if value == "sample" else "全量数据")
        baselines = st.text_input("Baselines", value="linear,forest")
        submitted = st.form_submit_button("运行 Benchmark", disabled=_has_running_job(active), width="stretch")
    if submitted:
        run_dir = _new_run_dir("outputs/runs", "benchmark", task)
        command = _phm_command(
            "benchmark",
            "--task",
            task,
            "--baselines",
            baselines,
            "--run-dir",
            run_dir,
            "--sample" if data_mode == "sample" else "--full",
        )
        _start_job(st, command, kind="benchmark", task=task, run_dir=run_dir)
    benchmark_run = _default_benchmark_run()
    if benchmark_run:
        st.markdown("**最近可用 Benchmark**")
        _render_run_summary(st, benchmark_run)
    jobs = list_jobs(DEFAULT_JOBS_ROOT)
    if jobs:
        st.markdown("**后台任务记录**")
        table = [
            {
                "job_id": item.get("job_id"),
                "kind": item.get("kind"),
                "task": item.get("task"),
                "status": item.get("status"),
                "exit_code": item.get("exit_code"),
                "run_dir": item.get("run_dir"),
                "created_at": item.get("created_at"),
            }
            for item in jobs[:20]
        ]
        st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="轴承 PHM 实验工作台",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1.2rem; }
        section[data-testid="stSidebar"] { min-width: 250px; }
        div[data-testid="stMetric"] { background: #f8fafc; border: 1px solid #d8dee9; padding: 0.65rem; border-radius: 6px; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("轴承 PHM 实验工作台")
    st.caption("数据加载、模型训练、模型加载、推理评测、Benchmark 和运行记录。")

    active = _active_job(st)
    with st.sidebar:
        _render_sidebar_summary(st, active)

    status_area, workspace_area = st.columns([0.8, 1.8], gap="large")
    with status_area:
        _render_main_status_area(st, active)
    with workspace_area:
        data_tab, train_tab, model_tab, inference_tab, benchmark_tab = st.tabs(
            ["数据", "训练", "模型", "推理/评测", "Benchmark/运行记录"]
        )
        with data_tab:
            _render_data_tab(st, active)
        with train_tab:
            _render_training_tab(st, active)
        with model_tab:
            _render_model_tab(st)
        with inference_tab:
            _render_inference_tab(st, active)
        with benchmark_tab:
            _render_benchmark_tab(st, active)


if __name__ == "__main__":
    main()
