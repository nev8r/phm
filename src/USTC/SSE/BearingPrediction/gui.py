"""
Streamlit GUI for the bearing PHM workbench.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.gui_jobs import (
    list_jobs,
    poll_job,
    read_job_log,
    start_cli_job,
)
from USTC.SSE.BearingPrediction.training_config import (
    list_training_config_files,
    load_training_config,
    resolve_training_config,
)


DEFAULT_PHM_ROOT = Path("data/loader_roots/phm2012")
DEFAULT_XJTU_ROOT = Path("data/loader_roots/xjtu")
DEFAULT_CACHE_ROOT = Path("cache/paper_features")
GUI_OUTPUT_ROOT = Path("outputs/gui")
RUN_OUTPUT_ROOT = Path("outputs/runs")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _format_number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if np.isnan(number) or np.isinf(number):
        return "-"
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    return f"{number:.{digits}g}"


def _format_bytes(value: Any) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "-"
    if size <= 0:
        return "-"
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return "-"


def _new_run_dir(root: Path | str, command: str, task: str) -> Path:
    output_root = Path(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_root / f"{timestamp}_{command}_{task}"
    index = 1
    while path.exists():
        path = output_root / f"{timestamp}_{command}_{task}_{index}"
        index += 1
    return path


def _phm_command(*args: str | Path) -> list[str]:
    return [sys.executable, "-m", "USTC.SSE.BearingPrediction.cli", *[str(arg) for arg in args]]


def _is_running(job: dict[str, Any] | None) -> bool:
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
    if not _is_running(job):
        st.session_state.pop("active_job_dir", None)
    return {**job, "job_dir": str(job_dir)}


def _start_job(st, command: list[str], *, kind: str, task: str, run_dir: Path | str) -> None:
    if _is_running(_active_job(st)):
        st.warning("已有任务正在运行，请等待任务结束。")
        return
    job = start_cli_job(command, kind=kind, task=task, run_dir=run_dir, cwd=Path.cwd())
    st.session_state["active_job_dir"] = str(job["job_dir"])
    st.session_state["selected_job_dir"] = str(job["job_dir"])
    st.session_state[f"last_{kind}_run_dir"] = str(run_dir)
    st.session_state[f"pending_{kind}_run_dir"] = str(run_dir)
    st.rerun()


def _cache_status(task: str, cache_root: Path = DEFAULT_CACHE_ROOT) -> dict[str, Any]:
    filename = {
        "rul": "phm2012_rul_fft256_full.npz",
        "fault": "xjtu_binary_fault_diagnosis_fft256_full.npz",
    }[task]
    path = cache_root / filename
    result: dict[str, Any] = {
        "task": task.upper(),
        "path": str(path),
        "exists": path.exists(),
        "size": _format_bytes(path.stat().st_size) if path.exists() else "-",
        "features": "-",
        "targets": "-",
        "updated": "-",
    }
    if not path.exists():
        return result
    result["updated"] = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    try:
        with np.load(path, allow_pickle=False) as data:
            if "features" in data.files:
                result["features"] = " x ".join(str(item) for item in data["features"].shape)
            target_key = "targets" if "targets" in data.files else "labels"
            if target_key in data.files:
                result["targets"] = " x ".join(str(item) for item in data[target_key].shape)
    except (OSError, KeyError, ValueError):
        result["features"] = "读取失败"
    return result


def _inspect_roots(phm_root: Path, xjtu_root: Path) -> pd.DataFrame:
    phm_files = list(phm_root.glob("Learning_set/Bearing*/acc_*.csv")) + list(
        phm_root.glob("Full_Test_Set/Bearing*/acc_*.csv")
    )
    xjtu_conditions = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
    xjtu_files: list[Path] = []
    for condition in xjtu_conditions:
        xjtu_files.extend((xjtu_root / condition).glob("Bearing*/*.csv"))
    rows = [
        {
            "数据集": "PHM2012",
            "任务": "RUL",
            "目录存在": phm_root.exists(),
            "文件数": len(phm_files),
            "轴承数": len({item.parent.name for item in phm_files}),
            "缓存": "ready" if _cache_status("rul")["exists"] else "missing",
            "路径": str(phm_root),
        },
        {
            "数据集": "XJTU-SY",
            "任务": "Fault",
            "目录存在": xjtu_root.exists(),
            "文件数": len(xjtu_files),
            "轴承数": len({item.parent.name for item in xjtu_files}),
            "缓存": "ready" if _cache_status("fault")["exists"] else "missing",
            "路径": str(xjtu_root),
        },
    ]
    return pd.DataFrame(rows)


def _list_runs(command: str | None = None, task: str | None = None) -> list[Path]:
    if not RUN_OUTPUT_ROOT.exists():
        return []
    runs: list[Path] = []
    for child in RUN_OUTPUT_ROOT.iterdir():
        if not child.is_dir():
            continue
        config = _load_json(child / "config.json")
        if command and config.get("command") != command:
            continue
        if task and config.get("task") != task:
            continue
        runs.append(child)
    return sorted(runs, key=lambda item: item.stat().st_mtime, reverse=True)


def _is_safe_output_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
        roots = [RUN_OUTPUT_ROOT.resolve(), GUI_OUTPUT_ROOT.resolve()]
        return any(resolved == root or root in resolved.parents for root in roots)
    except OSError:
        return False


def _delete_output_dir(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir() or not _is_safe_output_path(path):
        raise ValueError(f"refuse to delete unsafe path: {path}")
    shutil.rmtree(path)


def _default_training_run(task: str) -> Path | None:
    runs = _list_runs(command="train", task=task)
    return runs[0] if runs else None


def _validate_training_run(run_dir: Path, expected_task: str) -> dict[str, Any]:
    required = ["config.json", "metrics.json", "model_summary.json", "model_state.pt", "standardizer.npz"]
    missing = [name for name in required if not (run_dir / name).exists()]
    config = _load_json(run_dir / "config.json")
    metrics = _load_json(run_dir / "metrics.json")
    summary = _load_json(run_dir / "model_summary.json")
    task = str(config.get("task", summary.get("task", "")))
    return {
        "valid": not missing and config.get("command") == "train" and task == expected_task,
        "missing": missing,
        "task": task,
        "model": summary.get("model", "-"),
        "input_dim": summary.get("input_dim", "-"),
        "sequence_length": summary.get("sequence_length", "-"),
        "parameter_count": summary.get("parameter_count", "-"),
        "metrics": metrics.get("test", {}),
    }


def _metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)):
            rows.append({"指标": key, "值": _format_number(value)})
    return pd.DataFrame(rows)


def _run_display_name(path: Path) -> str:
    config = _load_json(path / "config.json")
    command = config.get("command", "-")
    task = config.get("task", "-")
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
    return f"{path.name} | {command}/{task} | {stamp}"


def _render_run_result(st, run_dir: Path, *, title: str = "结果") -> None:
    if not run_dir.exists():
        st.warning(f"结果目录不存在：{run_dir}")
        return
    config = _load_json(run_dir / "config.json")
    metrics = _load_json(run_dir / "metrics.json")
    st.markdown(f"#### {title}")
    st.caption(f"输出目录：{run_dir}")
    cols = st.columns(4)
    cols[0].metric("命令", str(config.get("command", "-")))
    cols[1].metric("任务", str(config.get("task", "-")).upper())
    cols[2].metric("模式", "sample" if config.get("sample") else "full")
    cols[3].metric("更新时间", datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%H:%M:%S"))

    metric_source = metrics.get("test", metrics.get("metrics", metrics))
    if isinstance(metric_source, dict):
        table = _metrics_table(metric_source)
        if not table.empty:
            st.dataframe(table, width="stretch", hide_index=True)
    _figure_cards(st, run_dir)


def _render_delete_controls(st, path: Path, *, key_prefix: str) -> None:
    clear_col, delete_col = st.columns([1, 2])
    if clear_col.button("清空当前展示", key=f"{key_prefix}_clear", width="stretch"):
        for key in [
            "last_analyze_run_dir",
            "last_train_run_dir",
            "last_evaluate_run_dir",
            "last_predict_run_dir",
        ]:
            if st.session_state.get(key) == str(path):
                st.session_state.pop(key, None)
        st.rerun()
    confirm = delete_col.checkbox("确认删除这个结果目录", key=f"{key_prefix}_confirm")
    if delete_col.button("删除结果目录", key=f"{key_prefix}_delete", disabled=not confirm, width="stretch"):
        try:
            _delete_output_dir(path)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"已删除：{path}")
            for key, value in list(st.session_state.items()):
                if isinstance(value, str) and value == str(path):
                    st.session_state.pop(key, None)
            st.rerun()


def _save_upload(uploaded, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded.name}"
    target.write_bytes(uploaded.getbuffer())
    return target


def _inspect_csv(path: Path) -> dict[str, Any]:
    try:
        frame = pd.read_csv(path, nrows=50)
    except Exception as exc:
        return {"ok": False, "message": f"CSV 读取失败：{exc}"}
    numeric = frame.select_dtypes(include=[np.number])
    return {
        "ok": len(frame) > 0 and numeric.shape[1] > 0,
        "rows": len(frame),
        "columns": frame.shape[1],
        "numeric_columns": numeric.shape[1],
    }


def _figure_cards(st, run_dir: Path) -> None:
    figure_dir = run_dir / "figures"
    if not figure_dir.exists():
        return
    figures = sorted([*figure_dir.glob("*.png"), *figure_dir.glob("*.jpg"), *figure_dir.glob("*.jpeg")])
    if not figures:
        return
    st.markdown("#### 图表")
    columns = st.columns(2)
    for index, figure in enumerate(figures[:8]):
        with columns[index % 2]:
            st.image(str(figure), caption=figure.name, width="stretch")


def _render_analysis_source(st, run_dir: Path) -> None:
    config = _load_json(run_dir / "config.json")
    metrics = _load_json(run_dir / "metrics.json")
    rows = []
    fallback_tasks = []
    for task, item in metrics.get("analyses", {}).items():
        source = item.get("source", {}) if isinstance(item, dict) else {}
        source_name = str(source.get("source", "-"))
        if source_name == "sample-fallback":
            fallback_tasks.append(task.upper())
        rows.append(
            {
                "任务": task.upper(),
                "来源": source_name,
                "模式": source.get("mode", "-"),
                "说明": source.get("warning", ""),
            }
        )
    st.caption(
        f"当前显示：{run_dir} | 任务：{config.get('task', '-')} | "
        f"特征集：{config.get('feature_set', '-')} | "
        f"规模：{'快速样本' if config.get('sample') else '全量'}"
    )
    if fallback_tasks:
        st.warning(
            f"{', '.join(fallback_tasks)} 没找到全量特征缓存，已回退到固定样本数据；"
            "重复生成会得到相同图谱。先构建对应缓存或换真实数据目录后，图谱才会变化。"
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_job_panel(st, job: dict[str, Any] | None) -> None:
    if not job:
        st.info("暂无任务。")
        return
    status = str(job.get("status", "unknown"))
    st.markdown(f"**任务** `{job.get('kind', '-')}` / `{job.get('task', '-')}` / `{status}`")
    st.caption(f"输出目录：{job.get('run_dir') or '-'}")
    if status == "running":
        st.progress(0.65, text="任务运行中，日志会自动追加。")
    elif status == "queued":
        st.progress(0.15, text="任务排队中。")
    elif status == "succeeded":
        st.success("任务完成。")
    elif status == "failed":
        st.error(f"任务失败，退出码：{job.get('exit_code')}")
    log_text = read_job_log(job["job_dir"], tail_bytes=16000) if job.get("job_dir") else ""
    st.code(log_text or "等待日志输出。", language="text")
    if _is_running(job):
        refresh_key = f"refresh_job_{str(job.get('job_dir', job.get('job_id', 'unknown'))).replace('/', '_')}"
        st.button("刷新任务状态", key=refresh_key, on_click=st.rerun)


def _overview_tab(st, active: dict[str, Any] | None) -> None:
    st.subheader("运行总览")
    cache_rows = [_cache_status("rul"), _cache_status("fault")]
    col1, col2, col3 = st.columns(3)
    col1.metric("RUL 缓存", "ready" if cache_rows[0]["exists"] else "missing")
    col2.metric("Fault 缓存", "ready" if cache_rows[1]["exists"] else "missing")
    col3.metric("后台任务", active["status"] if active else "idle")
    st.dataframe(pd.DataFrame(cache_rows), width="stretch", hide_index=True)

    latest_runs = _list_runs()[:6]
    if latest_runs:
        st.markdown("#### 最近输出")
        rows = []
        for run in latest_runs:
            config = _load_json(run / "config.json")
            rows.append(
                {
                    "名称": run.name,
                    "命令": config.get("command", "-"),
                    "任务": config.get("task", "-"),
                    "更新时间": datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "路径": str(run),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _data_tab(st, active: dict[str, Any] | None) -> None:
    st.subheader("数据与特征")
    left, right = st.columns([0.8, 1.2], gap="large")
    with left:
        phm_root = Path(st.text_input("PHM2012 根目录", value=str(st.session_state.get("phm_root", DEFAULT_PHM_ROOT))))
        xjtu_root = Path(st.text_input("XJTU-SY 根目录", value=str(st.session_state.get("xjtu_root", DEFAULT_XJTU_ROOT))))
        st.session_state["phm_root"] = str(phm_root)
        st.session_state["xjtu_root"] = str(xjtu_root)
        force = st.checkbox("强制重建缓存", value=False)
        disabled = _is_running(active)
        cache_cols = st.columns(3)
        for column, task, label in [
            (cache_cols[0], "rul", "构建 RUL"),
            (cache_cols[1], "fault", "构建 Fault"),
            (cache_cols[2], "all", "全部缓存"),
        ]:
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
    with right:
        st.dataframe(_inspect_roots(phm_root, xjtu_root), width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### 特征分析")
    task_col, feature_col, scale_col, run_col = st.columns([1, 1, 1, 1])
    task = task_col.selectbox("任务", ["all", "rul", "fault"], format_func=lambda value: value.upper())
    feature_set = feature_col.selectbox("特征集", ["domain", "tsfresh", "advanced"])
    scale = scale_col.radio(
        "规模",
        ["sample", "full"],
        horizontal=True,
        format_func=lambda value: "快速样本" if value == "sample" else "全量缓存",
    )
    if run_col.button("生成图谱", disabled=_is_running(active), width="stretch"):
        run_dir = _new_run_dir(RUN_OUTPUT_ROOT, "analyze", task)
        command = _phm_command(
            "analyze",
            "--task",
            task,
            "--feature-set",
            feature_set,
            "--run-dir",
            run_dir,
            "--sample" if scale == "sample" else "--full",
        )
        _start_job(st, command, kind="analyze", task=task, run_dir=run_dir)

    analyze_runs = _list_runs(command="analyze")
    pending_analyze = Path(st.session_state.get("pending_analyze_run_dir", ""))
    if active and active.get("kind") == "analyze" and _is_running(active):
        return
    if pending_analyze in analyze_runs:
        st.session_state.pop("pending_analyze_run_dir", None)
    elif st.session_state.get("pending_analyze_run_dir") and not (active and active.get("kind") == "analyze"):
        return
    if analyze_runs:
        last_analyze = Path(st.session_state.get("last_analyze_run_dir", ""))
        default_index = analyze_runs.index(last_analyze) if last_analyze in analyze_runs else 0
        selected = st.selectbox("查看分析结果", analyze_runs, index=default_index, format_func=_run_display_name)
        _render_analysis_source(st, selected)
        _figure_cards(st, selected)
        _render_delete_controls(st, selected, key_prefix="analyze_result")


def _training_tab(st, active: dict[str, Any] | None) -> None:
    st.subheader("训练")
    config_col, preview_col = st.columns([0.75, 1.25], gap="large")
    request: dict[str, Any] | None = None
    selected_path: Path | None = None
    with config_col:
        configs = list_training_config_files()
        labels = ["请选择配置"] + [str(path) for path in configs]
        default = next((index for index, label in enumerate(labels) if label.endswith("rul_smoke.yaml")), 0)
        selected_label = st.selectbox("训练配置", labels, index=default)
        custom_path = st.text_input("自定义 YAML 路径", value="")
        uploaded = st.file_uploader("上传 YAML", type=["yaml", "yml"])
        if uploaded is not None:
            selected_path = _save_upload(uploaded, GUI_OUTPUT_ROOT / "configs")
            st.caption(f"已保存：{selected_path}")
        elif custom_path:
            selected_path = Path(custom_path).expanduser()
        elif selected_label != "请选择配置":
            selected_path = Path(selected_label)
        error = ""
        if selected_path is not None:
            try:
                request = resolve_training_config(load_training_config(selected_path), config_path=selected_path)
            except Exception as exc:
                error = str(exc)
        if error:
            st.error(error)
        if st.button("开始训练", disabled=_is_running(active) or request is None or bool(error), width="stretch"):
            assert request is not None and selected_path is not None
            run_dir = _new_run_dir(RUN_OUTPUT_ROOT, "train", request["task"])
            _start_job(st, _phm_command("train", "--config", selected_path, "--run-dir", run_dir), kind="train", task=request["task"], run_dir=run_dir)
    with preview_col:
        if request is None:
            st.info("选择训练配置后会显示训练参数。")
            return
        st.markdown("#### 配置预览")
        metrics = st.columns(4)
        metrics[0].metric("任务", request["task"].upper())
        metrics[1].metric("预设", request["preset"])
        metrics[2].metric("数据", "sample" if request["sample"] else "full")
        metrics[3].metric("设备", request["device"])
        st.json(
            {
                "dataset": request["dataset_config"],
                "trainer": request["trainer_config"],
                "training": request["training_overrides"],
                "model": request["model_config"],
                "split": request["split_preview"],
            },
            expanded=False,
        )
    st.divider()
    train_runs = _list_runs(command="train")
    if train_runs:
        last_train = Path(st.session_state.get("last_train_run_dir", ""))
        default_index = train_runs.index(last_train) if last_train in train_runs else 0
        selected = st.selectbox("查看训练结果", train_runs, index=default_index, format_func=_run_display_name)
        _render_run_result(st, selected, title="训练结果")
        _render_delete_controls(st, selected, key_prefix="train_result")


def _evaluation_tab(st, active: dict[str, Any] | None) -> None:
    st.subheader("评测与推理")
    control_col, result_col = st.columns([0.8, 1.2], gap="large")
    with control_col:
        task = st.selectbox("任务", ["rul", "fault"], format_func=lambda value: "RUL 寿命预测" if value == "rul" else "Fault 故障诊断")
        default_run = _default_training_run(task)
        run_text = st.text_input("训练 run 目录", value=str(default_run or ""))
        run_path = Path(run_text).expanduser() if run_text else None
        mode = st.radio("模式", ["fixed_test", "feature_csv"], horizontal=True, format_func=lambda value: "固定测试集" if value == "fixed_test" else "特征 CSV")
        device = st.selectbox("设备", ["auto", "mps", "cuda", "cpu"])
        validation = _validate_training_run(run_path, task) if run_path and run_path.exists() else None
        csv_path: Path | None = None
        if mode == "feature_csv":
            csv_text = st.text_input("CSV 路径", value="")
            uploaded = st.file_uploader("上传 CSV", type=["csv"])
            if uploaded is not None:
                csv_path = _save_upload(uploaded, GUI_OUTPUT_ROOT / "uploads")
            elif csv_text:
                csv_path = Path(csv_text).expanduser()
            if csv_path and csv_path.exists():
                st.json(_inspect_csv(csv_path), expanded=False)
        disabled = _is_running(active) or validation is None or not validation["valid"]
        if mode == "fixed_test":
            if st.button("开始评测", disabled=disabled, width="stretch"):
                assert run_path is not None
                output_dir = _new_run_dir(GUI_OUTPUT_ROOT / "evaluations", "evaluate", task)
                _start_job(st, _phm_command("evaluate", "--run", run_path, "--device", device, "--output-dir", output_dir), kind="evaluate", task=task, run_dir=output_dir)
        else:
            if st.button("开始推理", disabled=disabled or csv_path is None or not csv_path.exists(), width="stretch"):
                assert run_path is not None and csv_path is not None
                output_dir = _new_run_dir(GUI_OUTPUT_ROOT / "predictions", "predict", task)
                _start_job(st, _phm_command("predict", "--run", run_path, "--csv", csv_path, "--device", device, "--output-dir", output_dir), kind="predict", task=task, run_dir=output_dir)
    with result_col:
        if run_path and not run_path.exists():
            st.error("训练 run 目录不存在。")
        elif validation is None:
            st.info("选择训练 run 后可评测或推理。")
        else:
            st.markdown("#### 模型状态")
            cols = st.columns(4)
            cols[0].metric("可用", "yes" if validation["valid"] else "no")
            cols[1].metric("模型", validation["model"])
            cols[2].metric("输入维度", validation["input_dim"])
            cols[3].metric("序列长度", validation["sequence_length"])
            if validation["missing"]:
                st.error(f"缺少文件：{', '.join(validation['missing'])}")
            if validation["task"] != task:
                st.error(f"任务不匹配：当前选择 {task}，run 是 {validation['task'] or '-'}")
            table = _metrics_table(validation["metrics"])
            if not table.empty:
                st.dataframe(table, width="stretch", hide_index=True)
    st.divider()
    result_options = [
        Path(value)
        for value in [
            st.session_state.get("last_evaluate_run_dir"),
            st.session_state.get("last_predict_run_dir"),
        ]
        if value and Path(value).exists()
    ]
    result_options.extend(path for path in sorted((GUI_OUTPUT_ROOT / "evaluations").glob("*"), key=lambda item: item.stat().st_mtime, reverse=True) if path.is_dir())
    result_options.extend(path for path in sorted((GUI_OUTPUT_ROOT / "predictions").glob("*"), key=lambda item: item.stat().st_mtime, reverse=True) if path.is_dir())
    deduped_results = []
    seen_results = set()
    for path in result_options:
        if path in seen_results:
            continue
        seen_results.add(path)
        deduped_results.append(path)
    if deduped_results:
        selected_result = st.selectbox("查看评测/推理结果", deduped_results, format_func=lambda path: f"{path.name} | {datetime.fromtimestamp(path.stat().st_mtime).strftime('%H:%M:%S')}")
        _render_run_result(st, selected_result, title="评测/推理结果")
        _render_delete_controls(st, selected_result, key_prefix="eval_result")


def _jobs_tab(st, active: dict[str, Any] | None) -> None:
    st.subheader("任务日志")
    jobs = list_jobs()
    if not jobs:
        st.info("暂无后台任务。")
        return
    rows = []
    for job in jobs:
        rows.append(
            {
                "任务": job.get("job_id", ""),
                "类型": job.get("kind", ""),
                "任务类型": job.get("task", ""),
                "状态": job.get("status", ""),
                "退出码": job.get("exit_code", ""),
                "开始": job.get("started_at", "") or job.get("created_at", ""),
                "输出": job.get("run_dir", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    selected_default = active["job_dir"] if active else st.session_state.get("selected_job_dir", jobs[0].get("job_dir"))
    options = [str(job["job_dir"]) for job in jobs]
    selected = st.selectbox("查看日志", options, index=options.index(selected_default) if selected_default in options else 0)
    try:
        job = {**poll_job(selected), "job_dir": selected}
    except FileNotFoundError:
        job = None
    _render_job_panel(st, job)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="轴承 PHM 工作台", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1.5rem; max-width: none; }
        header[data-testid="stHeader"], #MainMenu, footer { visibility: hidden; }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.7rem;
        }
        .phm-title {
            padding: 0.75rem 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            margin-bottom: 0.8rem;
        }
        .phm-title h1 { margin: 0; font-size: 1.55rem; }
        .phm-title p { margin: 0.25rem 0 0; color: #667085; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    active = _active_job(st)
    st.markdown(
        """
        <div class="phm-title">
          <h1>轴承 PHM 工作台</h1>
          <p>数据检查、特征缓存、训练、评测和推理统一入口。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("### 状态")
        if active:
            st.info(f"{active.get('kind')} / {active.get('task')} / {active.get('status')}")
            if st.button("刷新", width="stretch"):
                st.rerun()
        else:
            st.success("空闲")
        st.caption(f"工作目录：{Path.cwd()}")

    tabs = st.tabs(["总览", "数据", "训练", "评测/推理", "任务日志"])
    with tabs[0]:
        _overview_tab(st, active)
    with tabs[1]:
        _data_tab(st, active)
    with tabs[2]:
        _training_tab(st, active)
    with tabs[3]:
        _evaluation_tab(st, active)
    with tabs[4]:
        _jobs_tab(st, active)


if __name__ == "__main__":
    main()
