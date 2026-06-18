"""
Record the Streamlit GUI workbench smoke flow

this file is for driving the local GUI with Playwright and saving a workbench video

created by zy

copyright USTC

2026
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def _server_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= int(response.status) < 500
    except (OSError, urllib.error.URLError):
        return False


def _wait_for_server(url: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _server_ready(url):
            return
        time.sleep(1)
    raise TimeoutError(f"GUI server did not become ready: {url}")


def _start_gui(project_root: Path, host: str, port: int) -> subprocess.Popen[str] | None:
    url = f"http://{host}:{port}"
    if _server_ready(url):
        return None
    gui_path = project_root / "src" / "USTC" / "SSE" / "BearingPrediction" / "gui.py"
    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(gui_path),
            "--server.port",
            str(port),
            "--server.address",
            host,
            "--server.headless",
            "true",
        ],
        cwd=project_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def _install_chromium_if_needed() -> None:
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)


def _latest_job(project_root: Path, kind: str, since: float) -> dict | None:
    jobs_root = project_root / "outputs" / "gui" / "jobs"
    if not jobs_root.exists():
        return None
    candidates = []
    for job_path in jobs_root.glob("*/job.json"):
        if job_path.stat().st_mtime < since - 2:
            continue
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if job.get("kind") == kind:
            candidates.append((job_path.stat().st_mtime, job))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]


def _click_and_wait_job(page, project_root: Path, role_name: str, kind: str, timeout_seconds: int = 180) -> dict:
    started = time.time()
    page.get_by_role("button", name=role_name).click(timeout=30_000)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = _latest_job(project_root, kind, started)
        if job and job.get("status") in {"succeeded", "failed"}:
            if job.get("status") != "succeeded":
                raise RuntimeError(f"{kind} job failed: {job}")
            page.wait_for_timeout(1_000)
            return job
        page.wait_for_timeout(500)
    raise TimeoutError(f"{kind} job did not finish within {timeout_seconds} seconds")


def _assert_evaluation_uses_training_run(project_root: Path, train_job: dict, evaluate_job: dict) -> None:
    metrics_path = project_root / str(evaluate_job["run_dir"]) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = str(train_job["run_dir"])
    actual = str(metrics.get("source_run", ""))
    if actual != expected:
        raise AssertionError(f"evaluation source run mismatch: {actual} != {expected}")


def _record_browser(url: str, project_root: Path, video_dir: Path, minimum_seconds: int) -> Path:
    from playwright.sync_api import Error, sync_playwright

    video_tmp = video_dir / "tmp"
    video_tmp.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Error as exc:
            if "Executable doesn't exist" not in str(exc):
                raise
            _install_chromium_if_needed()
            browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 980},
            record_video_dir=str(video_tmp),
            record_video_size={"width": 1440, "height": 980},
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=120_000)
        page.get_by_text("轴承 PHM 实验工作台").first.wait_for(timeout=60_000)
        page.wait_for_timeout(2_000)
        page.get_by_role("tab", name="数据").click()
        page.get_by_text("PHM2012 根目录").first.wait_for(timeout=30_000)
        page.get_by_role("tab", name="训练").click()
        page.get_by_text("训练").first.wait_for(timeout=30_000)
        train_job = _click_and_wait_job(page, project_root, "开始训练", "train")
        page.get_by_role("tab", name="推理/评测").click()
        page.get_by_text("推理与评测").first.wait_for(timeout=30_000)
        evaluate_job = _click_and_wait_job(page, project_root, "运行固定测试集评测", "evaluate")
        _assert_evaluation_uses_training_run(project_root, train_job, evaluate_job)
        page.get_by_role("tab", name="Benchmark/运行记录").click()
        page.get_by_text("Benchmark 与运行记录").first.wait_for(timeout=30_000)
        _click_and_wait_job(page, project_root, "运行 Benchmark", "benchmark")
        remaining = minimum_seconds - int(time.monotonic() - started)
        if remaining > 0:
            page.wait_for_timeout(remaining * 1000)
        video = page.video
        context.close()
        browser.close()
        if video is None:
            raise RuntimeError("Playwright did not produce a video artifact")
        source = Path(video.path())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = video_dir / f"{timestamp}_phm_workbench_smoke.webm"
    source.replace(target)
    return target


def _convert_to_mp4(webm_path: Path) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    mp4_path = webm_path.with_suffix(".mp4")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(webm_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return mp4_path if mp4_path.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record the local PHM Streamlit GUI workbench smoke flow.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--output-dir", default="outputs/gui/recordings")
    parser.add_argument("--minimum-seconds", type=int, default=35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"http://{args.host}:{args.port}"
    process = _start_gui(project_root, args.host, args.port)
    try:
        _wait_for_server(url, timeout_seconds=90)
        webm_path = _record_browser(url, project_root, output_dir, args.minimum_seconds)
        mp4_path = _convert_to_mp4(webm_path)
        print(f"webm={webm_path}")
        if mp4_path is not None:
            print(f"mp4={mp4_path}")
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
