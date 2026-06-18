"""
Record the classroom Streamlit GUI demo

this file is for driving the local GUI with Playwright and saving a demo video

created by zy

copyright USTC

2026
"""

from __future__ import annotations

import argparse
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


def _click_and_wait(page, role_name: str, done_text: str, timeout_ms: int = 180_000) -> None:
    existing_count = page.locator("body").inner_text(timeout=30_000).count(done_text)
    page.get_by_role("button", name=role_name).click(timeout=30_000)
    page.wait_for_function(
        "([text, count]) => document.body.innerText.split(text).length - 1 > count",
        arg=[done_text, existing_count],
        timeout=timeout_ms,
    )
    page.wait_for_timeout(1_000)


def _record_browser(url: str, video_dir: Path, minimum_seconds: int) -> Path:
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
        page.get_by_text("系统概览").first.wait_for(timeout=60_000)
        page.wait_for_timeout(2_000)
        _click_and_wait(page, "加载 RUL 模型复推理", "复推理完成")
        _click_and_wait(page, "运行 RUL 训练 Demo", "训练 demo 完成")
        _click_and_wait(page, "运行 Benchmark Demo", "benchmark demo 完成")
        _click_and_wait(page, "加载 Fault 模型复推理", "复推理完成")
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
    target = video_dir / f"{timestamp}_phm_gui_demo.webm"
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
    parser = argparse.ArgumentParser(description="Record the local PHM Streamlit GUI demo.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--output-dir", default="outputs/gui_demo")
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
        webm_path = _record_browser(url, output_dir, args.minimum_seconds)
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
