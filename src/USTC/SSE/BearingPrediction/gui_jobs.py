"""
GUI background job management module

this file is for running CLI jobs behind the Streamlit workbench

created by zy

copyright USTC

2026
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_JOBS_ROOT = Path("outputs/gui/jobs")
_RUNNER_PROCESSES: list[subprocess.Popen] = []


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_job(job_dir: Path | str, payload: dict[str, Any]) -> None:
    path = Path(job_dir) / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def read_job(job_dir: Path | str) -> dict[str, Any]:
    path = Path(job_dir) / "job.json"
    if not path.exists():
        raise FileNotFoundError(f"job metadata does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_job_log(job_dir: Path | str, *, tail_bytes: int | None = None) -> str:
    path = Path(job_dir) / "stdout.log"
    if not path.exists():
        return ""
    if tail_bytes is None:
        return path.read_text(encoding="utf-8", errors="replace")
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - tail_bytes))
        return handle.read().decode("utf-8", errors="replace")


def list_jobs(jobs_root: Path | str = DEFAULT_JOBS_ROOT) -> list[dict[str, Any]]:
    root = Path(jobs_root)
    if not root.exists():
        return []
    jobs = []
    for child in root.iterdir():
        if not child.is_dir() or not (child / "job.json").exists():
            continue
        try:
            job = read_job(child)
        except (OSError, json.JSONDecodeError):
            continue
        job["job_dir"] = str(child)
        jobs.append(job)
    return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)


def start_cli_job(
    command: list[str],
    *,
    kind: str,
    jobs_root: Path | str = DEFAULT_JOBS_ROOT,
    task: str | None = None,
    run_dir: Path | str | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    jobs_root = Path(jobs_root)
    job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{kind}_{uuid4().hex[:8]}"
    job_dir = jobs_root / job_id
    job = {
        "job_id": job_id,
        "kind": kind,
        "task": task,
        "status": "queued",
        "command": command,
        "run_dir": str(run_dir) if run_dir is not None else "",
        "created_at": _now(),
        "started_at": "",
        "ended_at": "",
        "exit_code": None,
        "pid": None,
    }
    write_job(job_dir, job)
    (job_dir / "stdout.log").touch()
    runner = [
        sys.executable,
        "-m",
        "USTC.SSE.BearingPrediction.gui_jobs",
        "--job-dir",
        str(job_dir),
        "--",
        *command,
    ]
    _reap_runner_processes()
    process = subprocess.Popen(
        runner,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _RUNNER_PROCESSES.append(process)
    job["pid"] = process.pid
    write_job(job_dir, job)
    return {**job, "job_dir": job_dir}


def poll_job(job_dir: Path | str) -> dict[str, Any]:
    return read_job(job_dir)


def wait_for_job(job_dir: Path | str, *, timeout_seconds: float, poll_seconds: float = 0.5) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = poll_job(job_dir)
        if job.get("status") in {"succeeded", "failed"}:
            _reap_runner_processes()
            return job
        time.sleep(poll_seconds)
    raise TimeoutError(f"job did not finish within {timeout_seconds} seconds: {job_dir}")


def _reap_runner_processes() -> None:
    active = []
    for process in _RUNNER_PROCESSES:
        process.poll()
        if process.returncode is None:
            active.append(process)
    _RUNNER_PROCESSES[:] = active


def _run_job(job_dir: Path, command: list[str]) -> int:
    job = read_job(job_dir)
    job.update({"status": "running", "started_at": _now()})
    write_job(job_dir, job)
    start = time.perf_counter()
    with (job_dir / "stdout.log").open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    ended = _now()
    job = read_job(job_dir)
    job.update(
        {
            "status": "succeeded" if completed.returncode == 0 else "failed",
            "ended_at": ended,
            "exit_code": int(completed.returncode),
            "duration_seconds": round(time.perf_counter() - start, 3),
        }
    )
    write_job(job_dir, job)
    return int(completed.returncode)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PHM GUI background job.")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command after --")
    raise SystemExit(_run_job(Path(args.job_dir), command))


if __name__ == "__main__":
    main()
