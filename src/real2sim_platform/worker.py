from __future__ import annotations

import threading
import time
import traceback
from typing import Any

from pathlib import Path

from .adapters import TaskCanceled, execute_task, index_artifacts_for_task
from .config import PlatformConfig
from .store import PlatformStore


class TaskWorker:
    def __init__(self, store: PlatformStore, config: PlatformConfig) -> None:
        self.store = store
        self.config = config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.store.recover_interrupted_tasks()
        self._thread = threading.Thread(target=self._loop, name="real2sim-platform-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            task = self.store.next_queued_task()
            if task is None:
                time.sleep(0.5)
                continue
            self._run(task)

    def _run(self, task: dict[str, Any]) -> None:
        if self.store.get_task(task["id"])["cancel_requested"]:
            self.store.update_task(task["id"], status="canceled", error="task canceled before start", progress=0, stage="已取消", message="任务启动前已取消")
            return
        self.store.update_task(task["id"], status="running", progress=5, stage="启动中", message="任务已进入执行队列")
        try:
            result = execute_task(self.config, self.store, task)
        except TaskCanceled as exc:
            self._write_trace(task, exc)
            index_artifacts_for_task(self.store, task["id"], Path(task["run_dir"]))
            self.store.update_task(task["id"], status="canceled", error=str(exc), progress=0, stage="已取消", message=str(exc))
            return
        except Exception as exc:
            self._write_trace(task, exc)
            index_artifacts_for_task(self.store, task["id"], Path(task["run_dir"]))
            self.store.update_task(task["id"], status="failed", error=str(exc), stage="执行失败", message=str(exc))
            return
        status = "canceled" if self.store.get_task(task["id"])["cancel_requested"] else "succeeded"
        self.store.update_task(task["id"], status=status, result=result, progress=100 if status == "succeeded" else 0, stage="已完成" if status == "succeeded" else "已取消", message="任务完成" if status == "succeeded" else "任务已取消")

    def _write_trace(self, task: dict[str, Any], exc: Exception) -> None:
        log_path = self.config.runs_dir / task["id"] / "logs" / "task.log"
        if "run_dir" in task:
            log_path = __import__("pathlib").Path(task["run_dir"]) / "logs" / "task.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\nERROR: " + str(exc) + "\n")
            handle.write(traceback.format_exc() + "\n")
