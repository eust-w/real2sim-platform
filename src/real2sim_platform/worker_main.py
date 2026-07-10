from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

from .config import ensure_workspace, load_config
from .store import PlatformStore
from .worker import TaskWorker


def main() -> int:
    parser = argparse.ArgumentParser(description="Real2Sim serial GPU worker")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_workspace(config)
    store = PlatformStore(config.db_path)
    worker = TaskWorker(store, config)
    stopping = False

    def stop(*_args) -> None:
        nonlocal stopping
        stopping = True
        worker.stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    worker.start()
    while not stopping:
        time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
