from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class PlatformStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def create_dataset(
        self,
        *,
        name: str,
        path: Path,
        scan: dict[str, Any],
        item_id: str | None = None,
        source_type: str = "local",
        source_uri: str = "",
        version: str = "",
        parent_task_id: str | None = None,
        manifest_path: Path | None = None,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        item_id = item_id or uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into datasets(id,name,path,scan_json,source_type,source_uri,version,parent_task_id,manifest_path,metrics_json,tags_json,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item_id, name, str(path), _json(scan), source_type, source_uri, version, parent_task_id, str(manifest_path or ""), _json(metrics or {}), _json(tags or []), now, now),
            )
        return self.get_dataset(item_id)

    def list_datasets(self) -> list[dict[str, Any]]:
        return self._list("datasets", self._dataset_row)

    def get_dataset(self, item_id: str) -> dict[str, Any]:
        return self._get("datasets", item_id, self._dataset_row)

    def update_dataset(self, item_id: str, *, name: str | None = None, tags: list[str] | None = None, version: str | None = None) -> dict[str, Any]:
        self._update_entity("datasets", item_id, name=name, tags=tags, version=version)
        return self.get_dataset(item_id)

    def delete_dataset(self, item_id: str) -> None:
        self._delete("datasets", item_id)

    def create_scene(
        self,
        *,
        name: str,
        path: Path,
        metadata: dict[str, Any],
        item_id: str | None = None,
        source_type: str = "generated",
        source_uri: str = "",
        version: str = "",
        parent_task_id: str | None = None,
        manifest_path: Path | None = None,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        item_id = item_id or uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into scenes(id,name,path,metadata_json,source_type,source_uri,version,parent_task_id,manifest_path,metrics_json,tags_json,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item_id, name, str(path), _json(metadata), source_type, source_uri, version, parent_task_id, str(manifest_path or ""), _json(metrics or {}), _json(tags or []), now, now),
            )
        return self.get_scene(item_id)

    def list_scenes(self) -> list[dict[str, Any]]:
        return self._list("scenes", self._scene_row)

    def get_scene(self, item_id: str) -> dict[str, Any]:
        return self._get("scenes", item_id, self._scene_row)

    def update_scene(
        self,
        item_id: str,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._update_entity("scenes", item_id, name=name, tags=tags, version=version)
        if metadata is not None or metrics is not None:
            assignments: list[str] = []
            values: list[Any] = []
            if metadata is not None:
                assignments.append("metadata_json=?")
                values.append(_json(metadata))
            if metrics is not None:
                assignments.append("metrics_json=?")
                values.append(_json(metrics))
            assignments.append("updated_at=?")
            values.append(_now())
            values.append(item_id)
            with self._connect() as db:
                db.execute(f"update scenes set {','.join(assignments)} where id=?", values)
        return self.get_scene(item_id)

    def delete_scene(self, item_id: str) -> None:
        self._delete("scenes", item_id)

    def create_model(
        self,
        *,
        name: str,
        path: Path,
        metadata: dict[str, Any],
        item_id: str | None = None,
        source_type: str = "training",
        source_uri: str = "",
        version: str = "",
        parent_task_id: str | None = None,
        manifest_path: Path | None = None,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        item_id = item_id or uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into models(id,name,path,metadata_json,source_type,source_uri,version,parent_task_id,manifest_path,metrics_json,tags_json,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item_id, name, str(path), _json(metadata), source_type, source_uri, version, parent_task_id, str(manifest_path or ""), _json(metrics or {}), _json(tags or []), now, now),
            )
        return self.get_model(item_id)

    def list_models(self) -> list[dict[str, Any]]:
        return self._list("models", self._model_row)

    def get_model(self, item_id: str) -> dict[str, Any]:
        return self._get("models", item_id, self._model_row)

    def update_model(self, item_id: str, *, name: str | None = None, tags: list[str] | None = None, version: str | None = None) -> dict[str, Any]:
        self._update_entity("models", item_id, name=name, tags=tags, version=version)
        return self.get_model(item_id)

    def delete_model(self, item_id: str) -> None:
        self._delete("models", item_id)

    def create_task(self, *, task_type: str, params: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        item_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into tasks(id,type,status,params_json,result_json,run_dir,created_at,updated_at) values(?,?,?,?,?,?,?,?)",
                (item_id, task_type, "queued", _json(params), "{}", str(run_dir), now, now),
            )
        return self.get_task(item_id)

    def list_tasks(self) -> list[dict[str, Any]]:
        return self._list("tasks", self._task_row)

    def get_task(self, item_id: str) -> dict[str, Any]:
        return self._get("tasks", item_id, self._task_row)

    def next_queued_task(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("select * from tasks where status='queued' order by created_at limit 1").fetchone()
        return self._task_row(row) if row else None

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        progress: int | None = None,
        stage: str | None = None,
        message: str | None = None,
    ) -> None:
        task = self.get_task(task_id)
        next_progress = task["progress"] if progress is None else max(0, min(100, int(progress)))
        with self._connect() as db:
            db.execute(
                "update tasks set status=?, result_json=?, error=?, progress=?, stage=?, message=?, updated_at=? where id=?",
                (
                    status or task["status"],
                    _json(result if result is not None else task["result"]),
                    error,
                    next_progress,
                    task["stage"] if stage is None else stage,
                    task["message"] if message is None else message,
                    _now(),
                    task_id,
                ),
            )

    def set_task_pid(self, task_id: str, pid: int | None) -> None:
        with self._connect() as db:
            db.execute("update tasks set pid=?, updated_at=? where id=?", (pid, _now(), task_id))

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        task = self.get_task(task_id)
        with self._connect() as db:
            if task["status"] == "queued":
                db.execute(
                    "update tasks set status='canceled', cancel_requested=1, progress=0, stage='已取消', message='任务启动前已取消', updated_at=? where id=?",
                    (_now(), task_id),
                )
            elif task["status"] == "running":
                db.execute(
                    "update tasks set status='canceling', cancel_requested=1, stage='取消中', message='正在终止运行进程', updated_at=? where id=?",
                    (_now(), task_id),
                )
        return self.get_task(task_id)

    def recover_interrupted_tasks(self) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "update tasks set status='failed_recovered', error='service restarted while task was running', pid=null, stage='恢复失败', message='服务重启时任务仍在运行，已标记为可重试', updated_at=? where status in ('running','canceling')",
                (_now(),),
            )
            return cursor.rowcount

    def retry_task(self, task_id: str) -> dict[str, Any]:
        task = self.get_task(task_id)
        base = Path(task["run_dir"])
        index = 1
        retry_dir = base.with_name(f"{base.name}_retry{index}")
        while retry_dir.exists():
            index += 1
            retry_dir = base.with_name(f"{base.name}_retry{index}")
        return self.create_task(task_type=task["type"], params=task["params"], run_dir=retry_dir)

    def update_task_params(self, task_id: str, *, params: dict[str, Any]) -> dict[str, Any]:
        task = self.get_task(task_id)
        if task["status"] not in {"queued", "failed", "failed_recovered", "canceled"}:
            raise ValueError("only queued or terminal tasks can be edited")
        with self._connect() as db:
            db.execute("update tasks set params_json=?, updated_at=? where id=?", (_json(params), _now(), task_id))
        return self.get_task(task_id)

    def delete_task(self, task_id: str) -> None:
        task = self.get_task(task_id)
        if task["status"] in {"running", "canceling"}:
            raise ValueError("running tasks cannot be deleted")
        self._delete("tasks", task_id)

    def add_artifact(self, *, task_id: str | None, name: str, path: Path, kind: str) -> dict[str, Any]:
        item_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into artifacts(id,task_id,name,path,kind,created_at) values(?,?,?,?,?,?)",
                (item_id, task_id, name, str(path), kind, now),
            )
        return self.get_artifact(item_id)

    def list_task_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("select * from artifacts where task_id=? order by created_at", (task_id,)).fetchall()
        return [self._artifact_row(row) for row in rows]

    def get_artifact(self, item_id: str) -> dict[str, Any]:
        return self._get("artifacts", item_id, self._artifact_row)

    def _init(self) -> None:
        with self._connect() as db:
            db.execute("create table if not exists datasets(id text primary key,name text,path text,scan_json text,created_at real,updated_at real)")
            db.execute("create table if not exists scenes(id text primary key,name text,path text,metadata_json text,created_at real,updated_at real)")
            db.execute("create table if not exists models(id text primary key,name text,path text,metadata_json text,created_at real,updated_at real)")
            db.execute(
                "create table if not exists tasks("
                "id text primary key,type text,status text,params_json text,result_json text not null default '{}',"
                "run_dir text,error text,cancel_requested integer not null default 0,created_at real,updated_at real)"
            )
            db.execute("create table if not exists artifacts(id text primary key,task_id text,name text,path text,kind text,created_at real)")
            for table in ["datasets", "scenes", "models"]:
                self._ensure_column(db, table, "source_type", "text not null default 'local'")
                self._ensure_column(db, table, "source_uri", "text not null default ''")
                self._ensure_column(db, table, "version", "text not null default ''")
                self._ensure_column(db, table, "parent_task_id", "text")
                self._ensure_column(db, table, "manifest_path", "text not null default ''")
                self._ensure_column(db, table, "metrics_json", "text not null default '{}'")
                self._ensure_column(db, table, "tags_json", "text not null default '[]'")
            self._ensure_column(db, "tasks", "pid", "integer")
            self._ensure_column(db, "tasks", "progress", "integer not null default 0")
            self._ensure_column(db, "tasks", "stage", "text not null default ''")
            self._ensure_column(db, "tasks", "message", "text not null default ''")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def _ensure_column(self, db: sqlite3.Connection, table: str, column: str, spec: str) -> None:
        columns = {row["name"] for row in db.execute(f"pragma table_info({table})").fetchall()}
        if column not in columns:
            db.execute(f"alter table {table} add column {column} {spec}")

    def _list(self, table: str, mapper) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(f"select * from {table} order by created_at desc").fetchall()
        return [mapper(row) for row in rows]

    def _get(self, table: str, item_id: str, mapper) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(f"select * from {table} where id=?", (item_id,)).fetchone()
        if row is None:
            raise KeyError(item_id)
        return mapper(row)

    def _update_entity(self, table: str, item_id: str, *, name: str | None, tags: list[str] | None, version: str | None) -> None:
        self._get(table, item_id, lambda row: row)
        fields: list[str] = []
        values: list[Any] = []
        if name is not None:
            fields.append("name=?")
            values.append(name)
        if tags is not None:
            fields.append("tags_json=?")
            values.append(_json(tags))
        if version is not None:
            fields.append("version=?")
            values.append(version)
        if not fields:
            return
        fields.append("updated_at=?")
        values.append(_now())
        values.append(item_id)
        with self._connect() as db:
            db.execute(f"update {table} set {', '.join(fields)} where id=?", values)

    def _delete(self, table: str, item_id: str) -> None:
        self._get(table, item_id, lambda row: row)
        with self._connect() as db:
            db.execute(f"delete from {table} where id=?", (item_id,))

    def _dataset_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {**self._entity_common(row), "scan": json.loads(row["scan_json"])}

    def _scene_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {**self._entity_common(row), "metadata": json.loads(row["metadata_json"])}

    def _model_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {**self._entity_common(row), "metadata": json.loads(row["metadata_json"])}

    def _entity_common(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "path": row["path"],
            "source_type": row["source_type"],
            "source_uri": row["source_uri"],
            "version": row["version"],
            "parent_task_id": row["parent_task_id"],
            "manifest_path": row["manifest_path"],
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "tags": json.loads(row["tags_json"] or "[]"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _task_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "type": row["type"],
            "status": row["status"],
            "params": json.loads(row["params_json"]),
            "result": json.loads(row["result_json"] or "{}"),
            "run_dir": row["run_dir"],
            "error": row["error"],
            "progress": int(row["progress"] or 0),
            "stage": row["stage"] or "",
            "message": row["message"] or "",
            "cancel_requested": bool(row["cancel_requested"]),
            "pid": row["pid"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _artifact_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "task_id": row["task_id"], "name": row["name"], "path": row["path"], "kind": row["kind"], "created_at": row["created_at"]}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _now() -> float:
    return time.time()
