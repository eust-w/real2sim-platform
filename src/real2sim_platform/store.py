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
        storage_uri: str = "",
        checksum: str = "",
        cache_state: str = "local",
    ) -> dict[str, Any]:
        item_id = item_id or uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into datasets(id,name,path,scan_json,source_type,source_uri,version,parent_task_id,manifest_path,metrics_json,tags_json,storage_uri,checksum,cache_state,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item_id, name, str(path), _json(scan), source_type, source_uri, version, parent_task_id, str(manifest_path or ""), _json(metrics or {}), _json(tags or []), storage_uri, checksum, cache_state, now, now),
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
        storage_uri: str = "",
        checksum: str = "",
        cache_state: str = "local",
    ) -> dict[str, Any]:
        item_id = item_id or uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into scenes(id,name,path,metadata_json,source_type,source_uri,version,parent_task_id,manifest_path,metrics_json,tags_json,storage_uri,checksum,cache_state,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item_id, name, str(path), _json(metadata), source_type, source_uri, version, parent_task_id, str(manifest_path or ""), _json(metrics or {}), _json(tags or []), storage_uri, checksum, cache_state, now, now),
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
        storage_uri: str = "",
        checksum: str = "",
        cache_state: str = "local",
    ) -> dict[str, Any]:
        item_id = item_id or uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into models(id,name,path,metadata_json,source_type,source_uri,version,parent_task_id,manifest_path,metrics_json,tags_json,storage_uri,checksum,cache_state,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item_id, name, str(path), _json(metadata), source_type, source_uri, version, parent_task_id, str(manifest_path or ""), _json(metrics or {}), _json(tags or []), storage_uri, checksum, cache_state, now, now),
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

    def update_entity_storage(self, table: str, item_id: str, *, storage_uri: str, checksum: str = "", cache_state: str = "local") -> dict[str, Any]:
        if table not in {"datasets", "scenes", "models"}:
            raise ValueError("unsupported entity table")
        mapper = {"datasets": self._dataset_row, "scenes": self._scene_row, "models": self._model_row}[table]
        self._get(table, item_id, mapper)
        with self._connect() as db:
            db.execute(
                f"update {table} set storage_uri=?,checksum=?,cache_state=?,updated_at=? where id=?",
                (storage_uri, checksum, cache_state, _now(), item_id),
            )
        return self._get(table, item_id, mapper)

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

    def claim_next_queued_task(self, worker_id: str) -> dict[str, Any] | None:
        db = self._connect()
        try:
            db.execute("begin immediate")
            row = db.execute("select * from tasks where status='queued' order by created_at limit 1").fetchone()
            if row is None:
                db.commit()
                return None
            now = _now()
            cursor = db.execute(
                "update tasks set status='running',worker_id=?,claimed_at=?,attempt=attempt+1,progress=max(progress,1),stage='启动中',message='GPU worker 已领取任务',updated_at=? where id=? and status='queued'",
                (worker_id, now, now, row["id"]),
            )
            if cursor.rowcount != 1:
                db.rollback()
                return None
            claimed = db.execute("select * from tasks where id=?", (row["id"],)).fetchone()
            db.commit()
            return self._task_row(claimed)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def next_queued_task(self) -> dict[str, Any] | None:
        """Compatibility helper for older callers; new workers must claim atomically."""
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

    def create_user(self, *, username: str, password_hash: str, role: str, must_change_password: bool = True) -> dict[str, Any]:
        item_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into users(id,username,password_hash,role,disabled,must_change_password,created_at,updated_at) values(?,?,?,?,0,?,?,?)",
                (item_id, username, password_hash, role, int(must_change_password), now, now),
            )
        return self.get_user(item_id)

    def count_users(self) -> int:
        with self._connect() as db:
            return int(db.execute("select count(*) from users").fetchone()[0])

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("select * from users order by created_at").fetchall()
        return [self._user_row(row) for row in rows]

    def get_user(self, user_id: str) -> dict[str, Any]:
        return self._get("users", user_id, self._user_row)

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("select * from users where lower(username)=lower(?)", (username,)).fetchone()
        return self._user_row(row, include_password=True) if row else None

    def update_user(
        self,
        user_id: str,
        *,
        role: str | None = None,
        disabled: bool | None = None,
        password_hash: str | None = None,
        must_change_password: bool | None = None,
        last_login: float | None = None,
    ) -> dict[str, Any]:
        self.get_user(user_id)
        fields: list[str] = []
        values: list[Any] = []
        for name, value in (("role", role), ("password_hash", password_hash), ("last_login", last_login)):
            if value is not None:
                fields.append(f"{name}=?")
                values.append(value)
        for name, value in (("disabled", disabled), ("must_change_password", must_change_password)):
            if value is not None:
                fields.append(f"{name}=?")
                values.append(int(value))
        if fields:
            fields.append("updated_at=?")
            values.extend([_now(), user_id])
            with self._connect() as db:
                db.execute(f"update users set {','.join(fields)} where id=?", values)
        return self.get_user(user_id)

    def delete_user(self, user_id: str) -> None:
        self._delete("users", user_id)

    def create_session(self, *, token_hash: str, user_id: str, expires_at: float) -> None:
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into sessions(token_hash,user_id,expires_at,created_at,last_seen) values(?,?,?,?,?)",
                (token_hash, user_id, expires_at, now, now),
            )

    def get_session_user(self, token_hash: str) -> dict[str, Any] | None:
        now = _now()
        with self._connect() as db:
            row = db.execute(
                "select users.* from sessions join users on users.id=sessions.user_id where sessions.token_hash=? and sessions.expires_at>? and users.disabled=0",
                (token_hash, now),
            ).fetchone()
            if row:
                db.execute("update sessions set last_seen=? where token_hash=?", (now, token_hash))
            db.execute("delete from sessions where expires_at<=?", (now,))
        return self._user_row(row) if row else None

    def delete_session(self, token_hash: str) -> None:
        with self._connect() as db:
            db.execute("delete from sessions where token_hash=?", (token_hash,))

    def delete_user_sessions(self, user_id: str) -> None:
        with self._connect() as db:
            db.execute("delete from sessions where user_id=?", (user_id,))

    def create_upload(
        self,
        *,
        owner_user_id: str | None,
        filename: str,
        content_type: str,
        total_size: int,
        chunk_size: int,
        total_parts: int,
        checksum_expected: str,
        local_dir: Path,
        item_id: str | None = None,
    ) -> dict[str, Any]:
        item_id = item_id or uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into uploads(id,owner_user_id,filename,content_type,total_size,chunk_size,total_parts,received_parts_json,checksum_expected,checksum_actual,status,local_dir,storage_uri,error,created_at,updated_at) values(?,?,?,?,?,?,?,'[]',?,?,'uploading',?,'','',?,?)",
                (item_id, owner_user_id, filename, content_type, total_size, chunk_size, total_parts, checksum_expected, "", str(local_dir), now, now),
            )
        return self.get_upload(item_id)

    def get_upload(self, upload_id: str) -> dict[str, Any]:
        return self._get("uploads", upload_id, self._upload_row)

    def record_upload_part(self, upload_id: str, part_number: int) -> dict[str, Any]:
        upload = self.get_upload(upload_id)
        parts = sorted(set([*upload["received_parts"], int(part_number)]))
        with self._connect() as db:
            db.execute("update uploads set received_parts_json=?,updated_at=? where id=?", (_json(parts), _now(), upload_id))
        return self.get_upload(upload_id)

    def complete_upload(self, upload_id: str, *, checksum_actual: str, storage_uri: str) -> dict[str, Any]:
        with self._connect() as db:
            db.execute(
                "update uploads set status='completed',checksum_actual=?,storage_uri=?,error='',updated_at=? where id=?",
                (checksum_actual, storage_uri, _now(), upload_id),
            )
        return self.get_upload(upload_id)

    def fail_upload(self, upload_id: str, error: str) -> dict[str, Any]:
        with self._connect() as db:
            db.execute("update uploads set status='failed',error=?,updated_at=? where id=?", (error, _now(), upload_id))
        return self.get_upload(upload_id)

    def delete_upload(self, upload_id: str) -> None:
        self._delete("uploads", upload_id)

    def update_worker_heartbeat(self, worker_id: str, payload: dict[str, Any]) -> None:
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into worker_heartbeats(worker_id,payload_json,updated_at) values(?,?,?) on conflict(worker_id) do update set payload_json=excluded.payload_json,updated_at=excluded.updated_at",
                (worker_id, _json(payload), now),
            )

    def list_worker_heartbeats(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("select * from worker_heartbeats order by updated_at desc").fetchall()
        return [{"worker_id": row["worker_id"], "payload": json.loads(row["payload_json"] or "{}"), "updated_at": row["updated_at"]} for row in rows]

    def list_task_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("select * from artifacts where task_id=? order by created_at", (task_id,)).fetchall()
        return [self._artifact_row(row) for row in rows]

    def get_artifact(self, item_id: str) -> dict[str, Any]:
        return self._get("artifacts", item_id, self._artifact_row)

    def _init(self) -> None:
        with self._connect() as db:
            db.execute("pragma journal_mode=WAL")
            db.execute("pragma synchronous=NORMAL")
            db.execute("create table if not exists datasets(id text primary key,name text,path text,scan_json text,created_at real,updated_at real)")
            db.execute("create table if not exists scenes(id text primary key,name text,path text,metadata_json text,created_at real,updated_at real)")
            db.execute("create table if not exists models(id text primary key,name text,path text,metadata_json text,created_at real,updated_at real)")
            db.execute(
                "create table if not exists tasks("
                "id text primary key,type text,status text,params_json text,result_json text not null default '{}',"
                "run_dir text,error text,cancel_requested integer not null default 0,created_at real,updated_at real)"
            )
            db.execute("create table if not exists artifacts(id text primary key,task_id text,name text,path text,kind text,created_at real)")
            db.execute(
                "create table if not exists users(id text primary key,username text not null unique,password_hash text not null,role text not null,disabled integer not null default 0,must_change_password integer not null default 1,last_login real,created_at real,updated_at real)"
            )
            db.execute(
                "create table if not exists sessions(token_hash text primary key,user_id text not null,expires_at real not null,created_at real,last_seen real,foreign key(user_id) references users(id) on delete cascade)"
            )
            db.execute(
                "create table if not exists uploads(id text primary key,owner_user_id text,filename text not null,content_type text,total_size integer not null,chunk_size integer not null,total_parts integer not null,received_parts_json text not null default '[]',checksum_expected text,checksum_actual text,status text not null,local_dir text not null,storage_uri text,error text,created_at real,updated_at real)"
            )
            db.execute("create table if not exists worker_heartbeats(worker_id text primary key,payload_json text not null,updated_at real not null)")
            for table in ["datasets", "scenes", "models"]:
                self._ensure_column(db, table, "source_type", "text not null default 'local'")
                self._ensure_column(db, table, "source_uri", "text not null default ''")
                self._ensure_column(db, table, "version", "text not null default ''")
                self._ensure_column(db, table, "parent_task_id", "text")
                self._ensure_column(db, table, "manifest_path", "text not null default ''")
                self._ensure_column(db, table, "metrics_json", "text not null default '{}'")
                self._ensure_column(db, table, "tags_json", "text not null default '[]'")
                self._ensure_column(db, table, "storage_uri", "text not null default ''")
                self._ensure_column(db, table, "checksum", "text not null default ''")
                self._ensure_column(db, table, "cache_state", "text not null default 'local'")
            self._ensure_column(db, "tasks", "pid", "integer")
            self._ensure_column(db, "tasks", "progress", "integer not null default 0")
            self._ensure_column(db, "tasks", "stage", "text not null default ''")
            self._ensure_column(db, "tasks", "message", "text not null default ''")
            self._ensure_column(db, "tasks", "worker_id", "text not null default ''")
            self._ensure_column(db, "tasks", "claimed_at", "real")
            self._ensure_column(db, "tasks", "attempt", "integer not null default 0")
            db.execute("create index if not exists idx_tasks_status_created on tasks(status,created_at)")
            db.execute("create index if not exists idx_sessions_user on sessions(user_id)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("pragma busy_timeout=5000")
        db.execute("pragma foreign_keys=on")
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
            "storage_uri": row["storage_uri"],
            "checksum": row["checksum"],
            "cache_state": row["cache_state"],
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
            "worker_id": row["worker_id"],
            "claimed_at": row["claimed_at"],
            "attempt": int(row["attempt"] or 0),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _artifact_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "task_id": row["task_id"], "name": row["name"], "path": row["path"], "kind": row["kind"], "created_at": row["created_at"]}

    def _user_row(self, row: sqlite3.Row, *, include_password: bool = False) -> dict[str, Any]:
        item = {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "disabled": bool(row["disabled"]),
            "must_change_password": bool(row["must_change_password"]),
            "last_login": row["last_login"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if include_password:
            item["password_hash"] = row["password_hash"]
        return item

    def _upload_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "owner_user_id": row["owner_user_id"],
            "filename": row["filename"],
            "content_type": row["content_type"] or "application/octet-stream",
            "total_size": int(row["total_size"]),
            "chunk_size": int(row["chunk_size"]),
            "total_parts": int(row["total_parts"]),
            "received_parts": json.loads(row["received_parts_json"] or "[]"),
            "checksum_expected": row["checksum_expected"] or "",
            "checksum_actual": row["checksum_actual"] or "",
            "status": row["status"],
            "local_dir": row["local_dir"],
            "storage_uri": row["storage_uri"] or "",
            "error": row["error"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _now() -> float:
    return time.time()
