import json
import os
import sqlite3
import threading
import time
import traceback

from functools import wraps
from typing import Any

import gradio
import spaces

CONFIG = {
    "app_name": "StativerseDB",
    "bucket_dir": "/data",
    "tmp_dir": "/tmp",
    "use_bucket": os.path.isdir("/data"),
    "snapshot_db_path": os.path.join("/data" if os.path.isdir("/data") else "/tmp", "stativersedb.db"),
    "live_db_path": os.path.join("/tmp", "stativersedb-live.db") if os.path.isdir("/data") else os.path.join("/tmp", "stativersedb.db"),
    "write_concurrency_id": "stativersedb-write"
}

UNSET = object()

write_lock = threading.Lock()
snapshot_state_lock = threading.Lock()
last_snapshot_at = 0
last_snapshot_error = None

class AppError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

def now_ts() -> int:
    return int(time.time())

def ok_response(**payload) -> dict:
    return {"status": "ok", **payload}

def error_response(code: str, message: str) -> dict:
    return {"status": "error", "error": {"code": code, "message": message}}

def api_handler(fn):
    @wraps(fn)
    def wrapped(payload: dict | None = None) -> dict:
        try:
            return ok_response(**fn(require_payload(payload)))
        except AppError as exc:
            return error_response(exc.code, exc.message)
        except Exception:
            traceback.print_exc()
            return error_response("internal_error", "internal error")
    return wrapped

def require_payload(payload: dict | None) -> dict:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise AppError("invalid_payload", "payload must be an object")
    return payload

def require_name(payload: dict, field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise AppError("invalid_name", f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise AppError("invalid_name", f"{field_name} cannot be empty")
    return value

def optional_username(payload: dict):
    if "username" not in payload:
        return UNSET
    value = payload["username"]
    if value is None:
        return None
    if not isinstance(value, str):
        raise AppError("invalid_username", "username must be a string or null")
    value = value.strip()
    return value or None

def parse_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AppError("invalid_number", f"{field_name} must be a positive integer")
    if value <= 0:
        raise AppError("invalid_number", f"{field_name} must be a positive integer")
    return value

def require_positive_int(payload: dict, field_name: str) -> int:
    if field_name not in payload:
        raise AppError("missing_field", f"{field_name} is required")
    return parse_positive_int(payload[field_name], field_name)

def optional_positive_int(payload: dict, field_name: str):
    if field_name not in payload:
        return UNSET
    return parse_positive_int(payload[field_name], field_name)

def validate_json_value(value: Any) -> None:
    if value is None:
        raise AppError("invalid_value", "value cannot be null")
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise AppError("invalid_value", "value cannot contain NaN or Infinity")
        return
    if isinstance(value, str):
        return
    if isinstance(value, list):
        for item in value:
            validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AppError("invalid_value", "object keys must be strings")
            validate_json_value(item)
        return
    raise AppError("invalid_value", "value must be a number, string, boolean, array, or object")

def get_value_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"

def encode_value(value: Any) -> tuple[str, str]:
    validate_json_value(value)
    try:
        value_json = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        raise AppError("invalid_value", "value contains unsupported JSON data")
    return value_json, get_value_kind(value)

def decode_value(value_json: str) -> Any:
    return json.loads(value_json)

def open_connection(path: str, live: bool) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, cached_statements=256)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    if live:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -262144")
    return conn

def open_live_db() -> sqlite3.Connection:
    return open_connection(CONFIG["live_db_path"], live=True)

def open_snapshot_db() -> sqlite3.Connection:
    return open_connection(CONFIG["snapshot_db_path"], live=False)

def remove_live_files() -> None:
    for suffix in ("", "-wal", "-shm"):
        path = f"{CONFIG['live_db_path']}{suffix}"
        if os.path.exists(path):
            os.remove(path)

def restore_live_db() -> None:
    if not CONFIG["use_bucket"] or not os.path.isfile(CONFIG["snapshot_db_path"]):
        return
    remove_live_files()
    source = open_snapshot_db()
    target = open_live_db()
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()

def set_snapshot_state(error: str | None) -> None:
    global last_snapshot_at, last_snapshot_error
    with snapshot_state_lock:
        last_snapshot_at = now_ts()
        last_snapshot_error = error

def snapshot_live_db(source_conn: sqlite3.Connection | None = None) -> None:
    if not CONFIG["use_bucket"]:
        return
    target = open_snapshot_db()
    source = source_conn or open_live_db()
    close_source = source_conn is None
    try:
        source.backup(target)
        set_snapshot_state(None)
    except Exception as exc:
        print(f"SNAPSHOT | {exc}")
        set_snapshot_state(str(exc))
    finally:
        target.close()
        if close_source:
            source.close()

def init_db() -> None:
    conn = open_live_db()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                username TEXT UNIQUE,
                max_size_bytes INTEGER NOT NULL,
                used_size_bytes INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(user_id, name),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(project_id, name),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_id INTEGER NOT NULL,
                key_name TEXT NOT NULL,
                value_json TEXT NOT NULL,
                value_kind TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(collection_id, key_name),
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
            CREATE INDEX IF NOT EXISTS idx_collections_project_id ON collections(project_id);
            CREATE INDEX IF NOT EXISTS idx_keys_collection_id ON keys(collection_id);
            """
        )
        migrate_db(conn)
    finally:
        conn.close()

def migrate_db(conn: sqlite3.Connection) -> None:
    user_columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "used_size_bytes" not in user_columns:
        conn.execute("ALTER TABLE users ADD COLUMN used_size_bytes INTEGER NOT NULL DEFAULT 0")
    conn.execute(
        """
        UPDATE users
        SET used_size_bytes = COALESCE((
            SELECT SUM(length(CAST(k.value_json AS BLOB)))
            FROM keys k
            JOIN collections c ON c.id = k.collection_id
            JOIN projects p ON p.id = c.project_id
            WHERE p.user_id = users.id
        ), 0)
        """
    )

def initialize_storage() -> None:
    restore_live_db()
    init_db()
    if CONFIG["use_bucket"] and not os.path.isfile(CONFIG["snapshot_db_path"]):
        conn = open_live_db()
        try:
            snapshot_live_db(conn)
        finally:
            conn.close()

def run_read(fn):
    conn = open_live_db()
    try:
        return fn(conn)
    finally:
        conn.close()

def run_write(fn):
    with write_lock:
        conn = open_live_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            result = fn(conn)
            conn.execute("COMMIT")
            snapshot_live_db(conn)
            return result
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

def get_user_row(conn: sqlite3.Connection, user_id: str):
    return conn.execute(
        """
        SELECT id, user_id, username, max_size_bytes, used_size_bytes, created_at, updated_at
        FROM users
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchone()

def get_user_by_username(conn: sqlite3.Connection, username: str, exclude_id: int | None = None):
    if exclude_id is None:
        return conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,)
        ).fetchone()
    return conn.execute(
        "SELECT id FROM users WHERE username = ? AND id != ?",
        (username, exclude_id)
    ).fetchone()

def get_project_row(conn: sqlite3.Connection, user_row_id: int, project_name: str):
    return conn.execute(
        """
        SELECT id, name, created_at, updated_at
        FROM projects
        WHERE user_id = ? AND name = ?
        """,
        (user_row_id, project_name)
    ).fetchone()

def get_collection_row(conn: sqlite3.Connection, project_row_id: int, collection_name: str):
    return conn.execute(
        """
        SELECT id, name, created_at, updated_at
        FROM collections
        WHERE project_id = ? AND name = ?
        """,
        (project_row_id, collection_name)
    ).fetchone()

def get_project_scope(conn: sqlite3.Connection, user_id: str, project_name: str):
    return conn.execute(
        """
        SELECT
            u.id AS user_row_id,
            u.user_id,
            u.username,
            u.max_size_bytes,
            u.used_size_bytes,
            p.id AS project_row_id,
            p.name AS project_name
        FROM users u
        JOIN projects p ON p.user_id = u.id
        WHERE u.user_id = ? AND p.name = ?
        """,
        (user_id, project_name)
    ).fetchone()

def get_collection_scope(conn: sqlite3.Connection, user_id: str, project_name: str, collection_name: str):
    return conn.execute(
        """
        SELECT
            u.id AS user_row_id,
            u.user_id,
            u.username,
            u.max_size_bytes,
            u.used_size_bytes,
            p.id AS project_row_id,
            p.name AS project_name,
            c.id AS collection_row_id,
            c.name AS collection_name
        FROM users u
        JOIN projects p ON p.user_id = u.id
        JOIN collections c ON c.project_id = p.id
        WHERE u.user_id = ? AND p.name = ? AND c.name = ?
        """,
        (user_id, project_name, collection_name)
    ).fetchone()

def require_user_row(conn: sqlite3.Connection, user_id: str):
    row = get_user_row(conn, user_id)
    if not row:
        raise AppError("user_not_found", "user not found")
    return row

def require_project_row(conn: sqlite3.Connection, user_row_id: int, project_name: str):
    row = get_project_row(conn, user_row_id, project_name)
    if not row:
        raise AppError("project_not_found", "project not found")
    return row

def require_collection_row(conn: sqlite3.Connection, project_row_id: int, collection_name: str):
    row = get_collection_row(conn, project_row_id, collection_name)
    if not row:
        raise AppError("collection_not_found", "collection not found")
    return row

def require_project_scope(conn: sqlite3.Connection, user_id: str, project_name: str):
    row = get_project_scope(conn, user_id, project_name)
    if row:
        return row
    require_user_row(conn, user_id)
    raise AppError("project_not_found", "project not found")

def require_collection_scope(conn: sqlite3.Connection, user_id: str, project_name: str, collection_name: str):
    row = get_collection_scope(conn, user_id, project_name, collection_name)
    if row:
        return row
    project_scope = get_project_scope(conn, user_id, project_name)
    if project_scope:
        raise AppError("collection_not_found", "collection not found")
    require_user_row(conn, user_id)
    raise AppError("project_not_found", "project not found")

def get_total_value_size_for_project(conn: sqlite3.Connection, project_row_id: int) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(length(CAST(k.value_json AS BLOB))), 0) AS total_size
        FROM keys k
        JOIN collections c ON c.id = k.collection_id
        WHERE c.project_id = ?
        """,
        (project_row_id,)
    ).fetchone()
    return int(row["total_size"] or 0)

def get_total_value_size_for_collection(conn: sqlite3.Connection, collection_row_id: int) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(length(CAST(value_json AS BLOB))), 0) AS total_size
        FROM keys
        WHERE collection_id = ?
        """,
        (collection_row_id,)
    ).fetchone()
    return int(row["total_size"] or 0)

def set_user_used_size_bytes(conn: sqlite3.Connection, user_row_id: int, used_size_bytes: int) -> None:
    conn.execute(
        "UPDATE users SET used_size_bytes = ? WHERE id = ?",
        (used_size_bytes, user_row_id)
    )

def serialize_user(row: sqlite3.Row) -> dict:
    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "max_size_bytes": row["max_size_bytes"],
        "used_size_bytes": row["used_size_bytes"]
    }

def serialize_project(row: sqlite3.Row) -> dict:
    return {"name": row["name"]}

def serialize_collection(row: sqlite3.Row) -> dict:
    return {"name": row["name"]}

def serialize_key(row: sqlite3.Row) -> dict:
    return {"name": row["key_name"], "value": decode_value(row["value_json"])}

@api_handler
def create_user(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    username = optional_username(payload)
    max_size_bytes = require_positive_int(payload, "max_size_bytes")
    if username is UNSET:
        username = None

    def handler(conn: sqlite3.Connection):
        if get_user_row(conn, user_id):
            raise AppError("user_exists", "user already exists")
        if username is not None and get_user_by_username(conn, username):
            raise AppError("username_exists", "username already exists")
        timestamp = now_ts()
        conn.execute(
            """
            INSERT INTO users (user_id, username, max_size_bytes, used_size_bytes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, max_size_bytes, 0, timestamp, timestamp)
        )
        return {"user": {"user_id": user_id, "username": username, "max_size_bytes": max_size_bytes, "used_size_bytes": 0}}

    return run_write(handler)

@api_handler
def edit_user(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    username = optional_username(payload)
    max_size_bytes = optional_positive_int(payload, "max_size_bytes")

    def handler(conn: sqlite3.Connection):
        row = require_user_row(conn, user_id)
        updates = []
        values = []
        if username is not UNSET:
            if username is not None and get_user_by_username(conn, username, row["id"]):
                raise AppError("username_exists", "username already exists")
            updates.append("username = ?")
            values.append(username)
        if max_size_bytes is not UNSET:
            if max_size_bytes < row["used_size_bytes"]:
                raise AppError("quota_too_small", f"max_size_bytes cannot be lower than current usage ({row['used_size_bytes']} bytes)")
            updates.append("max_size_bytes = ?")
            values.append(max_size_bytes)
        if updates:
            values.extend([now_ts(), user_id])
            conn.execute(
                f"UPDATE users SET {', '.join(updates)}, updated_at = ? WHERE user_id = ?",
                values
            )
        return {
            "user": {
                "user_id": row["user_id"],
                "username": row["username"] if username is UNSET else username,
                "max_size_bytes": row["max_size_bytes"] if max_size_bytes is UNSET else max_size_bytes,
                "used_size_bytes": row["used_size_bytes"]
            }
        }

    return run_write(handler)

@api_handler
def get_user(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")

    def handler(conn: sqlite3.Connection):
        return {"user": serialize_user(require_user_row(conn, user_id))}

    return run_read(handler)

@api_handler
def delete_user(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")

    def handler(conn: sqlite3.Connection):
        row = require_user_row(conn, user_id)
        conn.execute("DELETE FROM users WHERE id = ?", (row["id"],))
        return {"deleted": True}

    return run_write(handler)

@api_handler
def create_project(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")

    def handler(conn: sqlite3.Connection):
        user_row = require_user_row(conn, user_id)
        if get_project_row(conn, user_row["id"], project_name):
            raise AppError("project_exists", "project already exists")
        timestamp = now_ts()
        conn.execute(
            """
            INSERT INTO projects (user_id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_row["id"], project_name, timestamp, timestamp)
        )
        return {"project": {"name": project_name}}

    return run_write(handler)

@api_handler
def edit_project(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    new_project_name = require_name(payload, "new_project_name")

    def handler(conn: sqlite3.Connection):
        user_row = require_user_row(conn, user_id)
        project_row = require_project_row(conn, user_row["id"], project_name)
        if project_name != new_project_name and get_project_row(conn, user_row["id"], new_project_name):
            raise AppError("project_exists", "project already exists")
        if project_name != new_project_name:
            conn.execute(
                "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
                (new_project_name, now_ts(), project_row["id"])
            )
        return {"project": {"name": new_project_name}}

    return run_write(handler)

@api_handler
def list_projects(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")

    def handler(conn: sqlite3.Connection):
        user_row = require_user_row(conn, user_id)
        rows = conn.execute(
            """
            SELECT name
            FROM projects
            WHERE user_id = ?
            ORDER BY name ASC
            """,
            (user_row["id"],)
        ).fetchall()
        return {"projects": [serialize_project(row) for row in rows]}

    return run_read(handler)

@api_handler
def delete_project(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")

    def handler(conn: sqlite3.Connection):
        project_scope = require_project_scope(conn, user_id, project_name)
        deleted_bytes = get_total_value_size_for_project(conn, project_scope["project_row_id"])
        conn.execute("DELETE FROM projects WHERE id = ?", (project_scope["project_row_id"],))
        if deleted_bytes:
            set_user_used_size_bytes(conn, project_scope["user_row_id"], project_scope["used_size_bytes"] - deleted_bytes)
        return {"deleted": True}

    return run_write(handler)

@api_handler
def create_collection(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    collection_name = require_name(payload, "collection_name")

    def handler(conn: sqlite3.Connection):
        project_scope = require_project_scope(conn, user_id, project_name)
        if get_collection_row(conn, project_scope["project_row_id"], collection_name):
            raise AppError("collection_exists", "collection already exists")
        timestamp = now_ts()
        conn.execute(
            """
            INSERT INTO collections (project_id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (project_scope["project_row_id"], collection_name, timestamp, timestamp)
        )
        return {"collection": {"name": collection_name}}

    return run_write(handler)

@api_handler
def edit_collection(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    collection_name = require_name(payload, "collection_name")
    new_collection_name = require_name(payload, "new_collection_name")

    def handler(conn: sqlite3.Connection):
        collection_scope = require_collection_scope(conn, user_id, project_name, collection_name)
        if collection_name != new_collection_name and get_collection_row(conn, collection_scope["project_row_id"], new_collection_name):
            raise AppError("collection_exists", "collection already exists")
        if collection_name != new_collection_name:
            conn.execute(
                "UPDATE collections SET name = ?, updated_at = ? WHERE id = ?",
                (new_collection_name, now_ts(), collection_scope["collection_row_id"])
            )
        return {"collection": {"name": new_collection_name}}

    return run_write(handler)

@api_handler
def list_collections(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")

    def handler(conn: sqlite3.Connection):
        project_scope = require_project_scope(conn, user_id, project_name)
        rows = conn.execute(
            """
            SELECT name
            FROM collections
            WHERE project_id = ?
            ORDER BY name ASC
            """,
            (project_scope["project_row_id"],)
        ).fetchall()
        return {"collections": [serialize_collection(row) for row in rows]}

    return run_read(handler)

@api_handler
def delete_collection(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    collection_name = require_name(payload, "collection_name")

    def handler(conn: sqlite3.Connection):
        collection_scope = require_collection_scope(conn, user_id, project_name, collection_name)
        deleted_bytes = get_total_value_size_for_collection(conn, collection_scope["collection_row_id"])
        conn.execute("DELETE FROM collections WHERE id = ?", (collection_scope["collection_row_id"],))
        if deleted_bytes:
            set_user_used_size_bytes(conn, collection_scope["user_row_id"], collection_scope["used_size_bytes"] - deleted_bytes)
        return {"deleted": True}

    return run_write(handler)

@api_handler
def set_value(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    collection_name = require_name(payload, "collection_name")
    key_name = require_name(payload, "key_name")
    if "value" not in payload:
        raise AppError("missing_field", "value is required")
    value_json, value_kind = encode_value(payload["value"])
    value_size = len(value_json.encode("utf-8"))

    def handler(conn: sqlite3.Connection):
        collection_scope = require_collection_scope(conn, user_id, project_name, collection_name)
        existing = conn.execute(
            """
            SELECT id, length(CAST(value_json AS BLOB)) AS value_size
            FROM keys
            WHERE collection_id = ? AND key_name = ?
            """,
            (collection_scope["collection_row_id"], key_name)
        ).fetchone()
        next_size_bytes = collection_scope["used_size_bytes"] - (int(existing["value_size"]) if existing else 0) + value_size
        if next_size_bytes > collection_scope["max_size_bytes"]:
            raise AppError("quota_exceeded", f"user data exceeds max_size_bytes ({collection_scope['max_size_bytes']} bytes)")
        timestamp = now_ts()
        conn.execute(
            """
            INSERT INTO keys (collection_id, key_name, value_json, value_kind, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(collection_id, key_name)
            DO UPDATE SET
                value_json = excluded.value_json,
                value_kind = excluded.value_kind,
                updated_at = excluded.updated_at
            """,
            (collection_scope["collection_row_id"], key_name, value_json, value_kind, timestamp, timestamp)
        )
        set_user_used_size_bytes(conn, collection_scope["user_row_id"], next_size_bytes)
        return {"created": existing is None, "key": {"name": key_name, "value": payload["value"]}}

    return run_write(handler)

@api_handler
def get_value(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    collection_name = require_name(payload, "collection_name")
    key_name = require_name(payload, "key_name")

    def handler(conn: sqlite3.Connection):
        collection_scope = require_collection_scope(conn, user_id, project_name, collection_name)
        row = conn.execute(
            """
            SELECT value_json
            FROM keys
            WHERE collection_id = ? AND key_name = ?
            """,
            (collection_scope["collection_row_id"], key_name)
        ).fetchone()
        return {"value": None if not row else decode_value(row["value_json"])}

    return run_read(handler)

@api_handler
def remove_value(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    collection_name = require_name(payload, "collection_name")
    key_name = require_name(payload, "key_name")

    def handler(conn: sqlite3.Connection):
        collection_scope = require_collection_scope(conn, user_id, project_name, collection_name)
        row = conn.execute(
            "SELECT id, length(CAST(value_json AS BLOB)) AS value_size FROM keys WHERE collection_id = ? AND key_name = ?",
            (collection_scope["collection_row_id"], key_name)
        ).fetchone()
        if not row:
            return {"removed": False}
        conn.execute("DELETE FROM keys WHERE id = ?", (row["id"],))
        set_user_used_size_bytes(conn, collection_scope["user_row_id"], collection_scope["used_size_bytes"] - int(row["value_size"]))
        return {"removed": True}

    return run_write(handler)

@api_handler
def list_values(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    collection_name = require_name(payload, "collection_name")

    def handler(conn: sqlite3.Connection):
        collection_scope = require_collection_scope(conn, user_id, project_name, collection_name)
        rows = conn.execute(
            """
            SELECT key_name, value_json
            FROM keys
            WHERE collection_id = ?
            ORDER BY key_name ASC
            """,
            (collection_scope["collection_row_id"],)
        ).fetchall()
        return {"data": {row["key_name"]: decode_value(row["value_json"]) for row in rows}}

    return run_read(handler)

def endpoint(payload: dict) -> dict:
    return {"received": payload, "message": "Working", "status": "ok"}

def ping() -> bool:
    print("SERVER | Space has been pinged.")
    return True

def health() -> dict:
    with snapshot_state_lock:
        return ok_response(
            app=CONFIG["app_name"],
            bucket_enabled=CONFIG["use_bucket"],
            live_db_path=CONFIG["live_db_path"],
            snapshot_db_path=CONFIG["snapshot_db_path"],
            last_snapshot_at=last_snapshot_at,
            last_snapshot_error=last_snapshot_error
        )

@spaces.GPU
def _(): return None

def register_api(fn, api_name: str, write: bool = False, queue: bool = True) -> None:
    kwargs = {"api_name": api_name, "queue": queue}
    if write:
        kwargs["concurrency_limit"] = 1
        kwargs["concurrency_id"] = CONFIG["write_concurrency_id"]
    gradio.api(fn, **kwargs)

initialize_storage()

with gradio.Blocks(title=CONFIG["app_name"]) as demo:
    register_api(endpoint, "endpoint")
    register_api(ping, "ping", queue=False)
    register_api(health, "health", queue=False)
    register_api(create_user, "create_user", write=True)
    register_api(edit_user, "edit_user", write=True)
    register_api(get_user, "get_user")
    register_api(delete_user, "delete_user", write=True)
    register_api(create_project, "create_project", write=True)
    register_api(edit_project, "edit_project", write=True)
    register_api(list_projects, "list_projects")
    register_api(delete_project, "delete_project", write=True)
    register_api(create_collection, "create_collection", write=True)
    register_api(edit_collection, "edit_collection", write=True)
    register_api(list_collections, "list_collections")
    register_api(delete_collection, "delete_collection", write=True)
    register_api(set_value, "set", write=True)
    register_api(get_value, "get")
    register_api(remove_value, "remove", write=True)
    register_api(list_values, "list")

demo.queue(default_concurrency_limit=16, max_size=256)

if os.environ.get("STATIVERSEDB_SKIP_LAUNCH") != "1": demo.launch(ssr_mode=False)
