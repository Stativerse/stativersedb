import base64
import json
import os
import sqlite3
import threading
import time
import traceback

from contextlib import contextmanager
from typing import Any

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS"
)

for env_name in THREAD_ENV_VARS:
    os.environ[env_name] = "2"

import uvicorn

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

try:
    import fcntl
except ImportError:
    fcntl = None

CONFIG = {
    "app_name": "StativerseDB",
    "bucket_dir": "/data",
    "tmp_dir": "/tmp",
    "use_bucket": os.path.isdir("/data"),
    "snapshot_db_path": os.path.join("/data" if os.path.isdir("/data") else "/tmp", "stativersedb.db"),
    "encrypted_snapshot_path": os.path.join("/data", "stativersedb.db.enc"),
    "live_db_path": os.path.join("/tmp", "stativersedb-live.db") if os.path.isdir("/data") else os.path.join("/tmp", "stativersedb.db"),
    "startup_lock_path": os.path.join("/tmp", "stativersedb-startup.lock"),
    "write_lock_path": os.path.join("/tmp", "stativersedb-write.lock"),
    "host": "0.0.0.0",
    "port": int(os.environ.get("PORT", "7860")),
    "workers": 2,
    "torch_num_threads": 2,
    "torch_num_interop_threads": 1,
    "sqlite_cache_size_kib": 131072,
    "encryption_chunk_bytes": 4 * 1024 * 1024,
    "encryption_magic": b"SVDBENC1"
}

UNSET = object()

write_lock = threading.Lock()
startup_lock = threading.Lock()
snapshot_state_lock = threading.Lock()
last_snapshot_at = 0
last_snapshot_error = None


def configure_runtime() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        torch.set_num_threads(CONFIG["torch_num_threads"])
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(CONFIG["torch_num_interop_threads"])
    except Exception:
        pass


configure_runtime()

app = FastAPI(title=CONFIG["app_name"], version="0.1.0")


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def now_ts() -> int:
    return int(time.time())


def ok_response(**payload) -> dict:
    return {"status": "ok", **payload}


def error_response(code: str, message: str) -> dict:
    return {"status": "error", "error": {"code": code, "message": message}}


def internal_error_response() -> dict:
    return error_response("internal_error", "internal error")


def handle_route(fn, payload: dict | None = None):
    try:
        return ok_response(**fn(require_payload(payload)))
    except AppError as exc:
        return JSONResponse(status_code=exc.status_code, content=error_response(exc.code, exc.message))
    except Exception:
        traceback.print_exc()
        return JSONResponse(status_code=500, content=internal_error_response())


def require_payload(payload: dict | None) -> dict:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise AppError("invalid_payload", "payload must be an object")
    return payload


def require_name(payload: dict, field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise AppError("invalid_name", f"{field_name} must be a string", 400)
    value = value.strip()
    if not value:
        raise AppError("invalid_name", f"{field_name} cannot be empty", 400)
    return value


def optional_username(payload: dict):
    if "username" not in payload:
        return UNSET
    value = payload["username"]
    if value is None:
        return None
    if not isinstance(value, str):
        raise AppError("invalid_username", "username must be a string or null", 400)
    value = value.strip()
    return value or None


def parse_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AppError("invalid_number", f"{field_name} must be a positive integer", 400)
    if value <= 0:
        raise AppError("invalid_number", f"{field_name} must be a positive integer", 400)
    return value


def require_positive_int(payload: dict, field_name: str) -> int:
    if field_name not in payload:
        raise AppError("missing_field", f"{field_name} is required", 400)
    return parse_positive_int(payload[field_name], field_name)


def optional_positive_int(payload: dict, field_name: str):
    if field_name not in payload:
        return UNSET
    return parse_positive_int(payload[field_name], field_name)


def validate_json_value(value: Any) -> None:
    if value is None:
        raise AppError("invalid_value", "value cannot be null", 400)
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise AppError("invalid_value", "value cannot contain NaN or Infinity", 400)
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
                raise AppError("invalid_value", "object keys must be strings", 400)
            validate_json_value(item)
        return
    raise AppError("invalid_value", "value must be a number, string, boolean, array, or object", 400)


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
        raise AppError("invalid_value", "value contains unsupported JSON data", 400)
    return value_json, get_value_kind(value)


def decode_value(value_json: str) -> Any:
    return json.loads(value_json)


def decode_secret_bytes(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    candidates = []
    if len(value.encode("utf-8")) == 32:
        candidates.append(value.encode("utf-8"))
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            decoded = decoder(padded)
        except Exception:
            continue
        if len(decoded) == 32:
            candidates.append(decoded)
    try:
        decoded = bytes.fromhex(value)
    except ValueError:
        decoded = None
    if decoded and len(decoded) == 32:
        candidates.append(decoded)
    if candidates:
        return candidates[0]
    raise RuntimeError("STATIVERSEDB_ENCRYPTION_KEY must decode to exactly 32 bytes")


def load_encryption_key() -> bytes | None:
    if not CONFIG["use_bucket"]:
        return None
    raw_value = os.environ.get("STATIVERSEDB_ENCRYPTION_KEY", "").strip()
    if not raw_value:
        raise RuntimeError("STATIVERSEDB_ENCRYPTION_KEY is required when a bucket is attached")
    return decode_secret_bytes(raw_value)


ENCRYPTION_KEY = load_encryption_key()


@contextmanager
def startup_guard():
    os.makedirs(os.path.dirname(CONFIG["startup_lock_path"]), exist_ok=True)
    if fcntl is None:
        with startup_lock:
            yield
        return
    with open(CONFIG["startup_lock_path"], "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def write_guard():
    os.makedirs(os.path.dirname(CONFIG["write_lock_path"]), exist_ok=True)
    if fcntl is None:
        with write_lock:
            yield
        return
    with write_lock:
        with open(CONFIG["write_lock_path"], "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def open_connection(path: str, live: bool) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, cached_statements=256)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    if live:
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA cache_size = -{CONFIG['sqlite_cache_size_kib']}")
    return conn


def open_live_db() -> sqlite3.Connection:
    return open_connection(CONFIG["live_db_path"], live=True)


def open_snapshot_db() -> sqlite3.Connection:
    return open_connection(CONFIG["snapshot_db_path"], live=False)


def atomic_replace(source_path: str, target_path: str) -> None:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(source_path, "rb") as source_file:
        os.fsync(source_file.fileno())
    os.replace(source_path, target_path)


def encrypt_file(source_path: str, target_path: str) -> None:
    temp_target_path = f"{target_path}.tmp"
    nonce = os.urandom(12)
    encryptor = Cipher(algorithms.AES(ENCRYPTION_KEY), modes.GCM(nonce)).encryptor()
    with open(source_path, "rb") as source_file, open(temp_target_path, "wb") as target_file:
        target_file.write(CONFIG["encryption_magic"])
        target_file.write(nonce)
        while True:
            chunk = source_file.read(CONFIG["encryption_chunk_bytes"])
            if not chunk:
                break
            target_file.write(encryptor.update(chunk))
        encryptor.finalize()
        target_file.write(encryptor.tag)
        target_file.flush()
        os.fsync(target_file.fileno())
    atomic_replace(temp_target_path, target_path)


def decrypt_file(source_path: str, target_path: str) -> None:
    temp_target_path = f"{target_path}.tmp"
    source_size = os.path.getsize(source_path)
    header_size = len(CONFIG["encryption_magic"]) + 12 + 16
    if source_size <= header_size:
        raise RuntimeError("encrypted snapshot is invalid")
    with open(source_path, "rb") as source_file:
        if source_file.read(len(CONFIG["encryption_magic"])) != CONFIG["encryption_magic"]:
            raise RuntimeError("encrypted snapshot header is invalid")
        nonce = source_file.read(12)
        ciphertext_size = source_size - len(CONFIG["encryption_magic"]) - 12 - 16
        source_file.seek(source_size - 16)
        tag = source_file.read(16)
        source_file.seek(len(CONFIG["encryption_magic"]) + 12)
        decryptor = Cipher(algorithms.AES(ENCRYPTION_KEY), modes.GCM(nonce, tag)).decryptor()
        remaining = ciphertext_size
        with open(temp_target_path, "wb") as target_file:
            while remaining > 0:
                chunk = source_file.read(min(CONFIG["encryption_chunk_bytes"], remaining))
                if not chunk:
                    raise RuntimeError("encrypted snapshot payload is truncated")
                target_file.write(decryptor.update(chunk))
                remaining -= len(chunk)
            decryptor.finalize()
            target_file.flush()
            os.fsync(target_file.fileno())
    atomic_replace(temp_target_path, target_path)


def remove_live_files() -> None:
    for suffix in ("", "-wal", "-shm"):
        path = f"{CONFIG['live_db_path']}{suffix}"
        if os.path.exists(path):
            os.remove(path)


def restore_live_db() -> None:
    if not CONFIG["use_bucket"]:
        return
    if os.path.isfile(CONFIG["encrypted_snapshot_path"]):
        remove_live_files()
        decrypt_file(CONFIG["encrypted_snapshot_path"], CONFIG["live_db_path"])
        return
    if not os.path.isfile(CONFIG["snapshot_db_path"]):
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
    source = source_conn or open_live_db()
    close_source = source_conn is None
    try:
        checkpoint_row = source.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
        if checkpoint_row and int(checkpoint_row[0]) != 0:
            raise RuntimeError("wal checkpoint did not complete before bucket snapshot")
        encrypt_file(CONFIG["live_db_path"], CONFIG["encrypted_snapshot_path"])
        if os.path.isfile(CONFIG["snapshot_db_path"]):
            os.remove(CONFIG["snapshot_db_path"])
        set_snapshot_state(None)
    except Exception as exc:
        print(f"SNAPSHOT | {exc}")
        set_snapshot_state(str(exc))
    finally:
        if close_source:
            source.close()


def init_db() -> None:
    conn = open_live_db()
    try:
        conn.execute("PRAGMA journal_mode = WAL")
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
    with startup_guard():
        if not os.path.isfile(CONFIG["live_db_path"]):
            restore_live_db()
        init_db()
        if CONFIG["use_bucket"] and not os.path.isfile(CONFIG["encrypted_snapshot_path"]):
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
    with write_guard():
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
        return conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    return conn.execute("SELECT id FROM users WHERE username = ? AND id != ?", (username, exclude_id)).fetchone()


def get_project_row(conn: sqlite3.Connection, user_row_id: int, project_name: str):
    return conn.execute(
        """
        SELECT id, name
        FROM projects
        WHERE user_id = ? AND name = ?
        """,
        (user_row_id, project_name)
    ).fetchone()


def get_collection_row(conn: sqlite3.Connection, project_row_id: int, collection_name: str):
    return conn.execute(
        """
        SELECT id, name
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
        raise AppError("user_not_found", "user not found", 404)
    return row


def require_project_scope(conn: sqlite3.Connection, user_id: str, project_name: str):
    row = get_project_scope(conn, user_id, project_name)
    if row:
        return row
    require_user_row(conn, user_id)
    raise AppError("project_not_found", "project not found", 404)


def require_collection_scope(conn: sqlite3.Connection, user_id: str, project_name: str, collection_name: str):
    row = get_collection_scope(conn, user_id, project_name, collection_name)
    if row:
        return row
    project_scope = get_project_scope(conn, user_id, project_name)
    if project_scope:
        raise AppError("collection_not_found", "collection not found", 404)
    require_user_row(conn, user_id)
    raise AppError("project_not_found", "project not found", 404)


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
    conn.execute("UPDATE users SET used_size_bytes = ? WHERE id = ?", (max(used_size_bytes, 0), user_row_id))


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


def create_user(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    username = optional_username(payload)
    max_size_bytes = require_positive_int(payload, "max_size_bytes")
    if username is UNSET:
        username = None

    def handler(conn: sqlite3.Connection):
        if get_user_row(conn, user_id):
            raise AppError("user_exists", "user already exists", 409)
        if username is not None and get_user_by_username(conn, username):
            raise AppError("username_exists", "username already exists", 409)
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
                raise AppError("username_exists", "username already exists", 409)
            updates.append("username = ?")
            values.append(username)
        if max_size_bytes is not UNSET:
            if max_size_bytes < row["used_size_bytes"]:
                raise AppError("quota_too_small", f"max_size_bytes cannot be lower than current usage ({row['used_size_bytes']} bytes)", 400)
            updates.append("max_size_bytes = ?")
            values.append(max_size_bytes)
        if updates:
            values.extend([now_ts(), user_id])
            conn.execute(f"UPDATE users SET {', '.join(updates)}, updated_at = ? WHERE user_id = ?", values)
        return {
            "user": {
                "user_id": row["user_id"],
                "username": row["username"] if username is UNSET else username,
                "max_size_bytes": row["max_size_bytes"] if max_size_bytes is UNSET else max_size_bytes,
                "used_size_bytes": row["used_size_bytes"]
            }
        }

    return run_write(handler)


def get_user(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")

    def handler(conn: sqlite3.Connection):
        return {"user": serialize_user(require_user_row(conn, user_id))}

    return run_read(handler)


def list_users(payload: dict) -> dict:
    require_payload(payload)

    def handler(conn: sqlite3.Connection):
        rows = conn.execute(
            """
            SELECT user_id, username, max_size_bytes, used_size_bytes
            FROM users
            ORDER BY user_id ASC
            """
        ).fetchall()
        return {"users": [serialize_user(row) for row in rows]}

    return run_read(handler)


def delete_user(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")

    def handler(conn: sqlite3.Connection):
        row = require_user_row(conn, user_id)
        conn.execute("DELETE FROM users WHERE id = ?", (row["id"],))
        return {"deleted": True}

    return run_write(handler)


def create_project(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")

    def handler(conn: sqlite3.Connection):
        user_row = require_user_row(conn, user_id)
        if get_project_row(conn, user_row["id"], project_name):
            raise AppError("project_exists", "project already exists", 409)
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


def edit_project(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    new_project_name = require_name(payload, "new_project_name")

    def handler(conn: sqlite3.Connection):
        project_scope = require_project_scope(conn, user_id, project_name)
        if project_name != new_project_name and get_project_row(conn, project_scope["user_row_id"], new_project_name):
            raise AppError("project_exists", "project already exists", 409)
        if project_name != new_project_name:
            conn.execute(
                "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
                (new_project_name, now_ts(), project_scope["project_row_id"])
            )
        return {"project": {"name": new_project_name}}

    return run_write(handler)


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


def create_collection(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    collection_name = require_name(payload, "collection_name")

    def handler(conn: sqlite3.Connection):
        project_scope = require_project_scope(conn, user_id, project_name)
        if get_collection_row(conn, project_scope["project_row_id"], collection_name):
            raise AppError("collection_exists", "collection already exists", 409)
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


def edit_collection(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    collection_name = require_name(payload, "collection_name")
    new_collection_name = require_name(payload, "new_collection_name")

    def handler(conn: sqlite3.Connection):
        collection_scope = require_collection_scope(conn, user_id, project_name, collection_name)
        if collection_name != new_collection_name and get_collection_row(conn, collection_scope["project_row_id"], new_collection_name):
            raise AppError("collection_exists", "collection already exists", 409)
        if collection_name != new_collection_name:
            conn.execute(
                "UPDATE collections SET name = ?, updated_at = ? WHERE id = ?",
                (new_collection_name, now_ts(), collection_scope["collection_row_id"])
            )
        return {"collection": {"name": new_collection_name}}

    return run_write(handler)


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


def set_value(payload: dict) -> dict:
    user_id = require_name(payload, "user_id")
    project_name = require_name(payload, "project_name")
    collection_name = require_name(payload, "collection_name")
    key_name = require_name(payload, "key_name")
    if "value" not in payload:
        raise AppError("missing_field", "value is required", 400)
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
            raise AppError("quota_exceeded", f"user data exceeds max_size_bytes ({collection_scope['max_size_bytes']} bytes)", 400)
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


def execute_batch_action(action: str, payload: dict | None) -> dict:
    if action == "endpoint":
        parsed_payload = require_payload(payload)
        return ok_response(received=parsed_payload, message="Working")
    if action == "ping":
        print("SERVER | Docker Space has been pinged.")
        return ok_response(ping=True)
    if action == "health":
        with snapshot_state_lock:
            return ok_response(
                app=CONFIG["app_name"],
                bucket_enabled=CONFIG["use_bucket"],
                bucket_encryption_enabled=CONFIG["use_bucket"],
                live_db_path=CONFIG["live_db_path"],
                snapshot_db_path=CONFIG["encrypted_snapshot_path"] if CONFIG["use_bucket"] else CONFIG["snapshot_db_path"],
                last_snapshot_at=last_snapshot_at,
                last_snapshot_error=last_snapshot_error
            )
    handler = BATCH_ACTIONS.get(action)
    if handler is None:
        return error_response("invalid_action", "unknown action")
    try:
        return ok_response(**handler(payload))
    except AppError as exc:
        return error_response(exc.code, exc.message)
    except Exception:
        traceback.print_exc()
        return internal_error_response()


def batch_route_handler(payload: dict | None) -> dict:
    body = require_payload(payload)
    requests = body.get("requests")
    if not isinstance(requests, list) or not requests:
        raise AppError("invalid_payload", "requests must be a non-empty array", 400)
    continue_on_error = body.get("continue_on_error", True)
    if not isinstance(continue_on_error, bool):
        raise AppError("invalid_payload", "continue_on_error must be a boolean", 400)

    results = []
    stopped = False
    for index, item in enumerate(requests):
        if not isinstance(item, dict):
            result = error_response("invalid_payload", "batch item must be an object")
            results.append({"index": index, **result})
            if not continue_on_error:
                stopped = True
                break
            continue
        action = item.get("action")
        if not isinstance(action, str) or not action.strip():
            result = error_response("invalid_action", "action must be a non-empty string")
        else:
            result = execute_batch_action(action.strip(), item.get("payload"))
        item_result = {"index": index, "action": action, **result}
        if "id" in item:
            item_result["id"] = item["id"]
        results.append(item_result)
        if result["status"] == "error" and not continue_on_error:
            stopped = True
            break
    return {"results": results, "stopped": stopped}


BATCH_ACTIONS = {
    "create_user": create_user,
    "edit_user": edit_user,
    "get_user": get_user,
    "list_users": list_users,
    "delete_user": delete_user,
    "create_project": create_project,
    "edit_project": edit_project,
    "list_projects": list_projects,
    "delete_project": delete_project,
    "create_collection": create_collection,
    "edit_collection": edit_collection,
    "list_collections": list_collections,
    "delete_collection": delete_collection,
    "set": set_value,
    "get": get_value,
    "remove": remove_value,
    "list": list_values
}


@app.on_event("startup")
def startup() -> None:
    initialize_storage()


@app.post("/endpoint")
def endpoint_route(payload: dict | None = Body(default=None)):
    parsed_payload = require_payload(payload)
    return {"status": "ok", "received": parsed_payload, "message": "Working"}


@app.get("/ping")
def ping_route():
    print("SERVER | Docker Space has been pinged.")
    return {"status": "ok", "ping": True}


@app.get("/health")
def health_route():
    with snapshot_state_lock:
        return ok_response(
            app=CONFIG["app_name"],
            bucket_enabled=CONFIG["use_bucket"],
            bucket_encryption_enabled=CONFIG["use_bucket"],
            live_db_path=CONFIG["live_db_path"],
            snapshot_db_path=CONFIG["encrypted_snapshot_path"] if CONFIG["use_bucket"] else CONFIG["snapshot_db_path"],
            last_snapshot_at=last_snapshot_at,
            last_snapshot_error=last_snapshot_error
        )


@app.post("/batch")
def batch_route(payload: dict | None = Body(default=None)):
    return handle_route(batch_route_handler, payload)


def register_post(path: str, fn) -> None:
    def route(payload: dict | None = Body(default=None)):
        return handle_route(fn, payload)
    route.__name__ = f"{fn.__name__}_{path.strip('/').replace('/', '_')}_route"
    app.add_api_route(path, route, methods=["POST"])


register_post("/create_user", create_user)
register_post("/edit_user", edit_user)
register_post("/get_user", get_user)
register_post("/list_users", list_users)
register_post("/delete_user", delete_user)
register_post("/create_project", create_project)
register_post("/edit_project", edit_project)
register_post("/list_projects", list_projects)
register_post("/delete_project", delete_project)
register_post("/create_collection", create_collection)
register_post("/edit_collection", edit_collection)
register_post("/list_collections", list_collections)
register_post("/delete_collection", delete_collection)
register_post("/set", set_value)
register_post("/get", get_value)
register_post("/remove", remove_value)
register_post("/list", list_values)


if __name__ == "__main__":
    uvicorn.run("app:app", host=CONFIG["host"], port=CONFIG["port"], workers=CONFIG["workers"])
