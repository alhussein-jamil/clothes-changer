"""SQLite persistence — streamlined from v1."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

import argon2
from argon2 import PasswordHasher

from outfit_studio.config import get_settings
from outfit_studio.constants import (
    DEFAULT_NEW_USER_CREDITS,
    HISTORY_DB_LIMIT,
    MIN_PASSWORD_LENGTH,
)
from outfit_studio.db.schemas import UserOut

logger = logging.getLogger(__name__)
ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)


class DatabaseError(Exception):
    pass


def _row_to_user(row: sqlite3.Row) -> UserOut:
    created = row["created_at"]
    return UserOut(
        id=row["id"],
        username=row["username"],
        credits=row["credits"],
        is_admin=bool(row["is_admin"]),
        created_at=datetime.fromisoformat(created) if created else None,
    )


class Database:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        logger.info("Initializing database at %s", get_settings().resolved_db_path)
        self._init_schema()
        logger.debug("Database schema ready")

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        settings = get_settings()
        db_path = settings.resolved_db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        self._conn = conn
        return conn

    @contextmanager
    def _transaction(self):
        with self._lock:
            conn = self._ensure_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception("DB transaction rolled back")
                raise

    def _init_schema(self) -> None:
        with self._transaction() as conn:
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    credits INTEGER DEFAULT {DEFAULT_NEW_USER_CREDITS},
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1,
                    is_admin INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT,
                    prompt TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )
            self._migrate_columns(conn)
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                CREATE INDEX IF NOT EXISTS idx_images_user_id ON images(user_id);
                CREATE INDEX IF NOT EXISTS idx_images_user_created
                    ON images(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_users_signup_ip ON users(signup_ip);
                CREATE INDEX IF NOT EXISTS idx_users_device_fp ON users(device_fingerprint);
                """
            )

    @staticmethod
    def _migrate_columns(conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        migrations = {
            "email": "ALTER TABLE users ADD COLUMN email TEXT",
            "signup_ip": "ALTER TABLE users ADD COLUMN signup_ip TEXT",
            "device_fingerprint": "ALTER TABLE users ADD COLUMN device_fingerprint TEXT",
            "last_login_ip": "ALTER TABLE users ADD COLUMN last_login_ip TEXT",
        }
        for column, ddl in migrations.items():
            if column not in existing:
                conn.execute(ddl)

    def _enforce_signup_limits(
        self,
        conn: sqlite3.Connection,
        *,
        signup_ip: str | None,
        device_fingerprint: str | None,
    ) -> None:
        settings = get_settings()
        if settings.enforce_single_account_per_ip and signup_ip:
            row = conn.execute(
                "SELECT 1 FROM users WHERE signup_ip = ? LIMIT 1",
                (signup_ip,),
            ).fetchone()
            if row:
                raise DatabaseError("An account already exists from this network address")
        if settings.enforce_single_account_per_device and device_fingerprint:
            row = conn.execute(
                "SELECT 1 FROM users WHERE device_fingerprint = ? LIMIT 1",
                (device_fingerprint,),
            ).fetchone()
            if row:
                raise DatabaseError("An account already exists from this device")

    def register_user(
        self,
        username: str,
        password: str,
        credits: int = DEFAULT_NEW_USER_CREDITS,
        is_admin: bool = False,
        *,
        signup_ip: str | None = None,
        device_fingerprint: str | None = None,
    ) -> bool:
        if len(password) < MIN_PASSWORD_LENGTH:
            raise DatabaseError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
        try:
            with self._transaction() as conn:
                self._enforce_signup_limits(
                    conn,
                    signup_ip=signup_ip,
                    device_fingerprint=device_fingerprint,
                )
                conn.execute(
                    "INSERT INTO users "
                    "(username, password_hash, credits, is_admin, signup_ip, device_fingerprint) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        username,
                        ph.hash(password),
                        credits,
                        int(is_admin),
                        signup_ip,
                        device_fingerprint,
                    ),
                )
            logger.info(
                "Registered user %r (credits=%d, admin=%s, ip=%s)",
                username,
                credits,
                is_admin,
                signup_ip,
            )
            return True
        except sqlite3.IntegrityError as e:
            logger.warning("Registration failed — username %r already exists", username)
            raise DatabaseError("Username already exists") from e

    def record_login(self, username: str, ip: str | None) -> None:
        if not ip:
            return
        with self._transaction() as conn:
            conn.execute(
                "UPDATE users SET last_login_ip = ? WHERE username = ?",
                (ip, username),
            )

    def authenticate(self, username: str, password: str) -> bool:
        logger.debug("Authenticating user %r", username)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT id, password_hash, is_active FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if not row or not row["is_active"]:
            logger.info("Auth failed for %r (missing or inactive)", username)
            return False
        try:
            ph.verify(row["password_hash"], password)
        except argon2.exceptions.VerifyMismatchError:
            logger.info("Auth failed for %r (bad password)", username)
            return False
        logger.debug("Auth succeeded for %r", username)
        return True

    def get_user(self, username: str) -> UserOut | None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT id, username, credits, is_admin, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if not row:
            return None
        return _row_to_user(row)

    def user_exists(self, username: str) -> bool:
        with self._transaction() as conn:
            row = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        return row is not None

    def get_credits(self, username: str) -> int:
        user = self.get_user(username)
        return user.credits if user else 0

    def deduct_credit(self, username: str) -> bool:
        with self._transaction() as conn:
            row = conn.execute(
                "UPDATE users SET credits = credits - 1 "
                "WHERE username = ? AND credits > 0 AND is_active = 1 "
                "RETURNING credits",
                (username,),
            ).fetchone()
        if row is None:
            logger.warning("Could not deduct credit from %r", username)
            return False
        logger.info("Deducted 1 credit from %r (remaining=%d)", username, row["credits"])
        return True

    def set_credits(self, username: str, credits: int) -> bool:
        if credits < 0:
            logger.warning("set_credits rejected negative value for %r: %d", username, credits)
            return False
        with self._transaction() as conn:
            cur = conn.execute(
                "UPDATE users SET credits = ? WHERE username = ?", (credits, username)
            )
            ok = cur.rowcount > 0
        status = "ok" if ok else "user not found"
        logger.info("Set credits for %r → %d (%s)", username, credits, status)
        return ok

    def log_image(self, username: str, filename: str, prompt: str) -> None:
        with self._transaction() as conn:
            row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if not row:
                logger.warning("log_image: user %r not found", username)
                return
            conn.execute(
                "INSERT INTO images (user_id, filename, prompt) VALUES (?, ?, ?)",
                (row["id"], filename, prompt),
            )
        logger.info("Logged generation for %r: %s", username, filename)

    def get_history(self, username: str) -> list[dict]:
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT i.filename, i.prompt
                FROM images i
                JOIN users u ON u.id = i.user_id
                WHERE u.username = ?
                ORDER BY i.created_at DESC
                LIMIT ?
                """,
                (username, HISTORY_DB_LIMIT),
            ).fetchall()
        history = [dict(r) for r in rows]
        logger.debug("Fetched %d history rows for %r", len(history), username)
        return history

    def list_users(self) -> list[UserOut]:
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT id, username, credits, is_admin, created_at FROM users ORDER BY username"
            ).fetchall()
        users = [_row_to_user(r) for r in rows]
        logger.debug("Listed %d users", len(users))
        return users
