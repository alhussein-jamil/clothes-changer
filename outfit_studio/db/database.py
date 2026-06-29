"""SQLite persistence — streamlined from v1."""

from __future__ import annotations

import logging
import sqlite3
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
from outfit_studio.schemas import UserOut

logger = logging.getLogger(__name__)
ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)


class DatabaseError(Exception):
    pass


@contextmanager
def _conn():
    settings = get_settings()
    settings.resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Opening DB connection: %s", settings.resolved_db_path)
    conn = sqlite3.connect(str(settings.resolved_db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("DB transaction rolled back")
        raise
    finally:
        conn.close()


class Database:
    def __init__(self) -> None:
        logger.info("Initializing database at %s", get_settings().resolved_db_path)
        self._init_schema()
        logger.debug("Database schema ready")

    def _init_schema(self) -> None:
        with _conn() as conn:
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
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                CREATE INDEX IF NOT EXISTS idx_images_user_id ON images(user_id);
                """
            )

    def register_user(
        self, username: str, password: str, credits: int = 0, is_admin: bool = False
    ) -> bool:
        if len(password) < MIN_PASSWORD_LENGTH:
            raise DatabaseError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
        try:
            with _conn() as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, credits, is_admin) "
                    "VALUES (?, ?, ?, ?)",
                    (username, ph.hash(password), credits, int(is_admin)),
                )
            logger.info(
                "Registered user %r (credits=%d, admin=%s)",
                username,
                credits,
                is_admin,
            )
            return True
        except sqlite3.IntegrityError as e:
            logger.warning("Registration failed — username %r already exists", username)
            raise DatabaseError("Username already exists") from e

    def authenticate(self, username: str, password: str) -> bool:
        logger.debug("Authenticating user %r", username)
        with _conn() as conn:
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
        with _conn() as conn:
            row = conn.execute(
                "SELECT id, username, credits, is_admin, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if not row:
            return None
        created = row["created_at"]
        return UserOut(
            id=row["id"],
            username=row["username"],
            credits=row["credits"],
            is_admin=bool(row["is_admin"]),
            created_at=datetime.fromisoformat(created) if created else None,
        )

    def user_exists(self, username: str) -> bool:
        with _conn() as conn:
            row = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        return row is not None

    def get_credits(self, username: str) -> int:
        user = self.get_user(username)
        return user.credits if user else 0

    def deduct_credit(self, username: str) -> bool:
        with _conn() as conn:
            cur = conn.execute(
                "UPDATE users SET credits = credits - 1 "
                "WHERE username = ? AND credits > 0 AND is_active = 1",
                (username,),
            )
            ok = cur.rowcount > 0
        if ok:
            remaining = self.get_credits(username)
            logger.info("Deducted 1 credit from %r (remaining=%d)", username, remaining)
        else:
            logger.warning("Could not deduct credit from %r", username)
        return ok

    def set_credits(self, username: str, credits: int) -> bool:
        if credits < 0:
            logger.warning("set_credits rejected negative value for %r: %d", username, credits)
            return False
        with _conn() as conn:
            cur = conn.execute(
                "UPDATE users SET credits = ? WHERE username = ?", (credits, username)
            )
            ok = cur.rowcount > 0
        status = "ok" if ok else "user not found"
        logger.info("Set credits for %r → %d (%s)", username, credits, status)
        return ok

    def log_image(self, username: str, filename: str, prompt: str) -> None:
        with _conn() as conn:
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
        with _conn() as conn:
            rows = conn.execute(
                f"""
                SELECT i.id, i.filename, i.prompt, i.created_at
                FROM images i
                JOIN users u ON u.id = i.user_id
                WHERE u.username = ?
                ORDER BY i.created_at DESC
                LIMIT {HISTORY_DB_LIMIT}
                """,
                (username,),
            ).fetchall()
        history = [dict(r) for r in rows]
        logger.debug("Fetched %d history rows for %r", len(history), username)
        return history

    def list_users(self) -> list[UserOut]:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT id, username, credits, is_admin, created_at FROM users ORDER BY username"
            ).fetchall()
        users = [
            UserOut(
                id=r["id"],
                username=r["username"],
                credits=r["credits"],
                is_admin=bool(r["is_admin"]),
                created_at=datetime.fromisoformat(r["created_at"]) if r["created_at"] else None,
            )
            for r in rows
        ]
        logger.debug("Listed %d users", len(users))
        return users
