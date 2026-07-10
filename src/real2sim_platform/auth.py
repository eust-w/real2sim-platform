from __future__ import annotations

import hashlib
import re
import secrets
import time
from pathlib import Path
from typing import Any

from .config import PlatformConfig
from .store import PlatformStore


SESSION_COOKIE = "real2sim_session"
VALID_ROLES = {"admin", "engineer", "viewer"}
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{2,63}$")


class AuthenticationError(ValueError):
    pass


class AuthManager:
    def __init__(self, store: PlatformStore, config: PlatformConfig) -> None:
        self.store = store
        self.config = config
        self._hasher = _password_hasher() if config.auth_enabled else None

    def bootstrap_admin(self) -> dict[str, Any] | None:
        if not self.config.auth_enabled or self.store.count_users() > 0:
            return None
        password = self.config.bootstrap_admin_password or secrets.token_urlsafe(18)
        validate_password(password)
        user = self.store.create_user(
            username=self.config.bootstrap_admin_username,
            password_hash=self.hash_password(password),
            role="admin",
            must_change_password=True,
        )
        if not self.config.bootstrap_admin_password:
            _write_bootstrap_file(self.config.auth_bootstrap_path, user["username"], password)
        return user

    def hash_password(self, password: str) -> str:
        validate_password(password)
        if self._hasher is None:
            raise RuntimeError("authentication is disabled")
        return str(self._hasher.hash(password))

    def verify_password(self, password_hash: str, password: str) -> bool:
        if self._hasher is None:
            return False
        try:
            return bool(self._hasher.verify(password_hash, password))
        except Exception:
            return False

    def login(self, username: str, password: str) -> tuple[dict[str, Any], str]:
        user = self.store.get_user_by_username(username.strip())
        if not user or user["disabled"] or not self.verify_password(str(user["password_hash"]), password):
            raise AuthenticationError("用户名或密码错误")
        token = secrets.token_urlsafe(48)
        self.store.create_session(
            token_hash=token_digest(token),
            user_id=user["id"],
            expires_at=time.time() + self.config.session_ttl_seconds,
        )
        self.store.update_user(user["id"], last_login=time.time())
        return self.store.get_user(user["id"]), token

    def authenticate(self, token: str | None) -> dict[str, Any] | None:
        if not self.config.auth_enabled:
            return {
                "id": "local-development",
                "username": "local",
                "role": "admin",
                "disabled": False,
                "must_change_password": False,
            }
        if not token:
            return None
        return self.store.get_session_user(token_digest(token))

    def logout(self, token: str | None) -> None:
        if token:
            self.store.delete_session(token_digest(token))

    def create_user(self, *, username: str, password: str, role: str) -> dict[str, Any]:
        validate_username(username)
        validate_role(role)
        return self.store.create_user(
            username=username.strip(),
            password_hash=self.hash_password(password),
            role=role,
            must_change_password=True,
        )

    def change_password(self, user_id: str, *, current_password: str | None, new_password: str, admin_reset: bool = False) -> dict[str, Any]:
        user_with_hash = self.store.get_user_by_username(self.store.get_user(user_id)["username"])
        if not user_with_hash:
            raise KeyError(user_id)
        if not admin_reset and not self.verify_password(str(user_with_hash["password_hash"]), current_password or ""):
            raise AuthenticationError("当前密码不正确")
        password_hash = self.hash_password(new_password)
        self.store.delete_user_sessions(user_id)
        return self.store.update_user(user_id, password_hash=password_hash, must_change_password=admin_reset)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_username(username: str) -> None:
    if not _USERNAME_RE.fullmatch(username.strip()):
        raise ValueError("用户名需为 3-64 位字母、数字、点、下划线或连字符")


def validate_role(role: str) -> None:
    if role not in VALID_ROLES:
        raise ValueError("role must be admin, engineer, or viewer")


def validate_password(password: str) -> None:
    if len(password) < 12:
        raise ValueError("密码至少需要 12 个字符")
    classes = [any(c.islower() for c in password), any(c.isupper() for c in password), any(c.isdigit() for c in password), any(not c.isalnum() for c in password)]
    if sum(classes) < 3:
        raise ValueError("密码需包含大小写字母、数字或符号中的至少三类")


def _password_hasher():
    try:
        from argon2 import PasswordHasher
        from argon2.low_level import Type
    except ImportError as exc:
        raise RuntimeError("auth_enabled requires argon2-cffi") from exc
    return PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2, hash_len=32, salt_len=16, type=Type.ID)


def _write_bootstrap_file(path: Path, username: str, password: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"username={username}\npassword={password}\nforce_password_change=true\n", encoding="utf-8")
    path.chmod(0o600)
