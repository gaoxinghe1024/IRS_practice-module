# -*- coding: utf-8 -*-
"""
用户认证：注册 / 登录 / 会话 token。

课程项目演示级实现，答辩可逐行讲解：
- 密码用 PBKDF2-HMAC-SHA256 + 随机盐哈希后存储（不存明文，防拖库）；
- 会话 token 用 secrets 随机生成、存 sessions 表（简单直接）；
- demo / visitor 等种子用户无密码（password_hash 为空），只能走演示模式，不可登录。
- 上生产可平滑换成 JWT / OAuth，本模块对外函数签名不变。
"""
import hashlib
import secrets
import sqlite3
from datetime import datetime

from database import get_conn

_PBKDF2_ITER = 60000


def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(8)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITER)
    return f"{salt}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(_hash_password(password, salt), stored)


def register(username: str, password: str) -> dict:
    """注册新用户，成功返回 user_id + 会话 token。用户名冲突/参数不合法抛 ValueError。"""
    username = username.strip()
    if not (2 <= len(username) <= 20):
        raise ValueError("用户名长度需在 2-20 个字符之间")
    if len(password) < 6:
        raise ValueError("密码至少 6 位")
    conn = get_conn()
    try:
        if conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone():
            raise ValueError("用户名已被注册")
        cur = conn.execute(
            "INSERT INTO users (username, gender, mean_score, password_hash, created_at) VALUES (?,?,?,?,?)",
            (username, "Unknown", 0.0, _hash_password(password),
             datetime.now().isoformat(sep=" ", timespec="minutes")),
        )
        user_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    return {"user_id": user_id, "username": username, "token": _create_session(user_id)}


def login(username: str, password: str) -> dict:
    """登录校验，成功返回 user_id + 会话 token。失败抛 ValueError。"""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, username, password_hash FROM users WHERE username=?",
        (username.strip(),)).fetchone()
    conn.close()
    if not row or not row["password_hash"] or not _verify_password(password, row["password_hash"]):
        raise ValueError("用户名或密码错误")
    return {"user_id": row["id"], "username": row["username"], "token": _create_session(row["id"])}


def _create_session(user_id: int) -> str:
    token = secrets.token_hex(16)
    conn = get_conn()
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?,?,?)",
        (token, user_id, datetime.now().isoformat(sep=" ", timespec="minutes")))
    conn.commit()
    conn.close()
    return token


def user_from_token(token: str) -> dict | None:
    """token → 用户；无效返回 None。"""
    if not token:
        return None
    conn = get_conn()
    row = conn.execute(
        """SELECT u.id, u.username FROM sessions s JOIN users u ON u.id=s.user_id
           WHERE s.token=?""", (token,)).fetchone()
    conn.close()
    return dict(row) if row else None


def logout(token: str):
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()
