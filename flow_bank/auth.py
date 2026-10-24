"""
auth.py — التحقق من كلمة المرور.

لا نخزّن كلمة المرور أبدًا. نخزّن «بصمتها» فقط: PBKDF2-SHA256 مع ملح عشوائي و200 000 تكرار.
عند الدخول نحسب بصمة ما كُتب ونقارنها بالمحفوظة بمقارنة آمنة زمنيًا (hmac.compare_digest).
"""
import hashlib
import hmac
import secrets

ITERATIONS = 200_000
REQUIRED_KEYS = {"username", "salt", "iterations", "hash"}


def make_record(username, password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return {"username": username.strip(), "salt": salt.hex(), "iterations": ITERATIONS, "hash": digest.hex()}


def valid_config(cfg):
    return isinstance(cfg, dict) and REQUIRED_KEYS <= set(cfg)


def verify(username, password, cfg):
    if not valid_config(cfg):
        return False
    try:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(cfg["salt"]),
                                     int(cfg["iterations"]))
    except (ValueError, TypeError):
        return False
    user_ok = hmac.compare_digest(username.strip().encode("utf-8"), str(cfg["username"]).encode("utf-8"))
    pass_ok = hmac.compare_digest(digest.hex(), str(cfg["hash"]))
    return user_ok and pass_ok


def to_toml(record):
    lines = ["[auth]"]
    for k, v in record.items():
        lines.append(f"{k} = {v}" if isinstance(v, int) else f'{k} = "{v}"')
    return "\n".join(lines) + "\n"
