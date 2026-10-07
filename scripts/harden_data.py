"""Offline Pi migration. Stop all project containers before running as root."""

import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
from datetime import datetime, timezone


def protect_tree(root: Path, uid: int, gid: int) -> None:
    if root.is_symlink():
        raise RuntimeError(f"Ссылка вместо каталога: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for current, directories, files in os.walk(root, followlinks=False):
        paths = [Path(current), *(Path(current) / name for name in directories + files)]
        for path in paths:
            if path.is_symlink():
                continue
            os.chown(path, uid, gid, follow_symlinks=False)
            path.chmod(0o700 if path.is_dir() else 0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--uid", type=int, default=1002)
    parser.add_argument("--gid", type=int, default=1002)
    parser.add_argument("--rotate-auth", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    data = args.data.resolve(strict=True)
    database = data / "abs/config/absdatabase.sqlite"
    if not database.is_file() or database.is_symlink():
        raise RuntimeError("База Audiobookshelf не найдена или является ссылкой")
    backup = data / "backups" / datetime.now(timezone.utc).strftime("hardening-%Y%m%dT%H%M%SZ")
    backup.mkdir(parents=True)
    for name in ("config", "state", "abs/config"):
        source = data / name
        if source.exists():
            shutil.copytree(source, backup / name, symlinks=True)
    if args.rotate_auth:
        with sqlite3.connect(database) as connection:
            row = connection.execute("SELECT value FROM settings WHERE key = ?", ("server-settings",)).fetchone()
            if row is None:
                raise RuntimeError("Настройки сервера не найдены; секрет не изменён")
            settings = json.loads(row[0])
            if not settings.get("tokenSecret"):
                raise RuntimeError("Ключ подписи не найден; секрет не изменён")
            settings["tokenSecret"] = base64.b64encode(secrets.token_bytes(256)).decode("ascii")
            connection.execute("UPDATE settings SET value = ? WHERE key = ?", (json.dumps(settings), "server-settings"))
            connection.execute("DELETE FROM sessions")
            connection.execute("UPDATE apiKeys SET isActive = 0 WHERE name = ?", ("txt2audiobook-publisher",))
    token = data / "config/cloudflare-tunnel-token"
    target = data / "cloudflared/tunnel-token"
    target.parent.mkdir(exist_ok=True)
    if token.exists():
        if target.exists():
            raise RuntimeError("Два файла токена туннеля; проверьте их вручную")
        token.replace(target)
    for name in ("books", "audiobook", "state", "work", "voices", "config", "library", "abs", "abs-admin", "cloudflared", "backups"):
        protect_tree(data / name, args.uid, args.gid)
    print("Backup:", backup)
    print("Data permissions hardened; authentication rotated:", args.rotate_auth)


if __name__ == "__main__":
    main()
