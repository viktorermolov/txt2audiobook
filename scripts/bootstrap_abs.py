"""Run once on the Pi inside the converter image; never prints credentials."""

import argparse
import json
import os
import secrets
from pathlib import Path

import httpx
import yaml

from app.files import replace_and_fsync, write_json


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/bootstrap-config/config.yaml"))
    parser.add_argument("--admin-user", default="admin",
                        help="ABS administrator name, used only on first initialization")
    parser.add_argument("--public-url", help="HTTPS address sent in Telegram links")
    args = parser.parse_args()
    config_path = args.config
    credentials_path = Path("/admin/credentials.json")
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text())
    with httpx.Client(base_url="http://audiobookshelf:80", timeout=60) as client:
        status = client.get("/status")
        status.raise_for_status()
        if not credentials_path.exists():
            if status.json()["isInit"]:
                raise RuntimeError("Audiobookshelf уже настроен, файл доступа отсутствует; пароль не изменён")
            write_json(credentials_path, {
                "username": args.admin_user, "password": secrets.token_urlsafe(24),
            })
        credentials = json.loads(credentials_path.read_text())
        if not status.json()["isInit"]:
            response = client.post("/init", json={"newRoot": credentials})
            response.raise_for_status()
        response = client.post("/login", json=credentials, headers={"x-return-tokens": "true"})
        response.raise_for_status()
        user = response.json()["user"]
        client.headers["Authorization"] = "Bearer " + user["accessToken"]
        response = client.get("/api/libraries")
        response.raise_for_status()
        library = next((lib for lib in response.json()["libraries"] if lib["name"] == "Озвученные книги"), None)
        if library is None:
            response = client.post("/api/libraries", json={
                "name": "Озвученные книги", "mediaType": "book", "icon": "audiobooks",
                "folders": [{"fullPath": "/library"}],
            })
            response.raise_for_status()
            library = response.json()
        response = client.patch(f"/api/libraries/{library['id']}", json={
            "settings": {**library.get("settings", {}), "disableWatcher": False,
                         "autoScanCronExpression": "*/2 * * * *"},
        })
        response.raise_for_status()
        response = client.get("/api/users")
        response.raise_for_status()
        service = next((u for u in response.json()["users"] if u["username"] == "txt2audiobook"), None)
        permissions = {"accessAllLibraries": False, "delete": False, "download": False,
                       "update": False, "upload": False}
        if service is not None:
            response = client.patch(f"/api/users/{service['id']}", json={
                "type": "user", "permissions": permissions, "librariesAccessible": [library["id"]],
            })
            response.raise_for_status()
        abs_config = config.get("audiobookshelf", {})
        token_valid = False
        if abs_config.get("token") and service is not None:
            check = client.get("/api/me", headers={"Authorization": "Bearer " + abs_config["token"]})
            if check.status_code not in (200, 401, 403):
                check.raise_for_status()
            token_valid = check.status_code == 200 and check.json().get("id") == service["id"]
        if not token_valid:
            if service is None:
                response = client.post("/api/users", json={
                    "username": "txt2audiobook", "password": secrets.token_urlsafe(32),
                    "type": "user", "isActive": True,
                    "permissions": permissions,
                    "librariesAccessible": [library["id"]],
                })
                response.raise_for_status()
                payload = response.json()
                service = payload.get("user", payload)
            response = client.post("/api/api-keys", json={
                "name": "txt2audiobook-publisher", "userId": service["id"], "isActive": True,
            })
            response.raise_for_status()
            abs_config["token"] = response.json()["apiKey"]["apiKey"]
        abs_config.update(url="http://audiobookshelf:80", library_id=library["id"],
                          library_path="/library")
        if args.public_url:
            abs_config["public_url"] = args.public_url.rstrip("/")
        config["audiobookshelf"] = abs_config
        config.get("telegram", {}).pop("max_upload_mib", None)
        config.get("processing", {}).pop("max_part_mib", None)
        backup = config_path.with_name("config.before-audiobookshelf.yaml")
        if not backup.exists():
            backup.write_text(config_path.read_text())
        temporary = config_path.with_suffix(".yaml.tmp")
        temporary.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        replace_and_fsync(temporary, config_path)
        config_path.chmod(0o600)
        print("Audiobookshelf initialized; library:", library["id"], "; credentials stored privately on Pi")


if __name__ == "__main__":
    main()
