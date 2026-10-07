import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts.harden_data import main


class MaintenanceTests(unittest.TestCase):
    def test_offline_rotation_preserves_password_and_private_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            config = data / "config"
            config.mkdir()
            (config / "config.yaml").write_text("test configuration")
            (config / "cloudflare-tunnel-token").write_text("test tunnel token")
            database = data / "abs/config/absdatabase.sqlite"
            database.parent.mkdir(parents=True)
            with sqlite3.connect(database) as connection:
                connection.executescript("""
                    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
                    CREATE TABLE sessions (id TEXT);
                    CREATE TABLE apiKeys (name TEXT, isActive INTEGER);
                    CREATE TABLE users (pash TEXT);
                    INSERT INTO sessions VALUES ('old-session');
                    INSERT INTO apiKeys VALUES ('txt2audiobook-publisher', 1);
                    INSERT INTO apiKeys VALUES ('unrelated-key', 1);
                    INSERT INTO users VALUES ('existing-password-hash');
                """)
                connection.execute("INSERT INTO settings VALUES (?, ?)", (
                    "server-settings", json.dumps({"tokenSecret": "old-secret", "other": 7}),
                ))
            with patch("sys.argv", ["harden_data", "--data", str(data), "--uid", str(os.getuid()),
                                    "--gid", str(os.getgid()), "--rotate-auth"]):
                main()
            with sqlite3.connect(database) as connection:
                settings = json.loads(connection.execute("SELECT value FROM settings").fetchone()[0])
                self.assertNotEqual("old-secret", settings["tokenSecret"])
                self.assertEqual(7, settings["other"])
                self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
                self.assertEqual("existing-password-hash", connection.execute("SELECT pash FROM users").fetchone()[0])
                self.assertEqual(0, connection.execute("SELECT isActive FROM apiKeys WHERE name='txt2audiobook-publisher'").fetchone()[0])
                self.assertEqual(1, connection.execute("SELECT isActive FROM apiKeys WHERE name='unrelated-key'").fetchone()[0])
            backup = next((data / "backups").iterdir())
            with sqlite3.connect(backup / "abs/config/absdatabase.sqlite") as connection:
                self.assertEqual("old-secret", json.loads(connection.execute("SELECT value FROM settings").fetchone()[0])["tokenSecret"])
            self.assertEqual(0o700, database.parent.stat().st_mode & 0o777)
            self.assertEqual(0o600, database.stat().st_mode & 0o777)
            self.assertEqual(0o700, backup.stat().st_mode & 0o777)
            self.assertFalse((config / "cloudflare-tunnel-token").exists())
            self.assertEqual("test tunnel token", (data / "cloudflared/tunnel-token").read_text())


if __name__ == "__main__":
    unittest.main()
