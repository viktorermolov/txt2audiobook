"""Process-lifetime exclusion for the mutable service state."""

from __future__ import annotations

import fcntl
import os
import stat
from pathlib import Path


class ServiceAlreadyRunningError(RuntimeError):
    """Raised when another service instance owns the state directory."""


class ServiceLock:
    """An exclusive advisory lock that remains held until explicitly closed."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise ServiceAlreadyRunningError(
                "Небезопасная блокировка: система не поддерживает запрет символьных ссылок."
            )
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | nofollow, 0o600)
        except OSError as error:
            raise ServiceAlreadyRunningError(
                "Не удалось безопасно открыть файл блокировки состояния."
            ) from error
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ServiceAlreadyRunningError(
                    "Небезопасный файл блокировки: ожидается обычный файл."
                )
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ServiceAlreadyRunningError(
                "Сервис уже запущен: другой экземпляр удерживает блокировку состояния."
            ) from error
        except OSError as error:
            raise ServiceAlreadyRunningError(
                "Не удалось захватить блокировку состояния."
            ) from error
        except Exception:
            raise
        else:
            self._fd = fd
            return
        finally:
            if self._fd != fd:
                os.close(fd)

    def close(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "ServiceLock":
        self.acquire()
        return self

    def __exit__(self, *_args) -> None:
        self.close()
