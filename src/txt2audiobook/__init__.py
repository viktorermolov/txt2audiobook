from importlib.metadata import version, PackageNotFoundError

try:  # pragma: no cover - at runtime when installed
    __version__ = version("txt2audiobook")
except PackageNotFoundError:  # pragma: no cover - local dev
    __version__ = "0.0.0"

__all__ = ["__version__"]
