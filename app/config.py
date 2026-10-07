from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


# Validation errors must never echo raw input: a bad config would otherwise
# print most of the bot token into the container log.
_PRIVATE = ConfigDict(hide_input_in_errors=True)


class TelegramConfig(BaseModel):
    model_config = _PRIVATE

    bot_token: SecretStr
    allowed_user_id: int
    # HTTP timeout for Telegram API calls, seconds. Large uploads (tens of MB)
    # over a slow uplink need far more than the library default of 60 s.
    upload_timeout_sec: int = Field(default=600, ge=30, le=3600)

    @field_validator("bot_token")
    @classmethod
    def _token_set(cls, v: SecretStr) -> SecretStr:
        if v.get_secret_value() in {"", "PLACEHOLDER_BOT_TOKEN"}:
            raise ValueError(
                "telegram.bot_token не задан — впишите токен от @BotFather в config.yaml"
            )
        return v

    @field_validator("allowed_user_id")
    @classmethod
    def _user_set(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(
                "telegram.allowed_user_id не задан — впишите ваш Telegram user ID"
            )
        return v


class TTSConfig(BaseModel):
    # Legacy Piper keys (voice/length_scale) are silently ignored if present.
    # protected_namespaces=() lets us have a field literally named `model`.
    model_config = ConfigDict(extra="ignore", protected_namespaces=(), hide_input_in_errors=True)

    # Silero model package id (downloaded from models.silero.ai) and speaker.
    model: str = "v5_5_ru"
    speaker: str = "eugene"
    # Synthesis sample rate: Silero supports 8000 / 24000 / 48000.
    # 48000 measures the SAME speed as 24000 on the Pi (the vocoder is native
    # 48 kHz), so high quality is free — keep it.
    sample_rate: int = 48000

    @field_validator("sample_rate")
    @classmethod
    def _sample_rate(cls, value: int) -> int:
        if value not in {8000, 24000, 48000}:
            raise ValueError("Частота Silero должна быть 8000, 24000 или 48000 Гц")
        return value
    # Silero auto-places stress (accent) and ё — both improve Russian prosody.
    put_accent: bool = True
    put_yo: bool = True
    # Trailing pause appended after each chunk, seconds.
    sentence_silence: float = Field(default=0.4, ge=0, le=5)
    # Torch CPU threads. Measured on the Pi 4 under the 3.5-CPU compose quota:
    # 1 → 2.2x real time, 2 → 2.8x, 3 → 3.1x, 4 → 2.9x (quota contention).
    threads: int = Field(default=3, ge=1, le=4)


class PathsConfig(BaseModel):
    model_config = _PRIVATE

    books: Path = Path("/data/books")
    audiobook: Path = Path("/data/audiobook")
    state: Path = Path("/data/state")
    work: Path = Path("/data/work")
    voices: Path = Path("/data/voices")


class ProcessingConfig(BaseModel):
    model_config = _PRIVATE

    chunk_chars: int = Field(default=800, ge=100, le=800)
    audio_bitrate: str = "64k"
    audio_sample_rate: int = 48000
    synth_retries: int = Field(default=2, ge=0, le=5)
    # Playback tempo baked into the M4B (1.0 = narrator's natural pace).
    # Pitch-preserving (ffmpeg atempo). Most players can also do this live.
    tempo: float = Field(default=1.0, ge=0.5, le=2.0)


class AudiobookshelfConfig(BaseModel):
    model_config = _PRIVATE

    url: str = "http://audiobookshelf:80"
    public_url: str = "http://localhost:13378"
    token: SecretStr = SecretStr("")
    library_id: str = ""
    library_path: Path = Path("/library")
    scan_timeout_sec: int = Field(default=180, ge=5, le=900)
    retry_interval_sec: int = Field(default=120, ge=10, le=3600)

    @field_validator("url", "public_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        from urllib.parse import urlsplit
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Адрес Audiobookshelf должен быть HTTP(S) URL без пароля")
        return value.rstrip("/")


class AppConfig(BaseModel):
    model_config = _PRIVATE

    telegram: TelegramConfig
    tts: TTSConfig = Field(default_factory=TTSConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    audiobookshelf: AudiobookshelfConfig = Field(default_factory=AudiobookshelfConfig)

    def ensure_dirs(self) -> None:
        for p in (
            self.paths.books,
            self.paths.audiobook,
            self.paths.state,
            self.paths.work,
            self.paths.voices,
            self.audiobookshelf.library_path,
        ):
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    cfg_path = Path(path or os.environ.get("AUDIOBOOK_CONFIG", "/data/config/config.yaml"))
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Конфиг не найден: {cfg_path}. "
            "Скопируйте config.example.yaml в data/config/config.yaml и заполните."
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = AppConfig.model_validate(raw)
    cfg.ensure_dirs()
    return cfg
