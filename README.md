

# txt2audiobook

Конвертация .txt в русскую аудиокнигу (.mp3) на базе Coqui XTTS v2.

## Возможности

- Автоматическое разбиение по главам (маркер «Глава N») или по размеру
- Синтез речи XTTS v2, поддержка выбора голоса/языка
- Сборка глав в единый MP3 с ID3-тегами
- Удобный CLI и настраиваемые директории

## Требования

- Python 3.11
- FFmpeg (для экспорта MP3)

## Установка

### Через Poetry (рекомендуется для разработки)

```bash
poetry install
```

### Через pip (как пакет)

```bash
pip install .
```

## Быстрый старт

1) Положите .txt файлы в папку `txtbook/`.

2) Запустите:

```bash
poetry run txt2audiobook
# или, если установлено через pip
txt2audiobook
```

3) Готовые MP3 будут в `audiobook/`. Временные WAV глав — в `audio_chapters/`.

## CLI параметры

```bash
txt2audiobook [INPUT] [--out DIR] [--chapters DIR] [--tmp DIR] \
  [--model MODEL] [--speaker NAME] [--lang CODE] [--max-chars N] \
  [--delete-txt] [--keep-wavs]
```

- `INPUT`: папка с `.txt` (по умолчанию `txtbook`)
- `--out`: папка для MP3 (по умолчанию `audiobook`)
- `--chapters`: папка для WAV глав (по умолчанию `audio_chapters`)
- `--tmp`: временная папка для сегментов (по умолчанию `tmp`)
- `--model`: имя модели TTS (по умолчанию `tts_models/multilingual/multi-dataset/xtts_v2`)
- `--speaker`: имя голоса (по умолчанию `Viktor Menelaos`)
- `--lang`: язык (по умолчанию `ru`)
- `--max-chars`: максимум символов в сегменте (по умолчанию `180`)
- `--delete-txt`: удалять исходные `.txt` после успешной сборки
- `--keep-wavs`: не удалять WAV главы после сборки MP3

Те же параметры можно задавать через переменные окружения:

- `T2A_MODEL`, `T2A_SPEAKER`, `T2A_LANG`, `T2A_MAX_CHARS`

## Примечания

- Для работы необходим установленный FFmpeg.
- Модель XTTS v2 и Torch могут требовать специфичных версий CUDA при использовании GPU.
