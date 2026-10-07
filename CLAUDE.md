# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`CLAUDE.md` is the canonical engineering guide. `AGENTS.md` only points here.

## What this is

`txt2audiobook` is a persistent Raspberry Pi 4 (8 GB, ARM64, CPU-only) Docker
service. It accepts Russian text books from Telegram or `data/books/`, uses
local Silero TTS, assembles one M4B, and publishes it to Audiobookshelf (ABS).
It processes one book at a time and persists work for restart-safe recovery.

The code implements this workflow; do not state that end-to-end deployment or
public access has been completed without fresh Pi evidence. User-facing strings
and raised user-visible errors are intentionally Russian.

## Where it runs (read first)

The live deployment's SSH host, project directory and public URL are kept in
the untracked `CLAUDE.local.md` (gitignored; this repository is public). The
commands below write them as `$PI` and `$DIR`. The image is named exactly
`claude-audiobook:latest`. The local Mac lacks ARM
torch/ffmpeg, so local test runs error out — the Pi image is the reference. The
Pi has no git checkout: source is copied with `rsync`, and the deployed tree
should match the branch you think is deployed (compare SHA-256 of `app/`,
`tests/`, `scripts/`, `deploy/` before assuming).

```bash
# Lint (locally, ruff 0.7.x)
ruff check app scripts tests

# Ship source to the Pi. NEVER use --delete here: data/ holds state, library,
# voices, backups and secrets.
rsync -az --exclude data --exclude .git --exclude __pycache__ --exclude .ruff_cache \
  ./ "$PI:$DIR/"

# Unit tests in the existing image, BEFORE building or restarting services.
# app/ and tests/ are mounted over the image, so run rsync first.
ssh "$PI" "cd $DIR && \
  docker run --rm -v \$(pwd)/app:/app/app:ro -v \$(pwd)/tests:/app/tests:ro \
  claude-audiobook:latest python -m unittest discover -s tests -v"

# Single test: replace the discover command with e.g.
#   python -m unittest tests.test_workflow.WorkflowTests.test_ingest_marks_prepared_source_ingesting_before_queueing -v

# Build and start
ssh "$PI" "cd $DIR && docker compose build"
ssh "$PI" "cd $DIR && docker compose up -d"
```

Real Silero + ffmpeg smoke test with temporary state, no network and no writes
to the live library:

```bash
ssh "$PI" "cd $DIR && \
  docker run --rm --network none --cpus 2 --memory 3g \
  -v \$(pwd)/data/voices:/voice-cache:ro \
  claude-audiobook:latest python -m scripts.smoke_offline --voices /voice-cache"
```

`scripts/smoke_telegram.py` is a live end-to-end smoke (real Telegram + ABS) run
with the normal bot stopped; `scripts/verify_public.py` checks the public ABS
endpoint. Both touch real external services — run only when asked.

## Ingestion, metadata, and stages

Telegram documents may include one metadata field per caption line:
`Автор:`, `Название:`, `Серия:`, and `Номер:`. These values override extracted
metadata. FB2/EPUB may supply author/title/series/index; TXT must not acquire
invented series metadata.

Both Telegram and the folder watcher call `ingest.ingest_file`. It prepares a
copy at `work/<id>/source.*` before queueing and uses the transient
`INGESTING` stage. SHA-256 deduplicates active work. If a restart catches a job
in `INGESTING`, startup verifies the staged or promoted source against its
stored SHA-256 and completes the move before queueing. Missing or mismatched
sources fail explicitly rather than being converted.

Normal stages are `queued -> extracting -> cleaning -> synthesizing ->
assembling -> ready -> delivering -> done`, with `failed` and `cancelled` as
terminal alternatives. Conversion and publication remain separate: a failed
publication leaves the M4B and job in `ready`, then retries 120 seconds after
the previous attempt finishes. READY jobs are scheduled fairly; a slow failed
publication must not starve other ready books. Retries run in their own asyncio
task beside the conversion loop (`Worker._publication_retries`), so a
multi-hour synthesis never postpones them; `_publish_lock` serialises all
publishers. Once `publication.json` records an indexed `item_id`, a retry only
re-sends the Telegram notice and never re-stages files.

Startup reconciliation also deletes upload leftovers no job references
(`work/_staging/*` without a job row, all of `work/_incoming/*`). Staging copies
of failed ingests are referenced by their job and deliberately kept.

## Pipeline and cache rules

`worker.py` owns the blocking pipeline in one executor thread while aiogram
continues on the asyncio event loop. Intermediate job artifacts are
`source.*`, `cleaned.txt`, `meta.json`, `plan.json`, `audio/`, quality data, and
`publication.json` under `data/work/<id>/`.

`synth.py` stores a cache manifest that includes the complete plan and all
voice-affecting settings. A missing or nonmatching manifest invalidates old WAV
caches; never weaken this guard, because mixing voice settings or sample rates
corrupts a resumed M4B. Silero rejects any character outside `model.symbols`
(lowercase Cyrillic, `ё`, `_~|!+,-.:;?`, en-dash, ellipsis, space), so all text
is filtered to that set before `apply_tts`. After successful assembly, per-chunk
WAVs are removed to protect Pi disk space; publication retries use the retained
M4B.

The worker calls assembly with `max_part_mib=None`. Production output is one
M4B with no Telegram-derived output-size cap or multipart splitting.

Before assembly, optional EPUB/FB2 artwork is re-extracted from the retained
source into `work/<id>/cover.jpg`, normalized as a bounded, metadata-free JPEG.
The M4B embeds it as `attached_pic`; publication includes the same `cover.jpg`
in the atomic staging directory and validates its receipt on retries. Missing
or invalid artwork must not fail conversion. Never fetch external cover URLs.
Artwork limits are not source-text or audiobook-size limits.

## Audiobookshelf publication

`LibraryPublisher` stages publication in the library filesystem and atomically
renames it after writing a receipt. Its deterministic destination is:

```text
<author>/<series>/<index> - <title> [<sha256-prefix>-<job-id>]/
```

Without series it is `<author>/<title> [<sha256-prefix>-<job-id>]/`. It contains
numbered M4B files, `metadata.opf`, and `.txt2audiobook.json`. The receipt
validates content hash, job ID, filenames, sizes, and audio SHA-256 values on
retry. Do not overwrite a pre-existing invalid destination.

ABS discovers changes with its library watcher; a native scan cron provides a
two-minute fallback. The publisher does not send `POST` scan. It polls the API
until the destination is indexed with all audio files and nonzero duration.
Only then is the HTTPS public item URL sent to Telegram. `/publish [id]` starts
a manual repeat with an explicit outcome; `/resend [id]` is its alias. DONE jobs
never republish implicitly, even if their local receipt is lost. `/restore id`
requires a short-lived, one-time owner confirmation to republish a completed
book. `/library` returns the configured HTTPS public URL.

The native Telegram command menu is scoped to the owner's private chat. Both
message and callback handlers enforce that boundary. Confirmation tokens bind
the user, chat, job and action; re-check job state before executing them. Do not
change synthesis settings globally through the menu while a book is running.

Publication is retryable and idempotent at the library-path level. It is not an
exactly-once Telegram protocol: a crash between Telegram API acknowledgement and
the local marker write can cause a duplicate notification after recovery; do not
claim exactly-once delivery.

## Pi runtime and secrets

`docker-compose.yml` builds a hardened Audiobookshelf `2.37.0` derivative from
digest-pinned upstream and Node 24 images and an npm lockfile under
`deploy/audiobookshelf`. Since 2.37.0 upstream compiles the server from
TypeScript, so the entry point is `dist-server/index.js` (Dockerfile `CMD` and
the compose `command` must agree). To upgrade: pin the new multi-arch index
digest, take `version/main/bin/scripts/pkg` from upstream `package.json`, keep
our pinned patched runtime dependencies, compare upstream runtime lock trees
between versions, and bump the version checks in `Dockerfile` and `verify.cjs`.
Back up `data/abs/config` before starting a new version (DB migrations are
one-way). Native SQLite and nusqlite must be verified on ARM64
after runtime upgrades; dependency checks must resolve from their consumers,
not merely inspect top-level package versions. Do not run `npm audit fix
--force` without checking ABS behavior.

ABS is bound locally on `127.0.0.1:13378`; `data/library` MUST be writable by
ABS so web deletion removes files instead of resurrecting them on the next scan.
Completed bot jobs are not automatically republished after web deletion.
Both services run as UID/GID 1002 (overridable), with read-only root
filesystems, all capabilities dropped, and private writable data directories.
Cloudflared is pinned to `2026.9.1` behind the optional `tunnel` profile; the
public URL is `audiobookshelf.public_url` (see `CLAUDE.local.md`). Keep
`build.network: host`: Pi BuildKit needs it for DNS during Dockerfile `RUN`
steps.

Never commit deployment specifics (hostnames, domains, paths, usernames,
Telegram IDs): this repository is public. They belong in `CLAUDE.local.md`
or the Pi's `data/config/config.yaml`.

These private Pi-only files must never be committed, printed, copied to logs,
or exposed. Keep directories 0700 and files 0600:

- `data/config/config.yaml` — Telegram credentials and the ABS API token.
- `data/abs-admin/credentials.json` — ABS administrator credentials, outside
  the bot mount.
- `data/cloudflared/tunnel-token` — Cloudflare tunnel token, outside the
  converter's single-file read-only config mount.
- `data/abs/config/absdatabase.sqlite` — signing/session secrets; the same
  applies to its backups.

Keep backups under `data/backups`. Existing `data/audiobook/` files are not
imported into ABS; remove legacy outputs only with explicit user authorization
and a verified inventory, preserving source texts and the state backup.

`scripts/bootstrap_abs.py` performs one-time ABS initialization and token
repair. Run it from the built image with `data/abs-admin` mounted at `/admin`
and `data/config` at `/bootstrap-config` (writable only for this command), so
`credentials.json` never reaches the bot container. The publisher API user is
non-admin, library-scoped, and has no scan rights. Bootstrap must not print
generated secrets.

`scripts/harden_data.py` is an OFFLINE migration: stop project containers,
back up data, then set ownership/permissions and optionally rotate the ABS JWT
secret and sessions. Re-run bootstrap after rotation, keeping the admin
password. Never rotate authentication or permissions while ABS is running.

The Cloudflare request-header transform scoped to the library host
(`deploy/cloudflare-request-header-rule.json`) sets `x-client-ip` to `ip.src`,
preventing spoofed login rate-limit keys. Preserve it when changing
tunnel/proxy configuration. Silero downloads/cache require a reviewed SHA-256
allowlist entry before executable model deserialization. The process-lifetime
`service.lock` must cover recovery and executor shutdown. Do not introduce disk
reservation or M4B size caps without a separate decision.
