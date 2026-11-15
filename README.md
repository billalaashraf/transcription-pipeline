# transcription-pipeline

Audio in, structured text out. A small, production-shaped speech-to-text pipeline built from plain functions (no classes): ingest any audio or video file, normalise it, transcribe it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), handle long recordings by chunking, persist results, retry failures per chunk, and serve the whole thing over an async job API.

The focus is on engineering decisions, not model training. Every stage is one function with one job.

```
 audio/video ──► probe ──► normalise ──► transcribe ──► format ──► store
  (any fmt)    ffprobe    16k mono WAV   (chunked when   json/srt   blob + SQL
                                           long)          vtt/txt
```

## Design decisions

The pipeline is a chain of independent stages with one canonical data shape between them. Each decision below is tied to the stage it lives in; the longer discussion is in [`docs/DESIGN.md`](docs/DESIGN.md).

### Architecture

| Decision | Why |
|---|---|
| **One function per stage, no classes** | Each stage has explicit inputs and outputs and can be tested, replaced, or moved to another process on its own. There is no hidden state to reason about. |
| **One canonical intermediate: 16 kHz mono 16-bit PCM WAV** | Every downstream stage sees exactly one audio form. Format handling is confined to the ingest boundary, so adding a format or swapping the STT engine is a one-stage change. |
| **One canonical result shape** | `transcribe_audio`, `transcribe_long_audio`, and `retry_failed_transcription` all return the same dict (`text`, `segments[]`, `language`, `mean_logprob`, `needs_review`, …). Rendering, storage, and the API never branch on where a result came from. |
| **Confidence travels with the data** | Every segment carries `avg_logprob` and `no_speech_prob`; every result carries `needs_review`. Low-quality output is routed to review instead of silently passing downstream. |
| **Offline by default** | faster-whisper runs locally: no per-minute cost, no audio leaving the machine. A hosted provider is a one-function swap because the result shape is model-agnostic. |
| **In-memory queue and registry as deliberate stand-ins** | The project runs with zero infrastructure. `JOB_QUEUE` → Redis/SQS and `JOBS` → a database table are the only production changes in `jobs.py`; the function bodies stay the same. |

### Ingest and format handling (`audio.py`)

- **Detect by content, not extension.** `ffprobe` reads container and stream headers. A `.wav` that is really MP3 is handled; a text file named `.wav` is rejected before it reaches a worker.
- **One decode path via ffmpeg.** WAV, MP3, M4A/AAC, FLAC, OGG/Opus, WebM, and video containers (MP4, MKV, MOV) all go through pydub + ffmpeg. There is no per-format code to maintain.
- **Fail fast and cheaply.** Existence, extension, size, and emptiness are checked before decoding; duration bounds are checked right after. Decoding 200 MB just to reject it is wasted compute.
- **Quality problems warn, they do not reject.** 8 kHz telephony audio, AMR/GSM codecs, and low bitrates produce warnings that ride along with the job. The file still transcribes; the review stage can weight confidence.
- **Multi-channel is surfaced, not hidden.** Stereo call recordings often carry one speaker per channel. The probe reports channel count so a caller can transcribe per channel instead of blindly downmixing.
- **Video is audio with baggage.** The audio track is extracted; the video stream is ignored but flagged so logs and cost estimates reflect the real job size.
- **No loudness normalisation or denoising by default.** Whisper-class models are robust to level, and aggressive denoising usually lowers accuracy. That belongs behind a flag with evaluation data.

### Speech-to-text (`stt.py`)

- **faster-whisper over openai-whisper.** Same weights, CTranslate2 runtime, roughly 4x faster on CPU with int8 quantisation and far lower memory.
- **`small` as the default size.** Best accuracy-per-second on CPU for English. `medium`/`large-v3` are a parameter change.
- **VAD filter on.** Whisper hallucinates plausible sentences over silence and music. Trimming non-speech before inference is the highest-leverage free quality fix.
- **`condition_on_previous_text=False`.** The default improves coherence slightly but causes repetition loops on long recordings where one bad window poisons everything after it.
- **Model cached per process.** Loading `small` costs ~2 s and hundreds of MB. A warm worker pays once. The cache is also the seam tests use to inject a fake model.
- **The lazy segment iterator is materialised inside the function.** Inference happens on iteration; forcing it here means errors and latency are attributed to the STT stage, not to whatever consumes the result.

### Long files (`stt.py: transcribe_long_audio`)

- **The pipeline owns chunking.** Whisper works on 30 s windows internally; feeding it a 2-hour file whole only raises memory, latency, and drift.
- **Cut on silence, never on the clock.** `plan_chunks` aims for a target length, snaps to the nearest pause within a tolerance window, and hard-cuts at the maximum only when no silence exists (music beds, continuous speech). A fixed cut lands mid-word nearly every time.
- **Overlap plus an ownership rule, not fuzzy dedupe.** Each chunk carries a short trailing overlap so a boundary sentence is intact somewhere, then owns the time range up to its hard cut. Segments starting after that belong to the next chunk. Deterministic and language-independent.
- **Fail per chunk.** A corrupt region at minute 47 does not discard 118 good minutes. Failed chunks are reported with their time range; the job is dead-lettered with partial results, not failed.
- **Bounded parallelism.** On CPU two `small` instances saturate most machines; on GPU batch size matters more. The independent-chunk shape maps directly onto a distributed worker pool.
- **Language is voted across chunks.** Per-chunk auto-detect misfires on quiet or noisy chunks. Majority vote beats trusting chunk zero.

### Output rendering (`formatting.py`)

- **Rendering is separate from inference.** One structure, projected on demand into JSON, SRT, VTT, or TXT. A new consumer format never touches model code, and nothing is re-transcribed to re-render.
- **Merge fragments before rendering.** Whisper breaks on acoustic pauses, not sentence boundaries. Merging on a short gap with a hard length cap yields caption-sized blocks and coherent units for downstream NLP. Both thresholds are parameters.
- **Worst-case confidence on merge.** Averaging hides a garbled fragment inside a good block; the minimum keeps the review signal honest.
- **Integer millisecond timestamps.** Float formatting drifts (`00:00:59.1000`); rounding once and using `divmod` is exact.
- **Standard formats only.** SRT and VTT are what players and caption editors ingest. Rendered files are not stored because they are cheap to regenerate and would be a third copy to keep consistent.

### Storage (`storage.py`)

- **Blobs in object storage, metadata in a database, joined by content hash.** Audio bytes never go in the database; the only copy of a transcript never lives in an unindexed file.
- **Content-addressed, immutable audio keys** (`audio/<hh>/<sha256>.<ext>`). Deduplicates for free and is safe to cache forever. The original upload is kept; the normalised WAV is a derived artefact deleted after the job.
- **Segments in their own table.** Enables time-range queries ("jump to where X was said") and partial re-transcription. Full text also lives on the header row so the common read is one query.
- **`UNIQUE (content_hash, model)`.** Upgrading the model produces a new row, not an overwrite; re-running the same model replaces in place, so retries are idempotent.
- **Delete removes everything together.** Blob, header row, and segments. Data-deletion requests cannot leave an index behind.
- **SQLite and a local folder stand in for Postgres and S3.** The schema is standard SQL; the blob write is a single `copyfile`. Both swaps are configuration-level changes.

### Concurrent uploads (`jobs.py`)

- **Accept fast, process later.** The upload path validates, hashes, stores, probes, and enqueues. It never runs inference. The client gets a job ID and `202 Accepted`.
- **Stream with a cap.** Bytes are written in 1 MB chunks with the size limit enforced mid-stream, so a 5 GB upload dies at 200 MB rather than after filling memory. SHA-256 is computed on the same pass.
- **Dedupe on content and on key.** Identical bytes plus the same model return the existing job. An `Idempotency-Key` header covers the client-retried-before-upload-finished case.
- **Bounded queue with backpressure.** A full queue returns `503` with `Retry-After` instead of accepting work that cannot be finished. Workers are a fixed pool sized to hardware.
- **Per-job isolation.** Each upload gets its own directory, removed on completion, failure, or dedupe.
- **Workers never raise.** `process_job` records failure on the job and returns, so one bad file cannot stop the queue draining.

### Retry and recovery (`retry.py`)

- **Classify before retrying.** Input problems (`ValueError`, `FileNotFoundError`) are permanent and stop immediately. Everything else is transient and gets backoff.
- **Retry the chunk, not the file.** Failed chunks carry their time range, so recovery re-exports only those ranges from the stored original. A 2-hour file with one bad chunk costs ten minutes to recover.
- **Escalate configuration per attempt.** Repeat → pin language → step up one model size. Every attempt is recorded with its config and error so the final result is explainable.
- **Exponential backoff with jitter.** Stops a burst of failures from retrying in lockstep. `sleep` is injectable: tests do not wait, and production can swap in a delayed re-enqueue.
- **Dead-letter with history, publish partial results.** After the budget is spent the chunk is kept with its full attempt log, and the job ships with gap markers and `needs_review`. A 95% transcript beats none.
- **Resume from history.** A chunk with recorded attempts continues from the next attempt number instead of starting over.

### API (`api.py`)

- **Asynchronous job API.** Transcription takes seconds to minutes; synchronous request/response hits load-balancer timeouts. Submit → poll → fetch.
- **Not-found, not-ready, and failed are distinct.** `404` unknown job, `409` + `Retry-After` while queued or processing, `422` when the job failed, `503` + `Retry-After` when the queue is full. Clients branch on status codes, not message text.
- **Render on demand.** `/transcript?format=` runs the formatter against stored segments. Nothing is pre-rendered.
- **Request ID on every response.** Correlates client and server logs.
- **Liveness and readiness are separate.** `/healthz` says the process is up; `/readyz` reports queue depth, capacity, and worker count so an orchestrator can stop routing to a saturated instance.
- **Intentionally left for production:** authentication and tenant scoping, presigned direct-to-storage uploads, HMAC-signed webhooks, per-tenant rate limits, metrics export. The hooks exist (tenant on the hash key, `Retry-After` on backpressure, request IDs); the mock-data scope does not need the infrastructure.

### Testing

- **The STT model is replaced by a deterministic fake.** It emits one segment every 2 s with predictable text, so stitching, offsets, and dedupe are checkable to the millisecond. The suite runs in seconds with no weight download or GPU.
- **Audio fixtures are synthesised** with pydub as tone bursts separated by silence, so the chunker has real pauses to find and cut placement can be asserted.
- **Failures are injected by filename.** The fake raises on paths matching a token, which lets tests target a specific chunk and verify isolation, dead-lettering, and recovery.
- **ffmpeg is the only system dependency**, and tests that need it skip cleanly when it is absent.

## Features

- Accepts WAV, MP3, M4A/AAC, FLAC, OGG/Opus, WebM, and video containers (MP4, MKV, MOV). One decode path via ffmpeg.
- Content-based format detection (ffprobe), not extension trust. Mislabelled files are handled; non-media is rejected early.
- Normalises everything to 16 kHz mono 16-bit PCM once at the boundary.
- Transcription with per-segment timestamps, confidence (`avg_logprob`), language detection, and a `needs_review` flag.
- Long files are split on silence (never mid-word), transcribed in parallel, and stitched back into one timeline. Chunk failures are isolated, not fatal.
- Output as JSON, SRT, VTT, or timestamped TXT, with fragment merging for readable blocks.
- Content-addressed blob storage for audio; SQLite (Postgres-shaped) for transcripts and segments, idempotent on `(content_hash, model)`.
- Upload handling with streaming size cap, SHA-256 dedupe, idempotency keys, bounded queue with backpressure, and a worker pool.
- Retry with failure classification (transient vs permanent), exponential backoff with jitter, model escalation, and dead-lettering with full attempt history.
- FastAPI service: `202 Accepted` job submission, status polling, rendered transcript, time-range segment queries, retry, delete, health/readiness.
- 90+ tests that run without downloading any model weights.

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `sudo apt-get install ffmpeg`
  - Windows: `winget install Gyan.FFmpeg` or <https://ffmpeg.org/download.html>

## Install

```bash
git clone https://github.com/<you>/transcription-pipeline.git
cd transcription-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The first transcription downloads the Whisper weights for the chosen model size (~460 MB for `small`). They are cached by faster-whisper under `~/.cache/huggingface`.

## Usage

### Command line

```bash
# JSON to stdout
transcribe meeting.mp3

# SRT captions to a file, medium model, English pinned
transcribe meeting.mp3 --format srt --model medium --language en -o meeting.srt

# Just inspect the file
transcribe recording.m4a --probe-only

# Transcribe and persist to the local store
transcribe lecture.wav --store
```

Or without installing the entry point: `python -m transcription_pipeline.cli meeting.mp3`.

### HTTP API

```bash
transcribe-api                      # or: uvicorn transcription_pipeline.api:app --port 8000
```

```bash
# Submit (returns 202 + job id)
curl -s -F "file=@meeting.mp3" "http://localhost:8000/v1/transcriptions?language=en" | jq
# {
#   "job_id": "7c3a...",
#   "status": "queued",
#   "status_url": "/v1/transcriptions/7c3a...",
#   ...
# }

# Poll
curl -s http://localhost:8000/v1/transcriptions/7c3a... | jq .status

# Fetch the transcript once done
curl -s "http://localhost:8000/v1/transcriptions/7c3a.../transcript?format=srt"

# Segments between 2:00 and 3:00
curl -s "http://localhost:8000/v1/transcriptions/7c3a.../segments?start=120&end=180" | jq

# Retry a dead-lettered or failed job
curl -s -X POST http://localhost:8000/v1/transcriptions/7c3a.../retry | jq

# Delete audio + transcript + segments
curl -s -X DELETE http://localhost:8000/v1/transcriptions/7c3a...
```

Interactive docs at <http://localhost:8000/docs>.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/transcriptions` | Submit a file (`multipart/form-data`). Query: `language`, `model`. Header: `Idempotency-Key`. Returns `202`. |
| `GET` | `/v1/transcriptions` | List jobs, optional `?status=`. |
| `GET` | `/v1/transcriptions/{id}` | Job status, confidence, warnings, failed chunks. |
| `GET` | `/v1/transcriptions/{id}/transcript` | `?format=json\|srt\|vtt\|txt`. `409` while processing, `422` if failed. |
| `GET` | `/v1/transcriptions/{id}/segments` | `?start=&end=` time-range query. |
| `POST` | `/v1/transcriptions/{id}/retry` | Re-queue a failed job or recover a dead-lettered one chunk by chunk. |
| `DELETE` | `/v1/transcriptions/{id}` | Remove everything for the job. |
| `GET` | `/healthz`, `/readyz` | Liveness; readiness with queue depth and worker count. |

Status codes: `400` bad input, `404` unknown job, `409` not ready / nothing to retry, `422` transcription failed, `503` queue full (with `Retry-After`).

### As a library

```python
from transcription_pipeline import load_audio, transcribe_audio, format_timestamped_transcript

meta = load_audio("interview.m4a")
result = transcribe_audio(meta["normalised_path"], model_size="small")
print(format_timestamped_transcript(result, fmt="vtt"))
```

## Configuration

All settings are environment variables with the `TP_` prefix.

| Variable | Default | Meaning |
|---|---|---|
| `TP_MODEL_SIZE` | `small` | faster-whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `TP_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `TP_MAX_FILE_BYTES` | `209715200` | Upload size cap (200 MB) |
| `TP_MAX_DURATION_SEC` | `10800` | Duration cap (3 h) |
| `TP_LONG_THRESHOLD_SEC` | `900` | Files longer than this are chunked |
| `TP_CHUNK_TARGET_SEC` | `600` | Preferred chunk length |
| `TP_CHUNK_OVERLAP_SEC` | `2` | Trailing overlap per chunk |
| `TP_CHUNK_WORKERS` | `2` | Parallel chunk transcriptions |
| `TP_WORKER_COUNT` | `1` | Background job workers in the API process |
| `TP_QUEUE_CAPACITY` | `100` | Jobs accepted before `503` |
| `TP_RETRY_MAX_ATTEMPTS` | `3` | Per-chunk retry budget |
| `TP_RETRY_BASE_DELAY_SEC` | `30` | First backoff delay |
| `TP_RETRY_MAX_DELAY_SEC` | `600` | Backoff ceiling |
| `TP_DATA_DIR` | `./data` | Root for uploads, blobs, and the database |
| `TP_HOST` / `TP_PORT` | `0.0.0.0` / `8000` | API bind address |

## Docker

```bash
docker build -t transcription-pipeline .
docker run --rm -p 8000:8000 -v $(pwd)/data:/app/data transcription-pipeline
```

The image bundles ffmpeg. Model weights download on first use; mount `~/.cache/huggingface` to `/root/.cache/huggingface` to persist them across containers.

## Development

```bash
make test        # pytest with coverage
make lint        # ruff check + format check
make fmt         # ruff format
make run-api     # uvicorn with reload
```

Tests replace the Whisper model with a deterministic fake (see `tests/conftest.py`), so the suite runs in seconds and needs no GPU or network. Audio fixtures are synthesised with pydub: tone bursts separated by silence, so the chunker has real pauses to find.

## Project layout

```
src/transcription_pipeline/
  audio.py        probe_audio_format, load_audio
  stt.py          transcribe_audio, transcribe_long_audio, plan_chunks, model cache
  formatting.py   format_timestamped_transcript, merge_segments
  storage.py      store/load/query/delete transcription, content-addressed blobs
  jobs.py         submit_upload, process_job, worker pool, in-memory queue
  retry.py        retry_failed_transcription, classify_error, escalation_plan
  api.py          create_api (FastAPI), serve
  cli.py          transcribe entry point
  config.py       environment-driven settings
tests/            one file per module, plus conftest with fakes and fixtures
docs/DESIGN.md    the engineering decisions behind each stage
```

## Production notes

The in-memory queue and job registry are deliberate stand-ins so the project runs with zero infrastructure. To deploy for real:

- Replace `JOB_QUEUE` with Redis/SQS/RabbitMQ and `JOBS` with the `transcripts` table (plus a `jobs` table). The function bodies in `jobs.py` do not otherwise change.
- Run the API and workers as separate processes so they scale independently. Workers hold one warm model each.
- Swap the local blob folder for S3/GCS (`shutil.copyfile` → `put_object`) and SQLite for Postgres (connection string only; the SQL is standard).
- Add authentication, per-tenant rate limits, and a tenant column on every row and blob key.
- Mount a persistent volume or shared cache for model weights.

## License

MIT. See [LICENSE](LICENSE).