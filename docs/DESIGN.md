# Design notes

This document records the engineering decisions behind each stage. The code is deliberately function-only (no classes) so each stage is a single, testable unit with explicit inputs and outputs.

## 1. Ingest and normalisation (`audio.py`)

**Normalise at the boundary.** Every STT model wants roughly the same input (16 kHz mono PCM). Converting once at ingest means transcription, chunking, and VAD never branch on format, and swapping the STT backend is a one-stage change.

**ffmpeg via pydub rather than per-format decoders.** One dependency covers WAV, MP3, M4A, FLAC, OGG, WebM, and video containers, and it handles the odd encodings (VBR MP3, 24-bit WAV) that hand-written decoders trip on. Adding a format is zero code.

**Detect by content, not extension.** `probe_audio_format` runs ffprobe before anything else. Extensions lie (`.wav` files that are MP3, `.mp3` that are AAC, Zoom recordings uploaded as "audio"). Non-media is rejected with a real reason before it reaches a worker.

**Fail fast and cheaply.** Existence, extension, size, and emptiness are checked before decoding; decoding 200 MB just to reject it is wasted compute. Duration bounds protect the STT stage from pathological inputs.

**Quality issues warn, they do not reject.** An 8 kHz phone recording or an AMR-encoded voicemail will transcribe worse but is still useful. Warnings ride along with the job so the review stage can weight confidence.

**Multi-channel is a routing decision.** Stereo call recordings often carry one speaker per channel. The probe surfaces channel count so a caller can transcribe per channel for free speaker separation instead of blindly downmixing.

**Not done by default:** loudness normalisation and noise reduction. Whisper-class models are robust to level, and aggressive denoising usually hurts accuracy. That belongs behind a flag with evaluation data.

## 2. Speech-to-text (`stt.py`)

**faster-whisper over openai-whisper or a hosted API.** Same weights, CTranslate2 runtime, roughly 4x faster on CPU with int8 and far lower memory. Fully offline, so no per-minute cost and no audio leaving the machine. The return shape is model-agnostic, so swapping to a hosted provider later touches one function.

**`small` as the default.** Best accuracy-per-second on CPU for English. `medium`/`large-v3` are a parameter change when accuracy matters more than latency.

**VAD filter on.** Whisper hallucinates plausible sentences over silence and music. Trimming non-speech before inference is the highest-leverage free quality fix.

**`condition_on_previous_text=False`.** The default improves coherence slightly but causes repetition loops on long recordings where one bad window poisons everything after it. For a batch pipeline, per-window independence is the safer trade.

**Return segments and confidence, not just a string.** Timestamps enable captions, search, and speaker attribution. `avg_logprob` and `no_speech_prob` let the pipeline route low-confidence output to review instead of silently passing bad text downstream.

**Model cache per process.** Loading `small` takes about 2 seconds and hundreds of MB. `get_model` caches by `(size, device)` so a warm worker pays once. The cache is also the seam tests use to inject a fake model.

## 3. Long files (`stt.py: transcribe_long_audio`)

Whisper works on 30-second windows internally, so feeding a 2-hour file whole does not improve accuracy; it just raises memory, latency, and the chance of drift. The pipeline owns chunking.

**Cut on silence, not on the clock.** A fixed cut lands mid-word nearly every time and loses that word on both sides. `plan_chunks` aims for a target length, snaps to the nearest pause within a tolerance window, and hard-cuts at the maximum when no silence exists (music beds, continuous speech).

**Overlap plus ownership, not fuzzy dedupe.** Each chunk gets a short trailing overlap so a sentence crossing the boundary is intact somewhere. Each chunk then owns the time range up to its hard cut; segments starting after that are dropped as belonging to the next chunk. Deterministic and language-independent.

**Fail per chunk.** A corrupt region at minute 47 should not discard 118 good minutes. Failed chunks are reported with their time range for retry; the job is dead-lettered, not failed.

**Bounded parallelism.** On CPU, two `small` instances saturate most machines; on GPU, batch size matters more than worker count. The independent-chunk shape maps directly onto a distributed worker pool.

**Same output shape as a short file.** The renderer and storage do not know chunking happened. Only `chunks` and `failed_chunks` are added.

**Language voted across chunks.** Per-chunk auto-detect misfires on quiet or noisy chunks. Majority vote is more robust than trusting chunk zero.

## 4. Output rendering (`formatting.py`)

**Rendering is separate from inference.** One canonical structure, projected into json/srt/vtt/txt on demand. A new consumer format never touches model code.

**Merge before rendering.** Whisper segments break on acoustic pauses, not sentence boundaries. Merging on a short gap with a hard length cap gives caption-sized blocks and coherent units for downstream NLP. Both thresholds are parameters because captions want ~7 s blocks and meeting notes tolerate 30 s.

**Worst-case confidence on merge.** Averaging hides a garbled fragment inside a good block; the minimum keeps the review signal honest.

**Integer millisecond timestamps.** Float formatting drifts and produces things like `00:00:59.1000`. Rounding once and using `divmod` is exact.

## 5. Storage (`storage.py`)

**Blobs in object storage, metadata in a database, joined by content hash.** Audio bytes never go in the database; the only copy of a transcript never lives in an unindexed file.

**Content-addressed, immutable audio keys.** `audio/<hh>/<sha256>.<ext>` deduplicates for free and makes keys safe to cache forever. The original upload is kept; the normalised WAV is a derived artefact deleted after the job.

**Segments in their own table.** Enables "jump to where X was said", time-range queries, and partial re-transcription. Full text also lives on the header row so the common case is one read.

**Rendered formats are not stored.** SRT/VTT are cheap to regenerate; storing them means three copies to keep consistent.

**Model version on every row, `UNIQUE (content_hash, model)`.** Upgrading the model produces a new row, not an overwrite. Re-running the same model replaces in place, so retries are idempotent.

**Delete removes everything.** Blob, header, and segments go together. Data-deletion requests cannot leave an index behind.

## 6. Concurrent uploads (`jobs.py`)

**Accept fast, process later.** The upload path validates, hashes, stores, probes, and enqueues; it never runs inference. The client gets a job ID and `202 Accepted`.

**Stream with a cap.** Bytes are written in 1 MB chunks with the size limit enforced mid-stream, so a 5 GB upload dies at 200 MB rather than after filling memory. The SHA-256 is computed on the same pass.

**Dedupe on content and on key.** Identical bytes plus same model return the existing job. An `Idempotency-Key` header handles the client-retry-before-upload-finished case.

**Bounded queue with backpressure.** A full queue returns `503` with `Retry-After` rather than accepting work that cannot be finished. Workers are a fixed pool sized to hardware.

**Per-job isolation.** Each upload gets its own directory, removed on completion, failure, or dedupe.

**Never raise out of a worker.** `process_job` records failure on the job and returns, so one bad file cannot stop the queue draining.

## 7. Retry and recovery (`retry.py`)

**Classify first.** Input problems (`ValueError`, `FileNotFoundError`) are permanent; retrying is wasted compute. Everything else is transient and gets backoff.

**Retry the chunk, not the file.** Failed chunks carry their time range, so recovery re-exports only those ranges from the stored original. A 2-hour file with one bad chunk costs ten minutes to recover.

**Escalate configuration.** Attempt 1 repeats; attempt 2 pins language; attempt 3 steps up one model size. Each attempt is recorded so the final result is explainable.

**Exponential backoff with jitter.** Stops a burst of failures from retrying in lockstep. `sleep` is injectable so tests do not wait and production can swap in a delayed re-enqueue.

**Dead-letter with history.** After the budget is spent, the chunk is kept with every attempt and error. The job publishes partial results with gap markers and `needs_review` rather than failing outright; a 95% transcript is usually more useful than none.

**Resume from history.** If a chunk already has attempts recorded, retry continues from the next attempt number rather than starting over.

## 8. API (`api.py`)

**Asynchronous job API.** Transcription takes seconds to minutes; synchronous request/response hits load-balancer timeouts. Submit → poll (or webhook in production) → fetch.

**Distinguish not-found from not-ready.** `404` for unknown jobs, `409` with `Retry-After` while queued or processing, `422` when the job failed. Clients can branch without parsing messages.

**Render on demand.** `/transcript?format=` calls the formatter against stored segments; nothing is pre-rendered.

**Request IDs on every response.** Correlates client logs with server logs.

**Health vs readiness.** `/healthz` is liveness; `/readyz` reports queue depth, capacity, and worker count so an orchestrator can stop routing to a saturated instance.

**What is intentionally left for production:** authentication and tenant scoping, presigned direct-to-storage uploads, webhooks with HMAC signatures, per-tenant rate limits, and metrics export. The hooks for each exist (tenant on the hash key, `Retry-After` on backpressure, request IDs), but the mock-data scope does not need the infrastructure.
