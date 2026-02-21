# CLAUDE.md — PhotoAIdent

## Project Overview

PhotoAIdent is a local, privacy-first desktop application for AI-powered face
recognition and photo search. It scans a local photo library, detects and embeds
faces using InsightFace/ArcFace, and provides a PySide6 desktop UI where the user
progressively labels faces — assigning them to known persons or marking them as
anonymous. Over time the app learns who is in the collection and surfaces match
suggestions automatically.

No cloud, no external API calls, no data leaves the machine.

---

## ⚠️ CRITICAL: Image Collection Is Read-Only

**The app must NEVER write to, modify, move, rename, or delete any file in the
user's image collection.**

This is an absolute, non-negotiable constraint. The original photos are the user's
irreplaceable personal archive spanning decades.

- All output (databases, thumbnails, caches) goes to XDG directories (see below)
- The indexer opens image files in read-only mode only
- No file watcher, sync tool, or any subsystem touches image files
- Any coding agent working on this project must never write code that performs
  write operations on paths outside of `~/.local/share/photoaident/` and
  `~/.cache/photoaident/`
- If in doubt: **do not touch the image collection**

---

## Agent Instructions

Before marking any task or phase as complete:

1. Lint the code using `uv run ruff check --fix` and fix any reported issues
2. Format the code consistently using `uv run black`
3. Ensure proper type checking using `uv run ty check` and fix any reported issues
4. Run `uv run scripts/verify.py` and ensure it exits with code 0
5. Fix any issues reported before declaring the work done
6. Never skip verification even if the changes appear trivial

---

## Scale Characteristics

The target collection is ~80,000 JPEG images spanning ~30 years. Key implications:

- **Estimated face vectors:** ~240,000 (assuming ~3 faces/image average)
- **FAISS index RAM usage:** ~500 MB (240k × 512 dims × 4 bytes)
- **Face crop cache:** ~2.4 GB (240k crops × ~10 KB average)
- **SQLite database:** comfortably small at this row count — no performance concern

`IndexFlatIP` (exact search) is appropriate at this scale. Search over 240k vectors
takes single-digit milliseconds on CPU. An approximate index (IVF/HNSW) is not
needed and would only add complexity. Reassess if the collection grows beyond ~5M
vectors.

The 30-year span means a person's appearance changes dramatically. A single mean
embedding per person will not work — see the **Embedding Clusters** section.

---

## XDG-Compliant File Locations

PhotoAIdent follows the XDG Base Directory Specification for all runtime data.
Nothing is written next to the binary or in the image collection.

| Content                             | Path                                           |
|-------------------------------------|------------------------------------------------|
| User config (settings, preferences) | `~/.config/photoaident/config.toml`            |
| SQLite database                     | `~/.local/share/photoaident/db/photoaident.db` |
| FAISS index                         | `~/.local/share/photoaident/db/faiss.index`    |
| Face crop thumbnails                | `~/.cache/photoaident/faces/<face_id>.jpg`     |
| Photo thumbnails                    | `~/.cache/photoaident/thumbs/<image_hash>.jpg` |

`~/.local/share/` is for data that should persist across reboots and is worth
backing up. `~/.cache/` is for data that can be regenerated — thumbnails and face
crops can be deleted and rebuilt by re-indexing.

In code, always resolve these paths via a central `AppPaths` helper class rather
than constructing them inline, so they can be overridden cleanly in tests.

---

## Core User Workflow

1. **Index** — user points the app at a folder. The app scans all images
   read-only, detects faces, computes embeddings, and stores everything in the
   local database. All faces start as unidentified.

2. **Label** — the app presents unidentified face crops to the user. The user
   assigns each face to a known person, creates a new person, or marks it as
   "anonymous" (a stranger the user doesn't want to track).

3. **Propagate** — after a face is labelled, the suggestion engine finds similar
   unidentified faces and queues them for fast confirm/reject review. Confirmed
   suggestions grow the person's embedding cluster.

4. **Search** — filter the photo library by person, date range, GPS area, and
   eventually scene tags.

---

## Embedding Clusters (Multi-Era Persons)

The 30-year span means a single mean embedding per person fails at the appearance
extremes (baby → adult). Each person therefore has one or more **embedding
clusters** — named groups representing distinct life stages or appearances.

A face matches a person if it is within the similarity threshold of **any** of
their clusters. Clusters can be named manually ("childhood", "adult") or left
unlabelled. New clusters can be added at any time as new eras are encountered.

---

## Face States

- `unidentified` — no person assigned, appears in the labelling queue
- `identified` — assigned to a person and cluster
- `anonymous` — permanently dismissed, never shown in the queue again

---

## Tech Stack

| Layer                      | Technology                        | Notes                      |
|----------------------------|-----------------------------------|----------------------------|
| Language                   | Python 3.12                       |                            |
| UI                         | PySide6 6.10+                     |                            |
| Face detection & embedding | InsightFace (ArcFace)             | GPU-accelerated            |
| ML runtime                 | ONNX Runtime GPU                  | CUDA via NVIDIA driver     |
| Vector search              | FAISS `IndexFlatIP`               | CPU, exact search          |
| Relational metadata        | SQLAlchemy 2.0 + SQLite           |                            |
| Schema migrations          | Alembic                           | Auto-applied at startup    |
| Package manager            | uv                                |                            |
| Linter/formatter           | ruff, black                       |                            |
| Type checker               | ty                                |                            |
| Testing                    | pytest, pytest-qt, pytest-alembic | In-memory SQLite for tests |
| Distribution               | PyInstaller + appimagetool        | AppImage                   |

---

## Project Structure

```
photo-aident/
├── src/photoaident/
│   ├── __init__.py
│   ├── __main__.py              # Entry point
│   ├── app.py                   # QApplication bootstrap, applies migrations
│   ├── paths.py                 # AppPaths: XDG path resolution, test override
│   ├── core/
│   │   ├── indexer.py           # Scans folders read-only, drives pipeline
│   │   ├── embeddings.py        # InsightFace/ArcFace wrapper
│   │   ├── search.py            # Similarity search + metadata filter logic
│   │   └── labeller.py          # Suggestion engine
│   ├── db/
│   │   ├── database.py          # SQLAlchemy models + session factory
│   │   ├── migrations/          # Alembic migration scripts
│   │   │   ├── env.py
│   │   │   ├── script.py.mako
│   │   │   └── versions/
│   │   └── vector_store.py      # FAISS index wrapper
│   ├── ui/
│   │   ├── main_window.py
│   │   ├── pages/
│   │   │   ├── library.py
│   │   │   ├── indexer.py
│   │   │   ├── labelling.py
│   │   │   └── persons.py
│   │   └── widgets/
│   │       ├── thumbnail_grid.py
│   │       ├── face_crop.py
│   │       └── person_card.py
│   └── utils/
│       └── image_utils.py       # Read-only image helpers, EXIF, thumbnails
├── tests/
│   ├── conftest.py              # Shared fixtures: temp AppPaths, in-memory DB
│   ├── test_database.py
│   ├── test_vector_store.py
│   ├── test_indexer.py
│   ├── test_search.py
│   ├── test_labeller.py
│   └── fixtures/
│       └── images/              # Small real JPEGs for integration tests
├── alembic.ini
├── photoaident.spec
├── pyproject.toml
└── uv.lock
```

---

## Database Schema

The schema is designed so that new analysis passes (metadata, scenery
classification, GPS search) can be added later without touching existing tables.
Each concern lives in its own table. Alembic manages all schema evolution.

### `images`

Core record per indexed image file.

- `id` — primary key
- `file_path` — absolute path to image (unique, never modified by the app)
- `file_hash` — SHA256 of file contents; used for deduplication and change
  detection; reindexing is triggered when hash changes
- `file_size` — bytes
- `indexed_at` — first indexed timestamp
- `updated_at` — last reindex timestamp
- `index_version` — integer, incremented on each reindex; compared against
  per-face `model_version` to identify outdated embeddings

### `image_metadata`

EXIF and filesystem metadata. Separate table so it can be populated, updated,
or skipped independently of face indexing.

- `id` — primary key
- `image_id` — FK to `images` (unique)
- `taken_at` — datetime from EXIF `DateTimeOriginal` (nullable)
- `taken_at_source` — enum: `exif` | `filesystem` | `manual`
- `camera_make` — nullable
- `camera_model` — nullable
- `gps_lat` — decimal degrees, nullable
- `gps_lon` — decimal degrees, nullable
- `gps_altitude` — metres, nullable
- `width` / `height` — pixels
- `orientation` — EXIF value 1–8

**Indexes:** `idx_metadata_taken_at` on `taken_at`; `idx_metadata_gps` on
`(gps_lat, gps_lon)` for bounding-box GPS queries.

### `image_tags`

Key-value store for future per-image classification labels. Adding a new
classifier (scenery, event type, season) means writing new rows here — no
schema migration required.

- `id` — primary key
- `image_id` — FK to `images`
- `tag_key` — e.g. `scene:beach`, `event:party`, `season:summer`
- `tag_value` — float confidence (0.0–1.0) or string for manual tags
- `tag_source` — enum: `model` | `manual`
- `model_name` — model identifier for `model` tags, e.g. `scenery-v1` (nullable)
- `created_at` — timestamp

**Index:** `idx_tags_key_value` on `(tag_key, tag_value)`.

### `faces`

One row per detected face in an image.

- `id` — primary key
- `image_id` — FK to `images`
- `faiss_id` — position in FAISS index (stable link between SQLite and FAISS)
- `bbox_x, bbox_y, bbox_w, bbox_h` — bounding box in source image (pixels)
- `detection_confidence` — float
- `person_id` — FK to `persons`, nullable
- `cluster_id` — FK to `embedding_clusters`, nullable
- `state` — enum: `unidentified` | `identified` | `anonymous`
- `labelled_at` — timestamp of last manual action
- `model_version` — embedding model version string, e.g. `arcface-r100-v1`;
  used to identify faces that need reindexing when the model is upgraded

**Indexes:** `idx_faces_state` on `state`; `idx_faces_image` on `image_id`.

### `persons`

A named person the user cares about identifying.

- `id` — primary key
- `name` — display name
- `notes` — optional free text
- `created_at` — timestamp

### `embedding_clusters`

A named group of embeddings for a person representing a life stage or appearance
era. Each person has one or more clusters.

- `id` — primary key
- `person_id` — FK to `persons`
- `label` — optional name, e.g. "childhood" (nullable)
- `created_at` — timestamp

The cluster's effective embedding for search is computed as the mean of all
`identified` face embeddings assigned to it, fetched from FAISS at query time.

### `suggestions`

Pending match suggestions from the labeller engine.

- `id` — primary key
- `face_id` — FK to `faces`
- `person_id` — FK to `persons`
- `cluster_id` — FK to `embedding_clusters`
- `similarity_score` — float
- `state` — enum: `pending` | `confirmed` | `rejected`
- `created_at` — timestamp

---

## Schema Migrations (Alembic)

Migrations run automatically at app startup — the user never runs CLI commands.

To generate a new migration after changing models:

```bash
uv run alembic revision --autogenerate -m "describe the change"
```

Always review autogenerated migrations before committing — Alembic's SQLite
support requires `render_as_batch=True` in `env.py` for column alterations.

---

## Reindexing Strategy

Reindexing re-runs face detection and embedding on already-indexed images without
losing any labelling work.

**Triggers:**

- User explicitly requests full reindex
- A new embedding model version is deployed (`model_version` mismatch)
- File hash changes (file was modified externally — though the collection should
  never be touched, external tools might update EXIF)

**What is always preserved:**

- All `Person` and `EmbeddingCluster` records
- All face assignments (`state = identified`) and dismissals (`state = anonymous`)
- All `image_metadata` and `image_tags` records
- All `Suggestion` records

**Reindex algorithm per image:**

1. Run detection + embedding on the image (read-only file access)
2. Match new detections to existing face records by bounding box IoU ≥ 0.5
3. For matched faces: update `faiss_id` and `model_version`, preserve state and
   person assignment
4. For new detections (no match): insert as `unidentified`
5. For old faces with no detection match: soft-delete (`deleted_at` timestamp),
   preserving history
6. Update `images.index_version` and `images.updated_at`

---

## Search Strategy

Search always combines FAISS (face similarity) with SQLite (metadata filters):

1. FAISS query returns candidate `faiss_id` list for the selected person/cluster
2. SQLite JOIN resolves face → image → image_metadata → image_tags
3. Metadata predicates applied in SQL (date range, GPS bbox, tag confidence)
4. Results deduplicated by image, returned to UI

For pure metadata queries with no person filter (future feature): skip FAISS,
query SQLite directly using the metadata indexes.

---

## Zero-Install Development & Testing

No external database server is required at any point. SQLite is embedded.

* `paths.py` — central path resolver
* `tests/conftest.py` — shared fixtures

Tests use `db_session` for full isolation — each test gets a clean transaction
that is rolled back afterwards. Migrations run once per session. No server, no
docker, no setup steps beyond `uv sync`.

---

## Test Coverage Targets

| Module               | Coverage target | Notes                                                 |
|----------------------|-----------------|-------------------------------------------------------|
| `db/database.py`     | 95%+            | Models, relations, state transitions                  |
| `db/vector_store.py` | 95%+            | Add, search, persist, reload                          |
| `core/indexer.py`    | 80%+            | Use fixture images; mock GPU calls                    |
| `core/search.py`     | 90%+            | End-to-end with fixture data                          |
| `core/labeller.py`   | 85%+            | Suggestion generation logic                           |
| `paths.py`           | 100%            | Trivial but critical                                  |
| `ui/`                | Best-effort     | pytest-qt for smoke tests; avoid testing Qt internals |

GPU-dependent code (`embeddings.py`) should be covered by integration tests that
are skipped when `CUDAExecutionProvider` is unavailable (`pytest.mark.gpu`).
All other tests must run without a GPU.

---

## Development Commands

```bash
uv sync                              # Install all dependencies
uv run photoaident                   # Run the app
uv run pytest                        # All tests (no GPU required)
uv run pytest -m gpu                 # GPU integration tests only
uv run pytest --cov=photoaident      # With coverage report
uv run ruff check src/               # Lint
uv run ruff format src/              # Format
uv run ty check src/                 # Type check
uv run alembic revision --autogenerate -m "description"  # New migration
uv run alembic upgrade head          # Apply migrations manually (dev use)
./scripts/build_pyinstaller.sh       # PyInstaller bundle
./scripts/build_appimage.sh          # AppImage
```

---

## GPU / CUDA Notes

- Target: NVIDIA RTX 4070, driver 580, CUDA 13
- ONNX Runtime GPU handles CUDA internally — no system `nvcc` needed
- InsightFace selects `CUDAExecutionProvider` automatically when available
- App works on CPU-only machines — indexing is slower but functional
- GPU availability shown in the status bar at startup
- All tests except `@pytest.mark.gpu` must pass on CPU-only CI

---

## Distribution

- PyInstaller bundles Python 3.12, PySide6, InsightFace, FAISS, and all deps
- Wrapped into `.AppImage` via `appimagetool` (not `appimage-builder`)
- Host provides: glibc, libGL/CUDA drivers, libxcb
- CUDA on target machine: optional — app falls back to CPU gracefully
