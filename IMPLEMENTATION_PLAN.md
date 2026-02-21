# PhotoAIdent — Implementation Plan

## Guiding Principle

Build vertically, not horizontally. Each phase delivers a working end-to-end slice
of functionality. The app is runnable and useful at the end of every phase.

The image collection is **never written to**. All output goes to XDG directories.

---

## Phase 1 — Foundation: Paths, DB, Migrations, Test Infrastructure

Everything else depends on this. No GPU, no UI.

### `paths.py` — central XDG path resolver

- `AppPaths` class with `data`, `cache`, `config` roots
- Properties: `db_path`, `faiss_path`, `face_crops_dir`, `thumbs_dir`
- Overridable roots so tests redirect everything to `tmp_path`

### Alembic setup

- `uv add alembic` to dependencies
- `uv run alembic init src/photoaident/db/migrations`
- Configure `env.py` to use `AppPaths` for the DB URL and `render_as_batch=True`
  (required for SQLite column alterations)
- `apply_migrations(db_url)` helper called at app startup

### SQLAlchemy models (`db/database.py`)

- `Image`, `ImageMetadata`, `ImageTag`, `Face`, `Person`, `EmbeddingCluster`,
  `Suggestion`
- All enums as Python `enum.Enum` mapped via SQLAlchemy
- `get_engine(db_path)` and `get_session_factory(engine)` helpers
- First Alembic migration generated from these models

### FAISS wrapper (`db/vector_store.py`)

- `VectorStore` wrapping `faiss.IndexFlatIP(512)`
- `add(embedding) -> int` (returns faiss_id)
- `search(embedding, k, threshold) -> list[tuple[int, float]]`
- `get_embedding(faiss_id) -> np.ndarray`
- `save()` / `load()` to/from `AppPaths.faiss_path`

### Test infrastructure (`tests/conftest.py`)

- `tmp_paths` fixture (session-scoped, `AppPaths` pointed at `tmp_path`)
- `db_engine` fixture (session-scoped, runs all Alembic migrations once)
- `db_session` fixture (function-scoped, rolls back after each test)
- `vector_store` fixture (session-scoped, in-memory)

### Tests

- All models: round-trip insert/query, FK relations, enum values
- `VectorStore`: add → search → verify, save → reload → search
- Alembic: `pytest-alembic` built-in tests (upgrade, downgrade consistency,
  model/DDL match)

**Done when:** `uv run pytest` passes with no GPU, no external processes, from a
fresh `uv sync`. Alembic migration history is clean and tested.

---

## Phase 2 — Embedding Pipeline

Get InsightFace running and producing embeddings from real images.

### `core/embeddings.py`

- `FaceResult` dataclass: `bbox`, `embedding` (512-dim L2-normalised ndarray),
  `detection_confidence`
- `EmbeddingEngine` — lazy model load, selects `CUDAExecutionProvider` if
  available, logs which provider is active
- `detect_and_embed(image_path: Path) -> list[FaceResult]`
  — opens file read-only, never modifies it

### `utils/image_utils.py`

- `compute_sha256(path: Path) -> str` — read-only
- `extract_face_crop(image_path, bbox, output_path) -> None` — saves to cache dir
- `generate_thumbnail(image_path, output_path, size=(200,200)) -> None`
- `extract_exif(image_path) -> dict` — returns parsed EXIF dict, no write

### `core/indexer.py` (plain Python, no threading yet)

- `index_directory(folder: Path, paths: AppPaths, session, store: VectorStore)`
- Walk folder for jpg/jpeg/png/webp/tiff
- Skip unchanged files: hash in DB + file hash match → skip
- For each new/changed image:
    1. Write `Image` record
    2. Extract and write `ImageMetadata` from EXIF (best-effort, skip on failure)
    3. `EmbeddingEngine.detect_and_embed()`
    4. For each face: add to FAISS, write `Face` record (state=unidentified),
       save face crop to `paths.face_crops_dir`
- Never opens a file for writing in the image collection

### Tests

- `test_indexer.py`: use `tests/fixtures/images/` (small real JPEGs committed
  to repo); verify face records appear in DB, crops appear in cache dir
- `test_embeddings.py`: marked `@pytest.mark.gpu` — skipped on CPU CI
- All file access verified to be read-only (mock `open` in write mode to assert
  it is never called on image paths)

**Done when:** Pointing the indexer at `tests/fixtures/images/` produces face
records in the DB and crop files in the temp cache dir.

---

## Phase 3 — Main Window Shell

Establish the UI skeleton. No real functionality yet.

### `ui/main_window.py`

- `QMainWindow` with a left sidebar (`QFrame`) and `QStackedWidget`
- Sidebar buttons: Library | Label | Persons | Index
- Bottom status bar: GPU provider, image count, unidentified face count
- `switch_page(name: str)` slot

### `app.py`

- Construct `AppPaths`
- Call `apply_migrations()`
- Initialise `VectorStore`
- Show `MainWindow`

### Placeholder pages

- `LibraryPage`, `LabellingPage`, `PersonsPage`, `IndexerPage` — each a
  `QWidget` with a centred label

**Done when:** Window opens, navigation works, status bar visible, app exits
cleanly.

---

## Phase 4 — Indexer Page

First fully functional UI page.

### Threaded indexer (`core/indexer.py`)

- Wrap plain indexer in `QThread` subclass
- Signals: `progress(current: int, total: int)`, `face_found(path: str, n: int)`,
  `finished(images: int, faces: int)`, `error(msg: str)`

### `ui/pages/indexer.py`

- Folder picker (`QFileDialog`)
- Start / Stop button
- `QProgressBar`
- `QPlainTextEdit` log (append-only)
- Completion summary

**Done when:** Selecting a real photo folder and clicking Start indexes images
in the background. Face crops appear in `~/.cache/photoaident/faces/`.

---

## Phase 5 — Labelling Page

The core feature.

### `ui/widgets/face_crop.py`

- Displays face crop image + small source photo thumbnail for context
- Filename and date label below

### `ui/pages/labelling.py`

- Header: "N faces remaining"
- Central `FaceCrop` widget
- Prev / Skip / Next navigation
- Action panel:
    - Assign to person (dropdown + cluster selector)
    - New person (opens `NewPersonDialog`)
    - Mark anonymous
- Keyboard shortcuts: A / N / S / X

### `NewPersonDialog`

- Name field, optional cluster label field

### Labelling logic

- Assign: set `state=identified`, `person_id`, `cluster_id`, `labelled_at`
- Anonymous: set `state=anonymous`
- After each assignment: trigger suggestion engine in background thread

**Done when:** Working through the labelling queue, assigning faces, and marking
strangers as anonymous all work correctly and persist to the DB.

---

## Phase 6 — Suggestion Engine

After labelling a face, automatically surface similar ones.

### `core/labeller.py`

- `generate_suggestions(cluster_id, session, store, threshold=0.5, max_n=50)`
- Compute mean embedding of cluster from FAISS
- Query for nearest unidentified faces within threshold
- Write `Suggestion` records (state=pending), skip duplicates

### Suggestions review in `LabellingPage`

- Second section: "Suggestions to review (N pending)"
- Face crop + "Suggested: Alice — 87% match"
- Confirm / Reject (keyboard: Y / N)
- Confirm → same as manual assign; Reject → `suggestion.state=rejected`

### Tests

- `test_labeller.py`: insert known embeddings, label one, verify suggestions
  generated for similar vectors, verify dissimilar vectors not suggested

**Done when:** Labelling yourself in one photo causes the app to suggest similar
faces from the rest of the collection.

---

## Phase 7 — Persons Page

Manage persons and their clusters.

### `ui/pages/persons.py`

- Left: person list with face count and pending suggestion count
- Right: name editor, notes, cluster list with face sample grid
- Add / Delete person (delete resets all faces to unidentified)
- Add / Rename / Merge clusters

**Done when:** All persons and clusters are browseable and editable.

---

## Phase 8 — Library & Search Page

### `ui/widgets/thumbnail_grid.py`

- `QScrollArea` with flow-layout grid of photo thumbnails
- Click → open in system viewer via `QDesktopServices.openUrl`
- Filename and date below each thumbnail

### `ui/pages/library.py`

- Person dropdown + cluster dropdown
- Date range pickers (uses `image_metadata.taken_at`)
- Similarity threshold slider
- Search button + result count
- `ThumbnailGrid`

### `core/search.py`

- `search_by_person(person_id, cluster_id, threshold, date_from, date_to,
  gps_bbox, session, store) -> list[ImageResult]`
- FAISS query → SQLite JOIN with metadata filters → deduplicate by image

**Done when:** Searching for a person shows their photos, optionally filtered by
date range.

---

## Phase 9 — Polish

- Settings dialog (default folder, threshold defaults, thumbnail size)
- Re-index: detect moved files (same hash, new path → update path, no re-embed)
- Cluster health warning if internal variance is high (suggest split)
- About dialog with version and GPU provider info
- Graceful handling of corrupt/unreadable images during indexing
- Confirm before closing while indexing is active
- `@pytest.mark.gpu` integration test suite for full pipeline

---

## Phase 10 — Packaging

- Verify PyInstaller bundle works from clean shell (no venv active)
- Check `ldd` output for unexpected host dependencies
- Test AppImage on second machine / VM
- GitHub Actions workflow: build AppImage on Ubuntu 22.04

---

## Where to Start

**Phase 1.** It has no GPU or UI dependencies, establishes the test infrastructure
every subsequent phase depends on, and the schema is the hardest thing to change
once real data exists.

### First five files to write:

1. `src/photoaident/paths.py` — `AppPaths`
2. `src/photoaident/db/database.py` — SQLAlchemy models
3. Run `uv run alembic init` and configure `env.py`
4. `src/photoaident/db/vector_store.py` — FAISS wrapper
5. `tests/conftest.py` — shared fixtures

Then run `uv run pytest` and get to green before writing any other code.
