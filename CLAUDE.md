# CLAUDE.md — PhotoAIdent

## Project Overview

PhotoAIdent is a local, privacy-first desktop application for AI-powered face recognition and photo search. It scans a
local photo library, detects and embeds faces using InsightFace/ArcFace, and provides a PySide6 desktop UI where the
user progressively labels faces — assigning them to known persons or marking them as anonymous. Over time, the app
learns who is in the collection and can suggest matches automatically.

No cloud, no external API calls, no data leaves the machine.

## Core User Workflow

1. **Index** — user points the app at a folder. The app scans all images, detects
   faces, computes embeddings, and stores everything locally. All faces start as
   unidentified.

2. **Label** — the app presents unidentified face crops to the user one by one (or
   in clusters). The user assigns each face to a known person, creates a new person,
   or marks the face as "anonymous" (a face the user doesn't care about identifying,
   e.g. strangers in the background).

3. **Propagate** — once a face is labelled, the app searches the rest of the index
   for similar faces and suggests matches. The user confirms or rejects suggestions.
   Confirmed matches become additional labelled embeddings for that person.

4. **Search** — once persons are established, the user can filter the photo library
   by person to find all photos containing them.

Over many labelling sessions the unidentified face queue shrinks. It never needs to
reach zero — the user only labels people they care about.

## Why a Single Mean Embedding Per Person Is Not Enough

The photo collection spans roughly 20 years. A person's appearance changes
dramatically over that time (babies → teenagers → adults). A single mean embedding
would average out these differences and produce poor matches at the extremes.

Instead, each person has **multiple embedding clusters** — groups of embeddings from
different life stages or appearances. Matching is done against all clusters, and a
face matches a person if it is within threshold distance of *any* of their clusters.

Clusters can be managed manually (user names them, e.g. "Alice — childhood",
"Alice — adult") or grown automatically as labelled embeddings accumulate.

## Face States

Every detected face in the database has one of three states:

- **Unidentified** — no person assigned yet, will appear in the labelling queue
- **Identified** — assigned to a known person
- **Anonymous** — explicitly marked as "not interesting", excluded from the queue
  and from search results

## Tech Stack

| Layer                      | Technology                                |
|----------------------------|-------------------------------------------|
| Language                   | Python 3.12                               |
| UI                         | PySide6 6.10+                             |
| Face detection & embedding | InsightFace (ArcFace model)               |
| ML runtime                 | ONNX Runtime GPU (CUDA via NVIDIA driver) |
| Vector search              | FAISS (faiss-cpu)                         |
| Relational metadata        | SQLAlchemy + SQLite                       |
| Package manager            | uv                                        |
| Linter/formatter           | ruff, black                               |
| Type checker               | ty                                        |
| Testing                    | pytest, pytest-qt, pytest-cov             |
| Distribution               | PyInstaller + appimagetool (AppImage)     |

## Project Structure

```
photo-aident/
├── src/photoaident/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── app.py               # QApplication bootstrap
│   ├── core/
│   │   ├── indexer.py       # Scans image folders, drives the pipeline
│   │   ├── embeddings.py    # InsightFace/ArcFace wrapper, returns 512-dim vectors
│   │   ├── search.py        # Similarity search logic against FAISS index
│   │   └── labeller.py      # Suggestion engine: given a face, find likely persons
│   ├── db/
│   │   ├── database.py      # SQLAlchemy models and session management
│   │   └── vector_store.py  # FAISS index load/save/query wrapper
│   ├── ui/
│   │   ├── main_window.py   # QMainWindow shell with sidebar navigation
│   │   ├── pages/
│   │   │   ├── library.py   # Photo grid with search/filter UI
│   │   │   ├── indexer.py   # Indexing progress page
│   │   │   ├── labelling.py # Face labelling queue page
│   │   │   └── persons.py   # Person and cluster management
│   │   └── widgets/
│   │       ├── thumbnail_grid.py
│   │       ├── face_crop.py      # Displays a single face crop with action buttons
│   │       └── person_card.py
│   └── utils/
│       └── image_utils.py   # Thumbnail generation, face crop extraction, EXIF
├── data/                    # Runtime data, gitignored
│   ├── db/
│   │   ├── photoaident.db   # SQLite database
│   │   └── faiss.index      # Serialized FAISS index
│   └── thumbnails/          # Cached thumbnails and face crops
├── assets/
│   └── icons/
│       └── app.png
├── scripts/
│   ├── build.sh             # PyInstaller build
│   └── build_appimage.sh    # AppImage packaging via appimagetool
├── tools/
│   └── appimagetool         # appimagetool binary, gitignored
├── tests/
├── photoaident.spec         # PyInstaller spec
├── pyproject.toml
└── uv.lock
```

---

## Database Schema (SQLite via SQLAlchemy)

The schema is designed so that new analysis passes (metadata extraction, scenery
classification, etc.) can be added later without touching existing tables or
invalidating existing data. Each analysis concern lives in its own table.

### `images`

Core record for an indexed image file. Intentionally lean — only identity and
indexing state live here. All derived data goes in separate tables.

- `id` — primary key
- `file_path` — absolute path to the image (unique)
- `file_hash` — SHA256 of file contents, used for deduplication and change detection
- `file_size` — bytes
- `indexed_at` — timestamp when first indexed
- `updated_at` — timestamp of last re-index (updated on each reindex pass)
- `index_version` — integer, incremented each time this image is reprocessed;
  allows querying which images were indexed with an older pipeline version

The `index_version` column is the key to supporting future reindexing. When a new
model or analysis pass is introduced, its results store their own `model_version`.
Comparing the two tells you which images need reprocessing without touching
unaffected records.

### `image_metadata`

EXIF and filesystem metadata. Stored separately from `images` so it can be
populated, updated, or skipped independently of face indexing.

- `id` — primary key
- `image_id` — FK to `images` (unique — one metadata record per image)
- `taken_at` — timestamp from EXIF DateTimeOriginal (nullable if not present)
- `taken_at_source` — enum: `exif` | `filesystem` | `manual`
  (records where the timestamp came from, important for reliability)
- `camera_make` — EXIF Make (nullable)
- `camera_model` — EXIF Model (nullable)
- `gps_lat` — decimal degrees, nullable
- `gps_lon` — decimal degrees, nullable
- `gps_altitude` — metres, nullable
- `gps_source` — enum: `exif` | `manual` (nullable)
- `width` — image width in pixels
- `height` — image height in pixels
- `orientation` — EXIF orientation value (1–8)

**Indexes for efficient filtering:**

- `idx_metadata_taken_at` on `taken_at` — date range queries
- `idx_metadata_gps` on `(gps_lat, gps_lon)` — bounding box GPS queries
  (for future map-based search; exact spatial indexing via R-tree can be added
  later if needed)

### `image_tags`

Flat key-value store for future scenery classification labels and any other
per-image tags. Avoids schema changes when new tag types are introduced.

- `id` — primary key
- `image_id` — FK to `images`
- `tag_key` — e.g. `scene:indoor`, `scene:beach`, `event:party`, `season:summer`
- `tag_value` — float confidence score (0.0–1.0) or string for manual tags
- `tag_source` — enum: `model` | `manual`
- `model_name` — name and version of the model that produced this tag (nullable
  for manual tags), e.g. `scenery-v1`
- `created_at` — timestamp

**Index:** `idx_tags_key_value` on `(tag_key, tag_value)` — efficient filtering
by tag type and minimum confidence threshold.

This table is not used yet but is created from day one. Adding a new classifier
later means writing new rows here — no migration needed.

### `faces`

A single detected face within an image.

- `id` — primary key
- `image_id` — FK to `images`
- `faiss_id` — integer index into the FAISS vector index
- `bbox_x, bbox_y, bbox_w, bbox_h` — bounding box in the source image (pixels)
- `detection_confidence` — float, confidence score from the detector
- `person_id` — FK to `persons`, nullable
- `cluster_id` — FK to `embedding_clusters`, nullable
- `state` — enum: `unidentified` | `identified` | `anonymous`
- `labelled_at` — timestamp of last manual action
- `model_version` — version string of the embedding model used to produce this
  face's embedding, e.g. `arcface-r100-v1`; allows targeted reindexing when the
  model is upgraded

**Index:** `idx_faces_state` on `state` — fast queue queries for unidentified faces.

### `persons`

A named person the user cares about identifying.

- `id` — primary key
- `name` — display name
- `notes` — optional free text
- `created_at` — timestamp

### `embedding_clusters`

A named group of embeddings for a person, representing a distinct life stage
or appearance era. A person has one or more clusters.

- `id` — primary key
- `person_id` — FK to `persons`
- `label` — optional name, e.g. "childhood", "adult" (nullable = auto)
- `created_at` — timestamp

The cluster's effective embedding for matching is the mean of all face embeddings
assigned to it, computed at query time from the FAISS vectors.

### `suggestions`

Pending match suggestions generated by the labeller engine.

- `id` — primary key
- `face_id` — FK to `faces` (the unidentified face)
- `person_id` — FK to `persons`
- `cluster_id` — FK to `embedding_clusters`
- `similarity_score` — cosine similarity 0.0–1.0
- `state` — enum: `pending` | `confirmed` | `rejected`
- `created_at` — timestamp

---

## Reindexing Strategy

Reindexing must be possible without losing labelling work (person assignments,
cluster memberships, anonymous flags). The schema supports this as follows:

**What reindexing means:**

- Re-running face detection + embedding on images already in the database
- Triggered when: a new model version is deployed, detection parameters change,
  or the user explicitly requests it

**What must be preserved across reindexing:**

- All `Person` and `EmbeddingCluster` records
- All `Face` records where `state = identified` or `state = anonymous`
- All `Suggestion` records
- All `image_metadata` records
- All `image_tags` records

**Reindex algorithm:**

1. Query images where `faces.model_version != current_model_version`
   (or all images if full reindex requested)
2. For each such image, run detection + embedding again
3. For each newly detected face:
    - If a face at approximately the same bounding box already exists with
      `state = identified` or `state = anonymous`: update its `faiss_id` and
      `model_version`, preserve all labelling
    - If a face at approximately the same bounding box exists with
      `state = unidentified`: replace embedding in FAISS, update `faiss_id`
    - If a face is newly detected (no bbox match): insert as unidentified
    - If an old face has no bbox match in the new detection: mark as deleted
      (soft delete — add `deleted_at` column) so labelling history is preserved
4. Update `images.index_version` and `images.updated_at`

Bounding box matching uses a configurable IoU (intersection over union) threshold
(default 0.5) to account for minor detector differences between model versions.

---

## Core Pipeline

### Indexing

1. Walk target directory recursively for image files (jpg, jpeg, png, webp, tiff)
2. Skip unchanged files via SHA256 hash check (hash unchanged = file unchanged)
3. For each image:
   a. Extract and store EXIF/metadata into `image_metadata`
   b. Detect faces via InsightFace RetinaFace
   c. For each face: compute 512-dim ArcFace embedding, L2-normalise
   d. Store embedding in FAISS, store face record with `state = unidentified`
   and current `model_version`
   e. Extract and cache face crop to `data/thumbnails/faces/<face_id>.jpg`
4. Run in a `QThread`, emit progress signals

### Labelling Queue

1. Query all `faces` with `state = unidentified`, ordered by
   `image_metadata.taken_at` ascending (oldest photos first)
2. Present face crop + source photo context to user
3. User action: assign to person/cluster | new person | anonymous | skip
4. After assignment: trigger suggestion engine in background

### Suggestion Engine (`core/labeller.py`)

1. Compute mean embedding of the updated cluster
2. Query FAISS for N nearest unidentified faces within threshold
3. Write pending `Suggestion` records
4. UI presents suggestions as a confirm/reject review queue

### Search

1. User selects person (+ optional cluster) and any metadata filters
   (date range, GPS bounding box, tags)
2. FAISS query returns candidate face records
3. Filter candidates through SQLite JOIN with `image_metadata` and `image_tags`
   applying the metadata predicates
4. Deduplicate by image, return results to UI

**Note on filtering order:** FAISS search is fast even at scale, so the pattern is
FAISS first (face similarity) → SQLite filter (metadata). For pure metadata queries
with no person filter (future feature), go directly to SQLite which has the indexes.

---

## Matching Strategy

- Embeddings are L2-normalised so inner product = cosine similarity
- `faiss.IndexFlatIP` — exact search, appropriate for home collection sizes
- A face matches a person if within threshold of **any** of their clusters
- Default similarity threshold: 0.5 (tunable per person)

---

## Key Design Decisions

- **Labelling is the primary input mechanism.** No separate reference photo upload.
  Persons and embeddings grow from the labelling process.
- **Multiple clusters per person** handle appearance changes over time.
- **`image_tags` key-value table** means adding scenery classification, event
  detection, or any other per-image labels in the future requires no schema
  migration — just new rows with a new `tag_key`.
- **`model_version` on `Face`** enables targeted reindexing: only faces produced
  by an outdated model need reprocessing. Labelling is preserved.
- **`image_metadata` is a separate table** so metadata extraction can be run,
  re-run, or skipped independently of face indexing.
- **FAISS index is append-only during indexing.** Deletions handled by soft-delete
  in SQLite; the FAISS vector is ignored thereafter.
- **GPU used only for InsightFace inference.** FAISS runs on CPU.
- **Indexing runs in QThread.** Never block the main thread.
- **Face crops pre-extracted and cached** at index time for fast labelling queue.
- **Suggestions are non-destructive** — never auto-confirmed without user action.

---

## Development Commands

```bash
uv sync                         # Install dependencies
uv run photoaident              # Run the app
uv run pytest                   # Run tests
uv run ruff check src/          # Lint
uv run ty check src/            # Type check
./scripts/build.sh              # PyInstaller bundle
./scripts/build_appimage.sh     # AppImage
```

## GPU / CUDA Notes

- Requires NVIDIA GPU with driver 520+
- ONNX Runtime GPU handles CUDA internally — no system CUDA toolkit needed
- InsightFace automatically uses `CUDAExecutionProvider` when available
- App works on CPU-only machines but indexing is significantly slower
- GPU status shown in status bar at startup

## Distribution

- PyInstaller bundles Python, Qt, InsightFace, FAISS, and all dependencies
- Wrapped into `.AppImage` using `appimagetool` directly (not `appimage-builder`)
- Host system provides: glibc, libGL/CUDA drivers, libxcb