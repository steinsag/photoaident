# AGENTS.md — PhotoAIdent

## App Functionality

PhotoAIdent is a local, privacy-first desktop app for AI-powered face recognition
and photo search (PySide6 + InsightFace/ArcFace + FAISS + SQLite). No cloud, no
external API calls.

**Core workflow:**

1. **Index** — scan images read-only, detect faces, compute embeddings, store in DB; all faces start as `unidentified`
2. **Label** — review face crops; assign to a person, create a new person, or mark as `anonymous`
3. **Search** — filter by person, date range, GPS area, scene tags

**Embedding clusters:** The 30-year photo span means a single embedding per person fails (baby → adult). Each person has exactly **5 fixed age-group clusters** created automatically on insert:

| Key         | Age range |
|-------------|-----------|
| `infant`    | 0–3       |
| `youngster` | 4–12      |
| `teenager`  | 13–19     |
| `adult`     | 20–75     |
| `senior`    | 75+       |

Canonical key list: `AGE_CLUSTERS` in `db/database.py`. A face matches a person if within threshold of **any** cluster; the UI pre-selects the best-matching slot when labelling.

**Face states:** `unidentified` (in queue) · `identified` (assigned to person+cluster) · `anonymous` (permanently dismissed)

**UI layout:** 4-page `QStackedWidget` in `MainWindow`:
- Index 0: `LibraryPage` — thumbnail grid + person/FAISS search
- Index 1: `BrowsePage` — folder-tree column browser
- Index 2: `PersonsPage` — manage persons and embedding clusters
- Index 3: `LabellingPage` — face-by-face labelling queue

---

## ⚠️ CRITICAL: Image Collection Is Read-Only

**Never write to, modify, move, rename, or delete any file in the user's image collection.**

- All output goes to XDG paths (see below); the indexer opens files read-only only
- Never write outside `~/.local/share/photoaident/` or `~/.cache/photoaident/`
- If in doubt: **do not touch the image collection**

---

## Key Files

| Concept                       | Path                                                      |
|-------------------------------|-----------------------------------------------------------|
| Entry point                   | `src/photoaident/__main__.py`                             |
| QApplication + MainWindow     | `src/photoaident/app.py`                                  |
| DB models (SQLAlchemy)        | `src/photoaident/db/database.py`                          |
| FAISS wrapper                 | `src/photoaident/db/vector_store.py`                      |
| Face embedding (ArcFace)      | `src/photoaident/core/embeddings.py`                      |
| Indexer (Qt worker threads)   | `src/photoaident/core/indexing.py` + `core/inventory.py`  |
| XDG paths                     | `src/photoaident/paths.py`                                |
| Settings (TOML)               | `src/photoaident/settings.py`                             |
| Pages (4-page stacked UI)     | `src/photoaident/ui/pages/`                               |
| Reusable widgets              | `src/photoaident/ui/widgets/`                             |

---

## Agent Checklist (before marking any task done)

1. Run `uv run scripts/verify.py` — runs black, ruff --fix, ty check, lupdate staleness check, and pytest in sequence
2. Fix all reported issues; never skip verification, even for trivial changes
3. **If you added a new module or widget without tests:** use the `test-writer` agent to generate them (target: 90%+ coverage)
4. **If you added, removed, or changed any user-facing string (`self.tr("...")`):**
   - Invoke the `/i18n` skill — it runs lupdate, fills in German translations, and recompiles `.qm` files
   - `verify.py` will fail if `.ts` files are stale

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
| Type checker               | pyright, ty                       |                            |
| Testing                    | pytest, pytest-qt, pytest-alembic | In-memory SQLite for tests |
| Distribution               | PyInstaller + linuxdeploy         | AppImage                   |

---

## XDG-Compliant File Locations

| Content                             | Path                                           |
|-------------------------------------|------------------------------------------------|
| User config (settings, preferences) | `~/.config/photoaident/config.toml`            |
| SQLite database                     | `~/.local/share/photoaident/db/photoaident.db` |
| FAISS index                         | `~/.local/share/photoaident/db/faiss.index`    |
| Face crop thumbnails                | `~/.cache/photoaident/faces/<face_id>.jpg`     |
| Photo thumbnails                    | `~/.cache/photoaident/thumbs/<image_hash>.jpg` |

`~/.cache/` data (thumbnails, face crops) can be regenerated by re-indexing.
Always resolve paths via `AppPaths` — never inline — so tests can override them.

---

## Scale Characteristics

Target: ~80,000 JPEGs, ~240,000 face vectors (512-dim). `IndexFlatIP` (exact search)
is appropriate — single-digit ms on CPU. Reassess beyond ~5M vectors.

- **FAISS RAM:** ~500 MB · **Face crop cache:** ~2.4 GB · **SQLite:** negligible

---

## Schema Migrations (Alembic)

Migrations run automatically at startup. To generate a new migration after changing models:

```bash
uv run alembic revision --autogenerate -m "describe the change"
```

Always review before committing — SQLite requires `render_as_batch=True` in `env.py`
for column alterations.

---

## Reindexing Strategy

Reindexing re-runs detection/embedding without losing labelling work.

**Always preserved:** `Person`, `EmbeddingCluster`, face assignments, `image_metadata`,
`image_tags`, `Suggestion` records.

**Algorithm per image:**

1. Run detection + embedding (read-only file access)
2. Match new detections to existing faces by bounding box IoU ≥ 0.5
3. Matched: update `faiss_id`/`model_version`, preserve state and person assignment
4. New detections (no match): insert as `unidentified`
5. Old faces with no match: soft-delete (`deleted_at`)
6. Update `images.index_version` and `images.updated_at`

**Triggers:** explicit user request · `model_version` mismatch · file hash change

---

## Search Strategy

1. FAISS query → candidate `faiss_id` list for selected person/cluster
2. SQLite JOIN: face → image → `image_metadata` → `image_tags`
3. Metadata predicates in SQL (date range, GPS bbox, tag confidence)
4. Deduplicated by image, returned to UI

For pure metadata queries (no person filter): skip FAISS, query SQLite directly.

---

## Testing

- Tests use in-memory SQLite with per-test transaction rollback (`tests/conftest.py`).
- Test coverage should be 90%+ when possible.
- `core/indexing.py` / `core/inventory.py` — `IndexingTask` and `InventoryTask`; use fixture images; mock GPU calls
- `ui/` - pytest-qt for smoke tests; avoid testing Qt internals
- GPU tests: `@pytest.mark.gpu` — skipped when CUDA unavailable.
- All others must pass on CPU-only CI.

---

## Development Commands

```bash
uv sync                              # Install all dependencies
uv run photoaident                   # Run the app
uv run scripts/verify.py             # Format + lint + type check + translations + tests
uv run pytest                        # Tests only
uv run pytest -m gpu                 # GPU integration tests only
uv run ruff check --fix --quiet      # Lint (auto-fix)
uv run black --quiet .               # Format
uv run pyright --pythonversion 3.12 src/ tests/           # Type check (pyright)
uv run ty check                      # Type check (ty)
uv run alembic revision --autogenerate -m "description"  # New migration
uv run alembic upgrade head          # Apply migrations manually (dev use)
./scripts/build_pyinstaller.sh       # PyInstaller bundle
./scripts/build_appimage.sh          # AppImage

# Update translations after adding/changing UI strings:
uv run pyside6-lupdate -locations none -extensions py src/ -ts assets/translations/photoaident_de.ts assets/translations/photoaident_en.ts
for ts in assets/translations/*.ts; do uv run pyside6-lrelease "$ts" -qm "${ts%.ts}.qm"; done
```

---

## Code Standards: Python

You are a senior Python engineer.

- Follow PEP8 and idiomatic Python 3.12+
- Follow SOLID principles.
- Prioritize readability over cleverness.
- Use type hints everywhere
- Prefer dataclasses, enums, pathlib, typing, f-strings, list comprehensions,
  context managers (`with`) and modern stdlib
- Use clear naming (no abbreviations)
- Avoid mutable default arguments
- Use dependency injection instead of globals
- Include docstring of public functions
- Be testable and side-effect minimal
- Use proper logging instead of print
- Remove dead code when modifying files
- Improve surrounding code when touch a file (Boy Scout Rule)
- Refactor obvious duplication
- Simplify overly complex logic
- Replace legacy patterns with modern Python

---

## GPU / CUDA Notes

- Target: NVIDIA RTX, CUDA >= 13
- ONNX Runtime GPU handles CUDA internally — no system `nvcc` needed
- InsightFace selects `CUDAExecutionProvider` automatically when available
- App works on CPU-only machines — indexing is slower but functional
- GPU availability shown in the status bar at startup

---

## Distribution

- PyInstaller bundles Python 3.12, PySide6, InsightFace, FAISS, and all deps
- Wrapped into `.AppImage` via `linuxdeploy`
- Host provides: glibc, libGL/CUDA drivers, libxcb
- CUDA on target machine: optional — app falls back to CPU gracefully
