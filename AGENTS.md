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

1. **Verification runs automatically** — `verify.py` (black, ruff --fix, ty check, lupdate staleness check, pytest) runs when I finish
2. **If you added a new module or widget without tests:** use the `test-writer` agent to generate them (target: 90%+ coverage)
3. **If you added, removed, or changed any user-facing string (`self.tr("...")`):**
   - Invoke the `/i18n` skill — it runs lupdate, fills in German translations, and recompiles `.qm` files
   - Verification will catch stale `.ts` files

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
- **Never hardcode `/tmp` paths in tests.** Always use the `tmp_path` pytest fixture for temporary files — it provides isolated, automatically-cleaned directories per test.

---

## Development Commands

```bash
uv sync --group cuda                 # Install deps + CUDA runtime (NVIDIA GPU)
uv sync --group cpu_coreml          # Install deps + CPU/CoreML runtime
uv sync --group openvino            # Install deps + OpenVINO runtime (Intel)
uv run photoaident                   # Run the app
uv run scripts/verify.py             # Format + lint + type check + translations + tests
uv run pytest                        # Tests only
uv run pytest -m gpu                 # GPU integration tests only
uv run ruff check --fix --quiet      # Lint (auto-fix)
uv run black --quiet --target-version py312 .               # Format
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

## Hardware Acceleration Notes

Execution providers are selected automatically in priority order (see `core/providers.py`):

1. **CUDA** (`CUDAExecutionProvider`) — NVIDIA GPU; CUDA ≥ 13; no system `nvcc` needed
2. **CoreML** (`CoreMLExecutionProvider`) — Apple Silicon / macOS
3. **OpenVINO** (`OpenVINOExecutionProvider`) — Intel CPU / iGPU / NPU (e.g. N100)
4. **CPU** (`CPUExecutionProvider`) — always available as fallback

- App works on CPU-only machines — indexing is slower but fully functional
- Active provider shown in the status bar at startup
- Provider list lives in `src/photoaident/core/providers.py` — update there when adding new providers

---

## Distribution

- PyInstaller bundles Python 3.12, PySide6, InsightFace, FAISS, and all deps
- Wrapped into `.AppImage` via `linuxdeploy`
- Host provides: glibc, libGL/CUDA drivers, libxcb
- CUDA on target machine: optional — app falls back to CPU gracefully
