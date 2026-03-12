---
name: i18n
description: >
  Update Qt translation files. Invoke this skill automatically whenever a task
  adds, removes, or changes any user-facing string in Python source (i.e. any
  self.tr("...") call is added, deleted, or its text is modified). Runs
  lupdate, fills in missing German translations, then recompiles all .qm files.
---

Run the full Qt i18n workflow for PhotoAIdent:

1. Run the translate script (lupdate + vanished/unfinished/obsolete check + lrelease):
   uv run scripts/translate.py

2. If it exits non-zero due to unfinished/vanished/obsolete strings:
   - Read assets/translations/photoaident_de.ts and find all `<translation type="unfinished">` or empty `<translation>` entries.
   - For each unfinished German translation, propose an appropriate German translation (this is a desktop photo-management and face-recognition app). Ask the user to confirm or correct each suggestion before writing.
   - Write the confirmed translations into photoaident_de.ts (set the text, remove the `type="unfinished"` attribute).
   - Re-run `uv run scripts/translate.py` and repeat until it exits 0.

3. Verify the translation check passes:
   uv run scripts/verify.py
