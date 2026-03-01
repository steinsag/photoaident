---
name: i18n
description: >
  Update Qt translation files. Invoke this skill automatically whenever a task
  adds, removes, or changes any user-facing string in Python source (i.e. any
  self.tr("...") call is added, deleted, or its text is modified). Runs
  lupdate, fills in missing German translations, then recompiles all .qm files.
---

Run the full Qt i18n workflow for PhotoAIdent:

1. Run lupdate to extract new/changed strings:
   uv run pyside6-lupdate -locations none -extensions py src/ -ts assets/translations/photoaident_de.ts assets/translations/photoaident_en.ts

2. Read assets/translations/photoaident_de.ts and find all entries with `type="unfinished"` or an empty `<translation>` tag.

3. For each unfinished German translation, provide an appropriate German translation (this is a desktop photo-management and face-recognition app). Ask the user to confirm or correct each suggestion before writing.

4. Write the confirmed translations into photoaident_de.ts (replace `type="unfinished"` entries with the final German text).

5. Recompile all translation files:
   for ts in assets/translations/*.ts; do uv run pyside6-lrelease "$ts" -qm "${ts%.ts}.qm"; done

6. Verify the translation check passes:
   uv run scripts/verify.py
