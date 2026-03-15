import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application settings for PhotoAIdent."""

    collection_path: str = ""
    filepath_date_enabled: bool = False
    filepath_date_pattern: str = ""

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings from a TOML file.

        If the file doesn't exist, returns default settings.
        Invalid filepath date patterns are treated as disabled to guard
        against manual file tampering.
        """
        if not path.exists():
            return cls()

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            raw_enabled = data.get("filepath_date_enabled", False)
            if not isinstance(raw_enabled, bool):
                logger.warning(
                    "filepath_date_enabled must be a boolean, got %r; disabling.",
                    raw_enabled,
                )
                raw_enabled = False
            pattern = str(data.get("filepath_date_pattern", ""))
            enabled = raw_enabled and bool(pattern)

            if enabled:
                from photoaident.core.filepath_date import compile_pattern

                try:
                    compile_pattern(pattern)
                except ValueError:
                    logger.warning(
                        "Invalid filepath_date_pattern %r in settings; "
                        "disabling filepath date extraction.",
                        pattern,
                    )
                    enabled = False

            return cls(
                collection_path=data.get("collection_path", ""),
                filepath_date_enabled=enabled,
                filepath_date_pattern=pattern,
            )
        except Exception:
            logger.error("Failed to load settings from %s", path, exc_info=True)
            return cls()

    def save(self, path: Path) -> None:
        """Save settings to a TOML file."""
        try:
            with open(path, "w") as f:
                f.write(f'collection_path = "{self.collection_path}"\n')
                f.write(
                    f"filepath_date_enabled = "
                    f"{'true' if self.filepath_date_enabled else 'false'}\n"
                )
                f.write(f'filepath_date_pattern = "{self.filepath_date_pattern}"\n')
        except Exception:
            logger.error("Failed to save settings to %s", path, exc_info=True)
