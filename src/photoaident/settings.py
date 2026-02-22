import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application settings for PhotoAIdent."""

    collection_path: str = ""

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings from a TOML file.

        If the file doesn't exist, returns default settings.
        """
        if not path.exists():
            return cls()

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            # For simplicity, just get known keys
            return cls(collection_path=data.get("collection_path", ""))
        except Exception as e:
            logger.error(f"Failed to load settings from {path}: {e}")
            return cls()

    def save(self, path: Path) -> None:
        """Save settings to a TOML file."""
        # Since we only have one setting for now and want to avoid new dependencies
        # if not strictly necessary, we write it manually.
        # If the settings grow, we should add tomli-w to dependencies.
        try:
            with open(path, "w") as f:
                f.write(f'collection_path = "{self.collection_path}"\n')
        except Exception as e:
            logger.error(f"Failed to save settings to {path}: {e}")
