import os
from pathlib import Path


class AppPaths:
    """Central XDG path resolver for PhotoAIdent.

    Follows XDG Base Directory Specification for config, data, and cache.
    Paths can be overridden by passing base_data, base_cache, and base_config
    to the constructor, which is useful for tests.
    """

    def __init__(
        self,
        base_data: Path | None = None,
        base_cache: Path | None = None,
        base_config: Path | None = None,
    ):
        # Use XDG environment variables if available,
        # otherwise default to ~/.local/share, ~/.cache, ~/.config
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME")

        default_data = (
            Path(xdg_data_home) if xdg_data_home else Path.home() / ".local/share"
        )
        default_cache = (
            Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
        )
        default_config = (
            Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
        )

        self.data = base_data or default_data / "photoaident"
        self.cache = base_cache or default_cache / "photoaident"
        self.config = base_config or default_config / "photoaident"

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database."""
        return self.data / "db" / "photoaident.db"

    @property
    def faiss_path(self) -> Path:
        """Path to the FAISS index."""
        return self.data / "db" / "faiss.index"

    @property
    def face_crops_dir(self) -> Path:
        """Directory for face crop thumbnails."""
        return self.cache / "faces"

    @property
    def thumbs_dir(self) -> Path:
        """Directory for full-photo thumbnails."""
        return self.cache / "thumbs"

    @property
    def config_file(self) -> Path:
        """Path to the TOML configuration file."""
        return self.config / "config.toml"

    @property
    def translations_dir(self) -> Path:
        """Directory for translation files."""
        # Use the root assets directory for translations
        return Path(__file__).parents[2] / "assets" / "translations"

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.face_crops_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.config.mkdir(parents=True, exist_ok=True)
