import os
from pathlib import Path

from photoaident.paths import AppPaths


def test_app_paths_defaults():
    # Clear XDG env vars to test defaults
    env_vars = ["XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"]
    old_values = {var: os.environ.get(var) for var in env_vars}
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    try:
        paths = AppPaths()
        assert paths.data == Path.home() / ".local/share/photoaident"
        assert paths.cache == Path.home() / ".cache/photoaident"
        assert paths.config == Path.home() / ".config/photoaident"

        assert paths.db_path == paths.data / "db" / "photoaident.db"
        assert paths.faiss_path == paths.data / "db" / "faiss.index"
        assert paths.face_crops_dir == paths.cache / "faces"
        assert paths.thumbs_dir == paths.cache / "thumbs"
    finally:
        # Restore env vars
        for var, val in old_values.items():
            if val is not None:
                os.environ[var] = val


def test_app_paths_overrides(tmp_path):
    base_data = tmp_path / "data"
    base_cache = tmp_path / "cache"
    base_config = tmp_path / "config"

    try:
        AppPaths._data_override = base_data
        AppPaths._cache_override = base_cache
        AppPaths._config_override = base_config

        paths = AppPaths()

        assert paths.data == base_data
        assert paths.cache == base_cache
        assert paths.config == base_config

        assert paths.db_path == base_data / "db" / "photoaident.db"
        assert paths.face_crops_dir == base_cache / "faces"
    finally:
        AppPaths._data_override = None
        AppPaths._cache_override = None
        AppPaths._config_override = None


def test_tmp_paths_fixture(tmp_app_paths):
    assert "photoaident" in str(tmp_app_paths.data)
    assert tmp_app_paths.data.exists()
    assert tmp_app_paths.cache.exists()
    assert tmp_app_paths.config.exists()
    assert tmp_app_paths.face_crops_dir.exists()
    assert tmp_app_paths.thumbs_dir.exists()
    assert tmp_app_paths.db_path.parent.exists()
