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

    paths = AppPaths(
        base_data=base_data, base_cache=base_cache, base_config=base_config
    )

    assert paths.data == base_data
    assert paths.cache == base_cache
    assert paths.config == base_config

    assert paths.db_path == base_data / "db" / "photoaident.db"
    assert paths.face_crops_dir == base_cache / "faces"


def test_app_paths_ensure_dirs(tmp_path):
    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )

    paths.ensure_dirs()

    assert (tmp_path / "data" / "db").exists()
    assert (tmp_path / "cache" / "faces").exists()
    assert (tmp_path / "cache" / "thumbs").exists()
    assert (tmp_path / "config").exists()


def test_tmp_paths_fixture(tmp_paths):
    assert "photoaident" in str(tmp_paths.data)
    assert tmp_paths.data.exists()
    assert tmp_paths.cache.exists()
    assert tmp_paths.config.exists()
    assert tmp_paths.db_path.parent.exists()
