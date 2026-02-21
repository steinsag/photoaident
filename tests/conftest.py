import pytest

from photoaident.paths import AppPaths


@pytest.fixture(scope="session")
def tmp_paths(tmp_path_factory) -> AppPaths:
    """Isolated XDG paths for the test session — never touches real user data."""
    base = tmp_path_factory.mktemp("photoaident")
    paths = AppPaths(
        base_data=base / "data",
        base_cache=base / "cache",
        base_config=base / "config",
    )
    paths.ensure_dirs()
    return paths
