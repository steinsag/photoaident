from photoaident.settings import Settings


def test_settings_load_default(tmp_path):
    config_file = tmp_path / "config.toml"
    settings = Settings.load(config_file)
    assert settings.collection_path == ""


def test_settings_save_load(tmp_path):
    config_file = tmp_path / "config.toml"
    settings = Settings(collection_path="/path/to/photos")
    settings.save(config_file)

    loaded_settings = Settings.load(config_file)
    assert loaded_settings.collection_path == "/path/to/photos"


def test_settings_load_corrupt(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("invalid = {")
    settings = Settings.load(config_file)
    assert settings.collection_path == ""
