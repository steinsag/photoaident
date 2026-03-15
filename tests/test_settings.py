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


def test_filepath_date_enabled_string_false_treated_as_disabled(tmp_path):
    """A string value like 'false' must not be accepted as a boolean True."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        'filepath_date_enabled = "false"\n'
        'filepath_date_pattern = "{YYYY}/{MM}/{DD}"\n'
    )
    settings = Settings.load(config_file)
    assert settings.filepath_date_enabled is False


def test_filepath_date_enabled_true_with_empty_pattern_is_disabled(tmp_path):
    """enabled=true with no pattern must produce filepath_date_enabled=False."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('filepath_date_enabled = true\nfilepath_date_pattern = ""\n')
    settings = Settings.load(config_file)
    assert settings.filepath_date_enabled is False


def test_filepath_date_enabled_true_with_valid_pattern(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "filepath_date_enabled = true\n" 'filepath_date_pattern = "{YYYY}/{MM}/{DD}"\n'
    )
    settings = Settings.load(config_file)
    assert settings.filepath_date_enabled is True
    assert settings.filepath_date_pattern == "{YYYY}/{MM}/{DD}"


def test_filepath_date_invalid_pattern_disables_feature(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "filepath_date_enabled = true\n"
        'filepath_date_pattern = "no-date-tokens-here"\n'
    )
    settings = Settings.load(config_file)
    assert settings.filepath_date_enabled is False


def test_filepath_date_pattern_integer_treated_as_disabled(tmp_path):
    """An integer pattern value must be rejected and feature disabled."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("filepath_date_enabled = true\nfilepath_date_pattern = 42\n")
    settings = Settings.load(config_file)
    assert settings.filepath_date_enabled is False
    assert settings.filepath_date_pattern == ""


def test_filepath_date_pattern_array_treated_as_disabled(tmp_path):
    """An array pattern value must be rejected and feature disabled."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "filepath_date_enabled = true\n"
        'filepath_date_pattern = ["{YYYY}", "{MM}", "{DD}"]\n'
    )
    settings = Settings.load(config_file)
    assert settings.filepath_date_enabled is False
    assert settings.filepath_date_pattern == ""
