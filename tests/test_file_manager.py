from unittest.mock import MagicMock, patch

from photoaident.utils.file_manager import reveal_in_file_manager


def test_reveal_in_file_manager_macos(tmp_path):
    p = tmp_path / "photo.jpg"
    p.touch()
    with (
        patch("photoaident.utils.file_manager.sys") as mock_sys,
        patch("photoaident.utils.file_manager.subprocess.Popen") as mock_popen,
    ):
        mock_sys.platform = "darwin"
        reveal_in_file_manager(str(p))
    mock_popen.assert_called_once_with(["open", "-R", str(p)])


def test_reveal_in_file_manager_windows(tmp_path):
    p = tmp_path / "photo.jpg"
    p.touch()
    with (
        patch("photoaident.utils.file_manager.sys") as mock_sys,
        patch("photoaident.utils.file_manager.subprocess.Popen") as mock_popen,
    ):
        mock_sys.platform = "win32"
        reveal_in_file_manager(str(p))
    mock_popen.assert_called_once_with(["explorer", "/select,", str(p)])


def test_reveal_in_file_manager_linux_dbus_success(tmp_path):
    """D-Bus call succeeds — no xdg-open fallback."""
    p = tmp_path / "photo.jpg"
    p.touch()
    mock_iface = MagicMock()
    mock_iface.isValid.return_value = True
    mock_reply = MagicMock()
    # Anything other than ErrorMessage — use a sentinel that won't equal ErrorMessage
    mock_reply.type.return_value = object()
    mock_iface.call.return_value = mock_reply

    with (
        patch("photoaident.utils.file_manager.sys") as mock_sys,
        patch("photoaident.utils.file_manager.subprocess.Popen") as mock_popen,
        patch("PySide6.QtDBus.QDBusInterface", return_value=mock_iface),
    ):
        mock_sys.platform = "linux"
        reveal_in_file_manager(str(p))
    mock_popen.assert_not_called()


def test_reveal_in_file_manager_linux_dbus_fallback(tmp_path):
    """D-Bus interface invalid — falls back to xdg-open."""
    p = tmp_path / "photo.jpg"
    p.touch()
    mock_iface = MagicMock()
    mock_iface.isValid.return_value = False

    with (
        patch("photoaident.utils.file_manager.sys") as mock_sys,
        patch("photoaident.utils.file_manager.subprocess.Popen") as mock_popen,
        patch("PySide6.QtDBus.QDBusInterface", return_value=mock_iface),
    ):
        mock_sys.platform = "linux"
        reveal_in_file_manager(str(p))
    mock_popen.assert_called_once()
    assert mock_popen.call_args[0][0][0] == "xdg-open"
