"""Unit tests for FilterPanel widget in isolation."""

from unittest.mock import patch

import pytest
from PySide6 import QtWidgets

from photoaident.core.date_range import DateRange
from photoaident.core.geo import GpsBoundingBox
from photoaident.db.database import Person
from photoaident.ui.widgets.filter_panel import FilterPanel


@pytest.fixture
def panel(qtbot, session_factory, tmp_app_paths):
    fp = FilterPanel(session_factory, tmp_app_paths)
    qtbot.addWidget(fp)
    return fp


# --- Construction ---


def test_panel_has_correct_fixed_width(panel):
    assert panel.maximumWidth() == 220
    assert panel.minimumWidth() == 220


def test_panel_has_location_button(panel):
    assert not panel.map_location_btn.isHidden()


def test_panel_has_date_button(panel):
    assert not panel.date_filter_btn.isHidden()


def test_panel_has_person_search_and_list(panel):
    assert not panel.search_edit.isHidden()
    assert not panel.person_list_widget.isHidden()


def test_clear_location_btn_initially_hidden(panel):
    assert panel.clear_location_btn.isHidden()


def test_clear_time_btn_initially_hidden(panel):
    assert panel.clear_time_btn.isHidden()


# --- Initial state ---


def test_gps_bbox_initially_none(panel):
    assert panel.gps_bbox() is None


def test_date_range_initially_none(panel):
    assert panel.date_range() is None


def test_selected_person_ids_initially_empty(panel):
    assert panel.selected_person_ids() == []


# --- populate_person_list ---


def test_populate_person_list_shows_persons(qtbot, session_factory, tmp_app_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.add(Person(name="Bob"))
        session.commit()

    fp = FilterPanel(session_factory, tmp_app_paths)
    qtbot.addWidget(fp)
    fp._populate_person_list()

    assert fp.person_list_widget.count() == 2


def test_populate_person_list_preserves_selection(
    qtbot, session_factory, tmp_app_paths
):
    with session_factory() as session:
        p = Person(name="Alice")
        session.add(p)
        session.commit()
        person_id = p.id

    fp = FilterPanel(session_factory, tmp_app_paths)
    qtbot.addWidget(fp)
    fp._populate_person_list()

    fp.person_list_widget.item(0).setSelected(True)
    assert person_id in fp.selected_person_ids()

    fp._populate_person_list()
    assert fp.person_list_widget.item(0).isSelected()
    assert person_id in fp.selected_person_ids()


def test_populate_person_list_sorts_alphabetically(
    qtbot, session_factory, tmp_app_paths
):
    with session_factory() as session:
        session.add(Person(name="Zara"))
        session.add(Person(name="Alice"))
        session.commit()

    fp = FilterPanel(session_factory, tmp_app_paths)
    qtbot.addWidget(fp)
    fp._populate_person_list()

    assert fp.person_list_widget.item(0).text() == "Alice"
    assert fp.person_list_widget.item(1).text() == "Zara"


def test_show_event_repopulates_person_list(qtbot, session_factory, tmp_app_paths):
    """FilterPanel repopulates the person list when it becomes visible."""
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.commit()

    fp = FilterPanel(session_factory, tmp_app_paths)
    qtbot.addWidget(fp)

    fp.person_list_widget.clear()
    assert fp.person_list_widget.count() == 0

    from PySide6 import QtGui

    fp.showEvent(QtGui.QShowEvent())

    assert fp.person_list_widget.count() == 1
    assert fp.person_list_widget.item(0).text() == "Alice"


# --- Person search filter ---


def test_search_filter_hides_non_matching(qtbot, session_factory, tmp_app_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.add(Person(name="Bob"))
        session.commit()

    fp = FilterPanel(session_factory, tmp_app_paths)
    qtbot.addWidget(fp)
    fp._populate_person_list()
    fp.search_edit.setText("ali")

    bob_item = fp.person_list_widget.item(1)
    assert bob_item is not None
    assert bob_item.isHidden()


def test_search_filter_is_case_insensitive(qtbot, session_factory, tmp_app_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.commit()

    fp = FilterPanel(session_factory, tmp_app_paths)
    qtbot.addWidget(fp)
    fp._populate_person_list()
    fp.search_edit.setText("ALICE")

    item = fp.person_list_widget.item(0)
    assert item is not None
    assert not item.isHidden()


def test_search_filter_shows_all_when_empty(qtbot, session_factory, tmp_app_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.add(Person(name="Bob"))
        session.commit()

    fp = FilterPanel(session_factory, tmp_app_paths)
    qtbot.addWidget(fp)
    fp._populate_person_list()
    fp.search_edit.setText("ali")
    fp.search_edit.clear()

    for i in range(fp.person_list_widget.count()):
        assert not fp.person_list_widget.item(i).isHidden()


def test_apply_search_filter_handles_none_item(panel):
    with patch.object(panel.person_list_widget, "item", side_effect=[None]):
        with patch.object(panel.person_list_widget, "count", return_value=1):
            panel._apply_search_filter("test")  # must not raise


# --- Location filter ---


def test_open_map_dialog_accepted_sets_gps_bbox(panel):
    bbox = GpsBoundingBox(south=52.0, west=13.0, north=53.0, east=14.0)

    with patch("photoaident.ui.widgets.filter_panel.MapLocationDialog") as MockDlg:
        mock_dlg = MockDlg.return_value
        mock_dlg.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        mock_dlg.selected_bbox.return_value = bbox

        panel._open_map_dialog()

    assert panel.gps_bbox() == bbox
    assert not panel.clear_location_btn.isHidden()
    assert panel.map_location_btn.isChecked()


def test_open_map_dialog_rejected_keeps_old_bbox(panel):
    original = GpsBoundingBox(south=1.0, west=1.0, north=2.0, east=2.0)
    panel._gps_bbox = original

    with patch("photoaident.ui.widgets.filter_panel.MapLocationDialog") as MockDlg:
        mock_dlg = MockDlg.return_value
        mock_dlg.exec.return_value = QtWidgets.QDialog.DialogCode.Rejected

        panel._open_map_dialog()

    assert panel.gps_bbox() == original


def test_on_location_cleared_resets_state(panel):
    panel._gps_bbox = GpsBoundingBox(south=52.0, west=13.0, north=53.0, east=14.0)
    panel._update_map_button()

    panel._on_location_cleared()

    assert panel.gps_bbox() is None
    assert panel.clear_location_btn.isHidden()
    assert not panel.map_location_btn.isChecked()


def test_location_changed_signal_emitted_on_accept(qtbot, panel):
    bbox = GpsBoundingBox(south=10.0, west=10.0, north=20.0, east=20.0)

    with patch("photoaident.ui.widgets.filter_panel.MapLocationDialog") as MockDlg:
        mock_dlg = MockDlg.return_value
        mock_dlg.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        mock_dlg.selected_bbox.return_value = bbox

        with qtbot.waitSignal(panel.location_changed, timeout=1000) as blocker:
            panel._open_map_dialog()

    assert blocker.args == [bbox]


def test_location_changed_signal_emitted_on_clear(qtbot, panel):
    panel._gps_bbox = GpsBoundingBox(south=10.0, west=10.0, north=20.0, east=20.0)

    with qtbot.waitSignal(panel.location_changed, timeout=1000) as blocker:
        panel._on_location_cleared()

    assert blocker.args == [None]


# --- Date filter ---


def test_open_date_dialog_accepted_sets_date_range(panel):
    date_range = DateRange(start_year=2020, end_year=2023)

    with patch("photoaident.ui.widgets.filter_panel.DateFilterDialog") as MockDlg:
        mock_dlg = MockDlg.return_value
        mock_dlg.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        mock_dlg.selected_range.return_value = date_range

        panel._open_date_dialog()

    assert panel.date_range() == date_range
    assert not panel.clear_time_btn.isHidden()
    assert panel.date_filter_btn.isChecked()


def test_open_date_dialog_rejected_keeps_old_range(panel):
    original = DateRange(start_year=2018)
    panel._date_range = original

    with patch("photoaident.ui.widgets.filter_panel.DateFilterDialog") as MockDlg:
        mock_dlg = MockDlg.return_value
        mock_dlg.exec.return_value = QtWidgets.QDialog.DialogCode.Rejected

        panel._open_date_dialog()

    assert panel.date_range() == original


def test_on_time_cleared_resets_state(panel):
    panel._date_range = DateRange(start_year=2020)
    panel._update_time_button()

    panel._on_time_cleared()

    assert panel.date_range() is None
    assert panel.clear_time_btn.isHidden()
    assert not panel.date_filter_btn.isChecked()


def test_date_range_changed_signal_emitted_on_accept(qtbot, panel):
    date_range = DateRange(start_year=2021)

    with patch("photoaident.ui.widgets.filter_panel.DateFilterDialog") as MockDlg:
        mock_dlg = MockDlg.return_value
        mock_dlg.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        mock_dlg.selected_range.return_value = date_range

        with qtbot.waitSignal(panel.date_range_changed, timeout=1000) as blocker:
            panel._open_date_dialog()

    assert blocker.args == [date_range]


def test_date_range_changed_signal_emitted_on_clear(qtbot, panel):
    panel._date_range = DateRange(start_year=2019)

    with qtbot.waitSignal(panel.date_range_changed, timeout=1000) as blocker:
        panel._on_time_cleared()

    assert blocker.args == [None]


# --- Person selection signal ---


def test_person_selection_changed_signal_emitted(qtbot, session_factory, tmp_app_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.commit()

    fp = FilterPanel(session_factory, tmp_app_paths)
    qtbot.addWidget(fp)
    fp._populate_person_list()

    with qtbot.waitSignal(fp.person_selection_changed, timeout=1000):
        fp.person_list_widget.item(0).setSelected(True)
