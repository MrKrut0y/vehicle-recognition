import pytest

from vehicle_recognition.progress import (
    estimate_remaining_seconds,
    format_duration,
    print_progress,
)


def test_print_progress_shows_percentage_and_vehicle_count(capsys):
    print_progress(
        frame_number=50,
        total_frames=100,
        vehicle_count=7,
        elapsed_seconds=20.0,
    )

    output = capsys.readouterr().out

    assert output.startswith("\r")
    assert "50.00%" in output
    assert "Кадр: 50/100" in output
    assert "Машин: 7" in output
    assert "Осталось: 00:00:20" in output


def test_print_progress_handles_unknown_total_frame_count(capsys):
    print_progress(
        frame_number=25,
        total_frames=None,
        vehicle_count=3,
        elapsed_seconds=10.0,
    )

    output = capsys.readouterr().out

    assert output == "\rКадр: 25 | Машин: 3 | Осталось: неизвестно"


def test_print_progress_does_not_exceed_one_hundred_percent(capsys):
    print_progress(
        frame_number=120,
        total_frames=100,
        vehicle_count=8,
        elapsed_seconds=30.0,
    )

    output = capsys.readouterr().out

    assert "100.00%" in output
    assert "Осталось: 00:00:00" in output


def test_estimate_remaining_seconds_uses_average_frame_time():
    remaining_seconds = estimate_remaining_seconds(
        frame_number=25,
        total_frames=100,
        elapsed_seconds=10.0,
    )

    assert remaining_seconds == pytest.approx(30.0)


def test_format_duration_includes_hours_minutes_and_seconds():
    assert format_duration(3660.2) == "01:01:01"
