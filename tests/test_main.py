import sys

import pytest

from vehicle_recognition.main import parse_arguments

STREAM_URL = "https://example.com/live/playlist.m3u8"


def test_stream_database_arguments_include_camera_name(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vehicle-recognition",
            STREAM_URL,
            "--save-to-database",
            "--camera-name",
            "City camera 1",
        ],
    )

    arguments = parse_arguments()

    assert arguments.camera_name == "City camera 1"


def test_stream_database_arguments_require_camera_name(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["vehicle-recognition", STREAM_URL, "--save-to-database"],
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_local_video_rejects_camera_name(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["vehicle-recognition", "video.mp4", "--camera-name", "Camera"],
    )

    with pytest.raises(SystemExit):
        parse_arguments()
