import logging
from unittest.mock import Mock

from vehicle_recognition.file_source import FileSource


def test_file_source_returns_no_frames_for_missing_file(tmp_path, caplog):
    source_path = tmp_path / "missing.mp4"
    file_source = FileSource(source_path)

    with caplog.at_level(logging.ERROR):
        frames = list(file_source.generate_frames())

    assert frames == []
    assert file_source.fps is None
    assert f"Файл не найден: {source_path}" in caplog.text


def test_file_source_reads_frames_and_releases_capture(tmp_path, monkeypatch):
    source_path = tmp_path / "video.mp4"
    source_path.touch()
    first_frame = object()
    second_frame = object()
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 25.0
    capture.read.side_effect = [
        (True, first_frame),
        (True, second_frame),
        (False, None),
    ]
    monkeypatch.setattr(
        "vehicle_recognition.file_source.cv2.VideoCapture",
        lambda _: capture,
    )
    file_source = FileSource(source_path)

    frames = list(file_source.generate_frames())

    assert frames == [first_frame, second_frame]
    assert file_source.fps == 25.0
    assert file_source.total_frames == 25
    capture.release.assert_called_once_with()


def test_file_source_releases_capture_when_video_cannot_be_opened(
    tmp_path,
    monkeypatch,
    caplog,
):
    source_path = tmp_path / "invalid.mp4"
    source_path.touch()
    capture = Mock()
    capture.isOpened.return_value = False
    monkeypatch.setattr(
        "vehicle_recognition.file_source.cv2.VideoCapture",
        lambda _: capture,
    )
    file_source = FileSource(source_path)

    with caplog.at_level(logging.ERROR):
        frames = list(file_source.generate_frames())

    assert frames == []
    assert f"Не удалось открыть видео: {source_path}" in caplog.text
    capture.release.assert_called_once_with()
