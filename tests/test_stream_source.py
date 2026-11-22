from unittest.mock import Mock

from vehicle_recognition.stream_source import StreamSource, is_stream_url


def test_is_stream_url_recognizes_supported_schemes():
    assert is_stream_url("https://example.com/live/playlist.m3u8") is True
    assert is_stream_url("rtsp://example.com/camera") is True
    assert is_stream_url("videos/test.mp4") is False


def test_stream_source_reads_metadata_and_releases_capture(monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 15.0
    monkeypatch.setattr(
        "vehicle_recognition.stream_source.cv2.VideoCapture",
        lambda *arguments: capture,
    )
    source = StreamSource("https://example.com/live.m3u8")

    opened = source.read_metadata()

    assert opened is True
    assert source.fps == 15.0
    assert source.total_frames is None
    capture.release.assert_called_once_with()


def test_stream_source_yields_frames_and_releases_capture(monkeypatch):
    first_frame = object()
    second_frame = object()
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 15.0
    capture.read.side_effect = [
        (True, first_frame),
        (True, second_frame),
    ]
    monkeypatch.setattr(
        "vehicle_recognition.stream_source.cv2.VideoCapture",
        lambda *arguments: capture,
    )
    source = StreamSource("https://example.com/live.m3u8")
    frames = source.generate_frames()

    assert next(frames) is first_frame
    assert next(frames) is second_frame

    frames.close()
    capture.release.assert_called_once_with()


def test_stream_source_reconnects_after_read_failure(monkeypatch):
    recovered_frame = object()
    failed_capture = Mock()
    failed_capture.isOpened.return_value = True
    failed_capture.get.return_value = 15.0
    failed_capture.read.return_value = (False, None)
    recovered_capture = Mock()
    recovered_capture.isOpened.return_value = True
    recovered_capture.get.return_value = 15.0
    recovered_capture.read.return_value = (True, recovered_frame)
    captures = iter((failed_capture, recovered_capture))
    monkeypatch.setattr(
        "vehicle_recognition.stream_source.cv2.VideoCapture",
        lambda *arguments: next(captures),
    )
    monkeypatch.setattr("vehicle_recognition.stream_source.time.sleep", lambda _: None)
    source = StreamSource(
        "https://example.com/live.m3u8",
        reconnect_attempts=2,
    )
    frames = source.generate_frames()

    assert next(frames) is recovered_frame

    frames.close()
    failed_capture.release.assert_called_once_with()
    recovered_capture.release.assert_called_once_with()
