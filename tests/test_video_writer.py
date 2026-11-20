from unittest.mock import Mock

import pytest

from vehicle_recognition.video_writer import VideoFileWriter


def test_video_writer_writes_frame_and_releases_resource(tmp_path, monkeypatch):
    output_path = tmp_path / "results" / "video.mp4"
    frame = object()
    writer_resource = Mock()
    writer_resource.isOpened.return_value = True
    video_writer_factory = Mock(return_value=writer_resource)
    monkeypatch.setattr(
        "vehicle_recognition.video_writer.cv2.VideoWriter_fourcc",
        lambda *_: 1234,
    )
    monkeypatch.setattr(
        "vehicle_recognition.video_writer.cv2.VideoWriter",
        video_writer_factory,
    )

    writer = VideoFileWriter(output_path, 25.0, (640, 480))
    writer.write(frame)
    writer.release()

    assert output_path.parent.is_dir()
    video_writer_factory.assert_called_once_with(
        str(output_path),
        1234,
        25.0,
        (640, 480),
    )
    writer_resource.write.assert_called_once_with(frame)
    writer_resource.release.assert_called_once_with()


def test_video_writer_raises_error_when_output_cannot_be_opened(
    tmp_path,
    monkeypatch,
):
    output_path = tmp_path / "video.mp4"
    writer_resource = Mock()
    writer_resource.isOpened.return_value = False
    monkeypatch.setattr(
        "vehicle_recognition.video_writer.cv2.VideoWriter_fourcc",
        lambda *_: 1234,
    )
    monkeypatch.setattr(
        "vehicle_recognition.video_writer.cv2.VideoWriter",
        lambda *_: writer_resource,
    )

    with pytest.raises(RuntimeError, match="Не удалось создать видеофайл"):
        VideoFileWriter(output_path, 25.0, (640, 480))
