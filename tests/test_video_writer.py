from unittest.mock import Mock

import pytest

from vehicle_recognition.video_writer import (
    SegmentedVideoFileWriter,
    VideoFileWriter,
)


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


def test_segmented_writer_rotates_after_frame_limit(tmp_path):
    created_writers = []

    def writer_factory(**arguments):
        writer = Mock()
        writer.output_path = arguments["output_path"]
        created_writers.append(writer)
        return writer

    writer = SegmentedVideoFileWriter(
        output_path=tmp_path / "stream.mp4",
        fps=2.0,
        frame_size=(640, 480),
        segment_duration_seconds=1.0,
        max_segments=2,
        writer_factory=writer_factory,
    )

    writer.write("frame-1")
    writer.write("frame-2")
    writer.write("frame-3")
    writer.release()

    assert [item.output_path.name for item in created_writers] == [
        "stream_0001.mp4",
        "stream_0002.mp4",
    ]
    created_writers[0].release.assert_called_once_with()
    created_writers[1].release.assert_called_once_with()


def test_segmented_writer_deletes_segments_beyond_limit(tmp_path):
    created_paths = []

    def writer_factory(**arguments):
        output_path = arguments["output_path"]
        output_path.write_bytes(b"video")
        created_paths.append(output_path)
        return Mock()

    writer = SegmentedVideoFileWriter(
        output_path=tmp_path / "stream.mp4",
        fps=1.0,
        frame_size=(640, 480),
        segment_duration_seconds=1.0,
        max_segments=2,
        writer_factory=writer_factory,
    )

    writer.write("frame-1")
    writer.write("frame-2")
    writer.write("frame-3")
    writer.release()

    assert created_paths[0].exists() is False
    assert created_paths[1].exists() is True
    assert created_paths[2].exists() is True


def test_segmented_writer_includes_existing_segments_in_limit(tmp_path):
    first_path = tmp_path / "stream_0001.mp4"
    second_path = tmp_path / "stream_0002.mp4"
    first_path.write_bytes(b"old")
    second_path.write_bytes(b"newer")

    def writer_factory(**arguments):
        arguments["output_path"].write_bytes(b"current")
        return Mock()

    writer = SegmentedVideoFileWriter(
        output_path=tmp_path / "stream.mp4",
        fps=1.0,
        frame_size=(640, 480),
        segment_duration_seconds=1.0,
        max_segments=2,
        writer_factory=writer_factory,
    )

    writer.write("frame")
    writer.release()

    assert first_path.exists() is False
    assert second_path.exists() is True
    assert (tmp_path / "stream_0003.mp4").exists() is True
