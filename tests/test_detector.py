from unittest.mock import MagicMock, Mock

import numpy as np

from vehicle_recognition.detector import VehicleDetector


def test_detection_crop_clamps_all_coordinates_to_frame_boundaries():
    box = Mock()
    box.id = np.array([1])
    box.xyxy = np.array([[-10.0, 0.0, -2.0, 10.0]])
    box.cls = np.array([2])
    box.conf = np.array([0.9])

    result = Mock()
    result.boxes = MagicMock()
    result.boxes.id = np.array([1])
    result.boxes.__iter__ = Mock(return_value=iter([box]))
    result.names = {2: "car"}

    detector = VehicleDetector.__new__(VehicleDetector)
    detector.model = Mock()
    detector.model.track.return_value = [result]

    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].appearance_histogram is None
    assert detections[0].x1 == 0
    assert detections[0].x2 == 0


def test_track_video_uses_one_stream_for_the_whole_source(tmp_path):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    result = Mock()
    result.orig_img = frame
    result.boxes = None
    result.names = {}
    detector = VehicleDetector.__new__(VehicleDetector)
    detector.model = Mock()
    detector.model.track.return_value = iter([result])
    source = tmp_path / "video.mp4"

    tracked_frames = list(detector.track_video(source))

    assert len(tracked_frames) == 1
    assert tracked_frames[0][1] == []
    detector.model.track.assert_called_once_with(
        source=str(source),
        stream=True,
        verbose=False,
        classes=[2, 3, 5, 7],
        conf=0.25,
        iou=0.45,
        agnostic_nms=True,
        persist=True,
        tracker="tracker_static_reid.yaml",
    )


def test_detection_ignores_boxes_without_tracker_ids():
    result = Mock()
    result.boxes = Mock()
    result.boxes.id = None

    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    detections = VehicleDetector._build_detections(result, frame)

    assert detections == []
