import numpy as np

from vehicle_recognition.detection import Detection
from vehicle_recognition.visualizer import draw_detections


def test_draw_detections_returns_annotated_copy_without_changing_source():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = Detection(
        class_id=2,
        class_name="car",
        confidence=0.9,
        x1=10.0,
        y1=20.0,
        x2=50.0,
        y2=60.0,
        track_id=7,
    )

    annotated_frame = draw_detections(frame, [detection], total=1)

    assert annotated_frame is not frame
    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(annotated_frame) > 0


def test_draw_detections_handles_detection_without_track_id():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = Detection(
        class_id=7,
        class_name="truck",
        confidence=0.8,
        x1=10.0,
        y1=20.0,
        x2=50.0,
        y2=60.0,
        track_id=None,
    )

    annotated_frame = draw_detections(frame, [detection], total=0)

    assert annotated_frame.shape == frame.shape
    assert np.count_nonzero(annotated_frame) > 0
