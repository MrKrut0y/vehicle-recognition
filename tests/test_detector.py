from unittest.mock import Mock

import numpy as np

from vehicle_recognition.detector import VehicleDetector


def test_detection_crop_clamps_all_coordinates_to_frame_boundaries():
    box = Mock()
    box.id = np.array([1])
    box.xyxy = np.array([[-10.0, 0.0, -2.0, 10.0]])
    box.cls = np.array([2])
    box.conf = np.array([0.9])

    result = Mock()
    result.boxes = [box]
    result.names = {2: "car"}

    detector = VehicleDetector.__new__(VehicleDetector)
    detector.model = Mock()
    detector.model.track.return_value = [result]

    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].appearance_histogram is None
