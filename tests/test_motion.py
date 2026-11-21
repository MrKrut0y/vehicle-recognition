from unittest.mock import Mock

import numpy as np

from vehicle_recognition.detection import Detection
from vehicle_recognition.motion import PixelMotionAnalyzer


def make_detection() -> Detection:
    return Detection(
        class_id=2,
        class_name="car",
        confidence=0.9,
        x1=0.0,
        y1=0.0,
        x2=20.0,
        y2=20.0,
        track_id=1,
    )


def test_pixel_motion_is_ignored_during_background_warmup():
    analyzer = PixelMotionAnalyzer(warmup_frames=30)
    analyzer.background_subtractor = Mock()
    analyzer.background_subtractor.apply.return_value = np.full(
        (20, 20), 255, dtype=np.uint8
    )
    detection = make_detection()

    analyzer.update(np.zeros((20, 20, 3), dtype=np.uint8), [detection], 30)

    assert detection.pixel_motion_ratio == 1.0
    assert detection.has_pixel_motion is False


def test_pixel_motion_is_confirmed_after_background_warmup():
    analyzer = PixelMotionAnalyzer(warmup_frames=30)
    analyzer.background_subtractor = Mock()
    analyzer.background_subtractor.apply.return_value = np.full(
        (20, 20), 255, dtype=np.uint8
    )
    detection = make_detection()

    analyzer.update(np.zeros((20, 20, 3), dtype=np.uint8), [detection], 31)

    assert detection.pixel_motion_ratio == 1.0
    assert detection.has_pixel_motion is True
