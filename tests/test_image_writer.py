import cv2
import numpy as np

from vehicle_recognition.image_writer import save_vehicle_images
from vehicle_recognition.track_state import TrackState


def test_save_vehicle_images_writes_only_confirmed_states_with_crops(tmp_path):
    saved_state = TrackState(
        best_crop=np.full((5, 5, 3), 255, dtype=np.uint8),
        class_scores={"car": 1.0},
    )
    empty_state = TrackState()

    saved_paths = save_vehicle_images(
        tmp_path,
        {1, 2},
        {1: saved_state, 2: empty_state},
    )

    assert saved_paths == [tmp_path / "vehicle_0001_car.jpg"]
    assert saved_paths[0].is_file()
    assert cv2.imread(str(saved_paths[0])) is not None
