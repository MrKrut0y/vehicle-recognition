import numpy as np
import pytest

from vehicle_recognition.track_state import TrackState


def test_first_state_initializes_track_state():
    state = TrackState()
    state.add_observation((10.0, 20.0), 5, (0.0, 0.0, 20.0, 40.0))

    assert state.first_center == (10.0, 20.0)
    assert state.last_center == (10.0, 20.0)
    assert state.first_frame == 5
    assert state.last_seen_frame == 5
    assert state.last_box == (0.0, 0.0, 20.0, 40.0)
    assert state.frames_seen == 1
    assert state.max_displacement_pixels == 0.0


def test_next_observation_updates_current_state():
    state = TrackState()
    state.add_observation((0.0, 0.0), 5, (0.0, 0.0, 20.0, 40.0))
    state.add_observation((3.0, 4.0), 6, (0.0, 0.0, 30.0, 50.0))

    assert state.first_center == (0.0, 0.0)
    assert state.last_center == (3.0, 4.0)
    assert state.first_frame == 5
    assert state.last_seen_frame == 6
    assert state.last_box == (0.0, 0.0, 30.0, 50.0)
    assert state.frames_seen == 2
    assert state.max_displacement_pixels == 5.0


def test_max_displacement_does_not_decrease_when_vehicle_moves_back():
    state = TrackState()
    state.add_observation((0.0, 0.0), 1, (0.0, 0.0, 10.0, 10.0))
    state.add_observation((6.0, 8.0), 2, (6.0, 8.0, 16.0, 18.0))
    state.add_observation((3.0, 4.0), 3, (3.0, 4.0, 13.0, 14.0))

    assert state.max_displacement_pixels == pytest.approx(10.0)


def test_displacement_ratio_uses_frame_diagonal():
    state = TrackState()
    state.add_observation((0.0, 0.0), 1, (0.0, 0.0, 10.0, 10.0))
    state.add_observation((3.0, 4.0), 2, (3.0, 4.0, 13.0, 14.0))

    assert state.displacement_ratio(10.0) == pytest.approx(0.5)


@pytest.mark.parametrize("frame_diagonal", [0.0, -10.0])
def test_displacement_ratio_is_zero_for_non_positive_diagonal(frame_diagonal):
    state = TrackState(max_displacement_pixels=5.0)

    assert state.displacement_ratio(frame_diagonal) == pytest.approx(0.0)


def test_first_histogram_is_copied_into_track_state():
    histogram = np.array([[0.25, 0.75]], dtype=np.float32)
    state = TrackState()

    state.add_observation(
        (5.0, 5.0),
        1,
        (0.0, 0.0, 10.0, 10.0),
        histogram,
    )
    histogram[0, 0] = 1.0

    np.testing.assert_allclose(
        state.appearance_histogram,
        np.array([[0.25, 0.75]], dtype=np.float32),
    )


def test_next_histogram_is_blended_and_normalized():
    first_histogram = np.array([[1.0, 0.0]], dtype=np.float32)
    second_histogram = np.array([[0.0, 1.0]], dtype=np.float32)
    state = TrackState()

    state.add_observation(
        (5.0, 5.0),
        1,
        (0.0, 0.0, 10.0, 10.0),
        first_histogram,
    )
    state.add_observation(
        (6.0, 5.0),
        2,
        (1.0, 0.0, 11.0, 10.0),
        second_histogram,
    )

    np.testing.assert_allclose(
        state.appearance_histogram,
        np.array([[0.9, 0.1]], dtype=np.float32),
    )


def test_velocity_vector_is_none_with_fewer_than_two_observations():
    state = TrackState()
    state.add_observation((10.0, 20.0), 1, (0.0, 0.0, 20.0, 40.0))

    assert state.velocity_vector() is None


def test_velocity_vector_uses_position_and_frame_differences():
    state = TrackState()
    state.add_observation((10.0, 20.0), 2, (0.0, 0.0, 20.0, 40.0))
    state.add_observation((18.0, 16.0), 6, (8.0, 0.0, 28.0, 40.0))

    assert state.velocity_vector() == pytest.approx((2.0, -1.0))


def test_recent_centers_keeps_only_last_eight_observations():
    state = TrackState()

    for frame_number in range(1, 11):
        center = (float(frame_number), 0.0)
        state.add_observation(center, frame_number, (0.0, 0.0, 10.0, 10.0))

    assert len(state.recent_centers) == 8
    assert state.recent_centers[0] == (3, (3.0, 0.0))
    assert state.recent_centers[-1] == (10, (10.0, 0.0))


def test_predicted_center_uses_recent_velocity():
    state = TrackState()
    state.add_observation((10.0, 20.0), 2, (0.0, 0.0, 20.0, 40.0))
    state.add_observation((18.0, 16.0), 6, (8.0, 0.0, 28.0, 40.0))

    assert state.predicted_center(8) == pytest.approx((22.0, 14.0))


def test_observation_accumulates_motion_class_and_best_crop():
    state = TrackState()
    crop = np.full((10, 10, 3), 127, dtype=np.uint8)

    state.add_observation(
        (5.0, 5.0),
        1,
        (0.0, 0.0, 10.0, 10.0),
        tracker_id=7,
        class_name="car",
        confidence=0.8,
        crop=crop,
        frame_area=1000.0,
        pixel_motion_ratio=0.25,
        has_pixel_motion=True,
    )
    crop.fill(0)

    assert state.pixel_motion_frames == 1
    assert state.max_pixel_motion_ratio == pytest.approx(0.25)
    assert state.tracker_id_frames == {7: 1}
    assert state.stable_class_name() == "car"
    assert state.stable_confidence() == pytest.approx(0.8)
    assert state.best_crop is not None
    assert np.all(state.best_crop == 127)
