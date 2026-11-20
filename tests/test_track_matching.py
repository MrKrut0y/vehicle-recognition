from collections import deque

import numpy as np

from vehicle_recognition.track_matching import concurrent_tracks_match
from vehicle_recognition.track_state import TrackState

MATCHING_HISTOGRAM = np.array([[0.25, 0.75]], dtype=np.float32)


def test_concurrent_tracks_match_returns_false_without_boxes():
    empty_state_1 = TrackState()
    empty_state_2 = TrackState()
    assert concurrent_tracks_match(empty_state_1, empty_state_2) is False


def test_concurrent_tracks_match_returns_false_when_one_box_is_missing():
    first_state = TrackState(last_box=(0.0, 0.0, 10.0, 10.0))
    second_state = TrackState()

    assert concurrent_tracks_match(first_state, second_state) is False


def test_concurrent_tracks_match_returns_false_without_centers():
    first_state = TrackState(last_box=(0.0, 0.0, 10.0, 10.0))
    second_state = TrackState(last_box=(0.0, 0.0, 10.0, 10.0))

    assert concurrent_tracks_match(first_state, second_state) is False


def test_concurrent_tracks_match_accepts_overlap_by_iou():
    first_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
        appearance_histogram=MATCHING_HISTOGRAM.copy(),
    )
    second_state = TrackState(
        last_box=(8.0, 0.0, 18.0, 10.0),
        last_center=(13.0, 5.0),
        appearance_histogram=MATCHING_HISTOGRAM.copy(),
    )

    assert concurrent_tracks_match(first_state, second_state) is True


def test_concurrent_tracks_match_accepts_nested_box_by_smaller_coverage():
    first_state = TrackState(
        last_box=(0.0, 0.0, 100.0, 100.0),
        last_center=(50.0, 50.0),
        appearance_histogram=MATCHING_HISTOGRAM.copy(),
    )
    second_state = TrackState(
        last_box=(10.0, 10.0, 30.0, 30.0),
        last_center=(20.0, 20.0),
        appearance_histogram=MATCHING_HISTOGRAM.copy(),
    )

    assert concurrent_tracks_match(first_state, second_state) is True


def test_concurrent_tracks_match_rejects_small_overlap():
    first_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
    )
    second_state = TrackState(
        last_box=(9.5, 0.0, 19.5, 10.0),
        last_center=(14.5, 5.0),
    )

    assert concurrent_tracks_match(first_state, second_state) is False


def test_concurrent_tracks_match_rejects_distant_boxes():
    first_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
    )
    second_state = TrackState(
        last_box=(100.0, 100.0, 110.0, 110.0),
        last_center=(105.0, 105.0),
    )

    assert concurrent_tracks_match(first_state, second_state) is False


def test_concurrent_tracks_match_rejects_excessive_center_distance():
    first_state = TrackState(
        last_box=(0.0, 0.0, 20.0, 1.0),
        last_center=(10.0, 0.5),
    )
    second_state = TrackState(
        last_box=(16.0, 0.0, 36.0, 1.0),
        last_center=(26.0, 0.5),
    )

    assert concurrent_tracks_match(first_state, second_state) is False


def test_concurrent_tracks_match_rejects_different_appearance():
    first_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
        appearance_histogram=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    second_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
        appearance_histogram=np.array([[0.0, 1.0]], dtype=np.float32),
    )

    assert concurrent_tracks_match(first_state, second_state) is False


def test_concurrent_tracks_match_accepts_matching_direction():
    first_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
        appearance_histogram=MATCHING_HISTOGRAM.copy(),
        recent_centers=deque([(1, (4.0, 5.0)), (2, (5.0, 5.0))], maxlen=8),
    )
    second_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
        appearance_histogram=MATCHING_HISTOGRAM.copy(),
        recent_centers=deque([(1, (3.0, 5.0)), (2, (5.0, 5.0))], maxlen=8),
    )

    assert concurrent_tracks_match(first_state, second_state) is True


def test_concurrent_tracks_match_rejects_opposite_directions():
    first_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
        appearance_histogram=MATCHING_HISTOGRAM.copy(),
        recent_centers=deque([(1, (4.0, 5.0)), (2, (5.0, 5.0))], maxlen=8),
    )
    second_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
        appearance_histogram=MATCHING_HISTOGRAM.copy(),
        recent_centers=deque([(1, (6.0, 5.0)), (2, (5.0, 5.0))], maxlen=8),
    )

    assert concurrent_tracks_match(first_state, second_state) is False


def test_concurrent_tracks_match_ignores_unreliable_direction():
    first_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
        appearance_histogram=MATCHING_HISTOGRAM.copy(),
        recent_centers=deque([(1, (4.9, 5.0)), (2, (5.0, 5.0))], maxlen=8),
    )
    second_state = TrackState(
        last_box=(0.0, 0.0, 10.0, 10.0),
        last_center=(5.0, 5.0),
        appearance_histogram=MATCHING_HISTOGRAM.copy(),
        recent_centers=deque([(1, (5.1, 5.0)), (2, (5.0, 5.0))], maxlen=8),
    )

    assert concurrent_tracks_match(first_state, second_state) is True
