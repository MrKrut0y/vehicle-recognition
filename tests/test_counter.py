import pytest

from vehicle_recognition.counter import VehicleCounter
from vehicle_recognition.detection import Detection
from vehicle_recognition.track_state import TrackState


def test_merge_vehicle_states_combines_observations():
    counter = VehicleCounter()
    primary_state = TrackState()
    secondary_state = TrackState()

    primary_state.add_observation((0.0, 0.0), 1, (0.0, 0.0, 10.0, 10.0))
    primary_state.add_observation((2.0, 0.0), 2, (2.0, 0.0, 12.0, 10.0))
    secondary_state.add_observation((4.0, 0.0), 3, (4.0, 0.0, 14.0, 10.0))
    secondary_state.add_observation((6.0, 0.0), 4, (6.0, 0.0, 16.0, 10.0))

    counter._merge_vehicle_states(primary_state, secondary_state)

    assert primary_state.first_frame == 1
    assert primary_state.first_center == (0.0, 0.0)
    assert primary_state.last_seen_frame == 4
    assert primary_state.last_center == (6.0, 0.0)
    assert primary_state.frames_seen == 4
    assert primary_state.max_displacement_pixels == pytest.approx(6.0)
    assert list(primary_state.recent_centers) == [
        (1, (0.0, 0.0)),
        (2, (2.0, 0.0)),
        (3, (4.0, 0.0)),
        (4, (6.0, 0.0)),
    ]


def test_merge_vehicle_states_averages_centers_from_same_frame():
    counter = VehicleCounter()
    primary_state = TrackState()
    secondary_state = TrackState()

    primary_state.add_observation((10.0, 10.0), 5, (0.0, 0.0, 20.0, 20.0))
    secondary_state.add_observation((14.0, 18.0), 5, (4.0, 8.0, 24.0, 28.0))

    counter._merge_vehicle_states(primary_state, secondary_state)

    assert list(primary_state.recent_centers) == [(5, (12.0, 14.0))]


def test_merge_vehicle_ids_remaps_tracker_ids():
    counter = VehicleCounter()
    counter.vehicles = {
        1: TrackState(),
        2: TrackState(),
    }
    counter.raw_to_vehicle = {
        101: 1,
        202: 2,
        203: 2,
    }

    counter._merge_vehicle_ids(1, 2)

    assert counter.raw_to_vehicle == {
        101: 1,
        202: 1,
        203: 1,
    }


def test_merge_vehicle_ids_removes_secondary_vehicle_and_related_state():
    counter = VehicleCounter()
    counter.vehicles = {
        1: TrackState(),
        2: TrackState(),
        3: TrackState(),
    }
    counter.counted_vehicle_ids = {2}
    counter.pending_recovery_vehicle_ids = {2}
    counter.concurrent_pair_evidence = {
        (1, 2): (5, 10),
        (2, 3): (2, 10),
        (1, 3): (1, 10),
    }

    counter._merge_vehicle_ids(1, 2)

    assert 2 not in counter.vehicles
    assert counter.counted_vehicle_ids == {1}
    assert 2 not in counter.pending_recovery_vehicle_ids
    assert all(2 not in pair for pair in counter.concurrent_pair_evidence)
    assert counter.concurrent_pair_evidence[(1, 3)] == (1, 10)


def test_concurrent_tracks_merge_only_after_required_consecutive_frames(monkeypatch):
    monkeypatch.setattr(
        "vehicle_recognition.counter.concurrent_tracks_match",
        lambda first_state, second_state: True,
    )
    counter = VehicleCounter(
        turn_recovery_confirm_frames=100,
        concurrent_merge_frames=5,
    )
    detections = [
        Detection(
            class_id=2,
            class_name="car",
            confidence=0.9,
            x1=10.0,
            y1=10.0,
            x2=30.0,
            y2=30.0,
            track_id=101,
        ),
        Detection(
            class_id=2,
            class_name="car",
            confidence=0.9,
            x1=11.0,
            y1=10.0,
            x2=31.0,
            y2=30.0,
            track_id=202,
        ),
    ]

    for frame_number in range(1, 5):
        counter.update(detections, (100, 100), frame_number)
        assert len(counter.vehicles) == 2

    counter.update(detections, (100, 100), 5)

    assert len(counter.vehicles) == 1
    assert counter.raw_to_vehicle[101] == counter.raw_to_vehicle[202]


def test_merged_tracker_ids_add_only_one_observation_per_frame():
    counter = VehicleCounter(turn_recovery_confirm_frames=100)
    counter.vehicles = {1: TrackState()}
    counter.raw_to_vehicle = {
        101: 1,
        202: 1,
    }
    detections = [
        Detection(
            class_id=2,
            class_name="car",
            confidence=0.8,
            x1=10.0,
            y1=10.0,
            x2=30.0,
            y2=30.0,
            track_id=101,
        ),
        Detection(
            class_id=2,
            class_name="car",
            confidence=0.9,
            x1=20.0,
            y1=10.0,
            x2=40.0,
            y2=30.0,
            track_id=202,
        ),
    ]

    counter.update(detections, (100, 100), 1)

    state = counter.vehicles[1]
    assert state.frames_seen == 1
    assert state.last_center == (30.0, 20.0)
    assert list(state.recent_centers) == [(1, (30.0, 20.0))]
