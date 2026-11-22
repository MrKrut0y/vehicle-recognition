from unittest.mock import Mock

import numpy as np
import pytest

from vehicle_recognition.result_persistence import persist_new_observations
from vehicle_recognition.track_state import TrackState


def test_persist_new_observations_saves_only_unpersisted_vehicles(tmp_path):
    repository = Mock()
    persisted_vehicle_ids = {1}
    new_state = TrackState(
        best_crop=np.full((5, 5, 3), 255, dtype=np.uint8),
        class_scores={"car": 1.0},
    )

    image_paths = persist_new_observations(
        repository=repository,
        run_id=7,
        counted_vehicle_ids={1, 2},
        vehicles={1: TrackState(), 2: new_state},
        persisted_vehicle_ids=persisted_vehicle_ids,
        image_output_directory=tmp_path,
        source_fps=15.0,
        frame_diagonal=100.0,
    )

    assert persisted_vehicle_ids == {1, 2}
    assert image_paths == {2: tmp_path / "vehicle_0002_car.jpg"}
    repository.save_observations.assert_called_once()
    assert repository.save_observations.call_args.kwargs["vehicle_ids"] == {2}


def test_persist_new_observations_skips_database_without_new_vehicles(tmp_path):
    repository = Mock()

    image_paths = persist_new_observations(
        repository=repository,
        run_id=7,
        counted_vehicle_ids={1},
        vehicles={1: TrackState()},
        persisted_vehicle_ids={1},
        image_output_directory=tmp_path,
        source_fps=15.0,
        frame_diagonal=100.0,
    )

    assert image_paths == {}
    repository.save_observations.assert_not_called()


def test_persist_new_observations_does_not_mark_failed_save_as_persisted(tmp_path):
    repository = Mock()
    repository.save_observations.side_effect = RuntimeError("database unavailable")
    persisted_vehicle_ids: set[int] = set()
    state = TrackState(
        best_crop=np.full((5, 5, 3), 255, dtype=np.uint8),
        class_scores={"car": 1.0},
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        persist_new_observations(
            repository=repository,
            run_id=7,
            counted_vehicle_ids={1},
            vehicles={1: state},
            persisted_vehicle_ids=persisted_vehicle_ids,
            image_output_directory=tmp_path,
            source_fps=15.0,
            frame_diagonal=100.0,
        )

    assert persisted_vehicle_ids == set()
