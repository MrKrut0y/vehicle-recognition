from pathlib import Path

import pytest

from vehicle_recognition.database import DatabaseConfig, MySQLRepository
from vehicle_recognition.track_state import TrackState


class FakeCursor:
    def __init__(self, fetchone_results=None) -> None:
        self.executed: list[tuple[str, object]] = []
        self.executed_many: list[tuple[str, object]] = []
        self.fetchone_results = list(fetchone_results or [])
        self.lastrowid: int | None = None
        self.rowcount = 0
        self.closed = False

    def execute(self, statement, parameters=None) -> None:
        self.executed.append((statement, parameters))

        if "INSERT INTO processing_runs" in statement:
            self.lastrowid = 7
        elif "INSERT INTO vehicle_observations" in statement:
            self.lastrowid = 42

    def executemany(self, statement, parameters) -> None:
        self.executed_many.append((statement, parameters))

    def fetchone(self):
        return self.fetchone_results.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, fetchone_results=None) -> None:
        self.cursor_instance = FakeCursor(fetchone_results)
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def make_config() -> DatabaseConfig:
    return DatabaseConfig(
        host="127.0.0.1",
        port=3306,
        database="vehicle_recognition",
        user="vehicle_app",
        password="secret",
    )


def test_database_config_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_HOST", "database.local")
    monkeypatch.setenv("MYSQL_PORT", "3307")
    monkeypatch.setenv("MYSQL_DATABASE", "vehicles")
    monkeypatch.setenv("MYSQL_USER", "app")
    monkeypatch.setenv("MYSQL_PASSWORD", "secret")
    monkeypatch.setenv("DATABASE_RETENTION_DAYS", "30")

    config = DatabaseConfig.from_environment(tmp_path / "missing.env")

    assert config == DatabaseConfig(
        host="database.local",
        port=3307,
        database="vehicles",
        user="app",
        password="secret",
        retention_days=30,
    )


def test_create_run_returns_generated_id_and_commits():
    connection = FakeConnection()
    repository = MySQLRepository(make_config(), lambda **arguments: connection)

    run_id = repository.create_run("video", "video.mp4", 100, 25.0)

    assert run_id == 7
    assert connection.committed is True
    assert connection.rolled_back is False
    assert connection.cursor_instance.closed is True
    assert connection.closed is True


def test_recover_interrupted_runs_marks_running_rows_as_failed():
    connection = FakeConnection()
    connection.cursor_instance.rowcount = 2
    repository = MySQLRepository(make_config(), lambda **arguments: connection)

    recovered_run_count = repository.recover_interrupted_runs()

    statement, parameters = connection.cursor_instance.executed[0]
    assert "run.status = 'failed'" in statement
    assert "WHERE run.status = 'running'" in statement
    assert "observation_summary.observation_count" in statement
    assert parameters is None
    assert recovered_run_count == 2
    assert connection.committed is True
    assert connection.rolled_back is False


def test_update_run_progress_updates_only_active_run():
    connection = FakeConnection()
    repository = MySQLRepository(make_config(), lambda **arguments: connection)

    repository.update_run_progress(
        run_id=7,
        processed_frames=200,
        accepted_observations=4,
    )

    statement, parameters = connection.cursor_instance.executed[0]
    assert "UPDATE processing_runs" in statement
    assert "status = 'running'" in statement
    assert parameters == (200, 4, 7)
    assert connection.committed is True
    assert connection.rolled_back is False


def test_save_observations_upserts_observation_and_tracker_ids():
    connection = FakeConnection()
    repository = MySQLRepository(make_config(), lambda **arguments: connection)
    state = TrackState()
    state.add_observation(
        center=(10.0, 10.0),
        frame_number=1,
        box=(5.0, 5.0, 15.0, 15.0),
        tracker_id=101,
        class_name="car",
        confidence=0.8,
        frame_area=10_000.0,
        pixel_motion_ratio=0.25,
        has_pixel_motion=True,
    )
    state.add_observation(
        center=(20.0, 10.0),
        frame_number=11,
        box=(15.0, 5.0, 25.0, 15.0),
        tracker_id=202,
        class_name="car",
        confidence=0.9,
        frame_area=10_000.0,
        pixel_motion_ratio=0.5,
        has_pixel_motion=True,
    )
    state.finalized = True
    state.completion_reason = "video_ended"

    repository.save_observations(
        run_id=7,
        source_fps=10.0,
        vehicle_ids={1},
        vehicles={1: state},
        frame_diagonal=100.0,
        image_paths={1: Path("results/vehicle_0001_car.jpg")},
    )

    statements = [statement for statement, _ in connection.cursor_instance.executed]
    assert any("INSERT INTO vehicle_observations" in sql for sql in statements)
    assert any("ON DUPLICATE KEY UPDATE" in sql for sql in statements)
    assert any("DELETE FROM observation_tracker_ids" in sql for sql in statements)
    assert connection.cursor_instance.executed_many[0][1] == [
        (42, 101, 1),
        (42, 202, 1),
    ]
    assert connection.committed is True
    assert connection.rolled_back is False


def test_complete_run_updates_status_without_reinserting_observations():
    connection = FakeConnection()
    repository = MySQLRepository(make_config(), lambda **arguments: connection)

    repository.complete_run(
        run_id=7,
        processed_frames=100,
        accepted_observations=3,
    )

    statement, parameters = connection.cursor_instance.executed[0]
    assert "UPDATE processing_runs" in statement
    assert "INSERT INTO vehicle_observations" not in statement
    assert parameters == ("completed", 100, 3, 7)
    assert connection.committed is True


def test_fail_run_records_error_and_commits():
    connection = FakeConnection()
    repository = MySQLRepository(make_config(), lambda **arguments: connection)

    repository.fail_run(7, 25, RuntimeError("processing failed"))

    statement, parameters = connection.cursor_instance.executed[0]
    assert "status = 'failed'" in statement
    assert parameters == (25, "processing failed", 7)
    assert connection.committed is True


def test_clear_results_deletes_processing_data_and_preserves_cameras():
    connection = FakeConnection(fetchone_results=[(0, 3, 12, 15)])
    repository = MySQLRepository(make_config(), lambda **arguments: connection)

    summary = repository.clear_results()

    statements = [statement for statement, _ in connection.cursor_instance.executed]
    assert any("DELETE FROM vehicle_observations" in sql for sql in statements)
    assert any("DELETE FROM processing_runs" in sql for sql in statements)
    assert all("DELETE FROM cameras" not in sql for sql in statements)
    assert summary.processing_runs == 3
    assert summary.vehicle_observations == 12
    assert summary.tracker_ids == 15
    assert connection.committed is True
    assert connection.rolled_back is False


def test_clear_results_refuses_to_delete_active_run():
    connection = FakeConnection(fetchone_results=[(1, 1, 0, 0)])
    repository = MySQLRepository(make_config(), lambda **arguments: connection)

    with pytest.raises(
        RuntimeError,
        match="Cannot clear results while a processing run is active",
    ):
        repository.clear_results()

    statements = [statement for statement, _ in connection.cursor_instance.executed]
    assert all("DELETE FROM" not in sql for sql in statements)
    assert connection.committed is False
    assert connection.rolled_back is True


def test_clear_results_older_than_deletes_only_expired_processing_data():
    connection = FakeConnection(fetchone_results=[(2, 8, 11)])
    repository = MySQLRepository(make_config(), lambda **arguments: connection)

    summary = repository.clear_results_older_than(30)

    statements = [statement for statement, _ in connection.cursor_instance.executed]
    assert any("DELETE FROM vehicle_observations" in sql for sql in statements)
    assert any("DELETE FROM processing_runs" in sql for sql in statements)
    assert all("DELETE FROM cameras" not in sql for sql in statements)
    assert all("status <> 'running'" in sql for sql in statements)
    assert summary.processing_runs == 2
    assert summary.vehicle_observations == 8
    assert summary.tracker_ids == 11
    assert connection.committed is True
    assert connection.rolled_back is False


def test_clear_results_older_than_rejects_non_positive_period():
    repository = MySQLRepository(make_config())

    with pytest.raises(ValueError, match="retention_days must be greater than zero"):
        repository.clear_results_older_than(0)
