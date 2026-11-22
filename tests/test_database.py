from pathlib import Path

from vehicle_recognition.database import DatabaseConfig, MySQLRepository
from vehicle_recognition.track_state import TrackState


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object]] = []
        self.executed_many: list[tuple[str, object]] = []
        self.lastrowid: int | None = None
        self.closed = False

    def execute(self, statement, parameters=None) -> None:
        self.executed.append((statement, parameters))

        if "INSERT INTO processing_runs" in statement:
            self.lastrowid = 7
        elif "INSERT INTO vehicle_observations" in statement:
            self.lastrowid = 42

    def executemany(self, statement, parameters) -> None:
        self.executed_many.append((statement, parameters))

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()
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

    config = DatabaseConfig.from_environment(tmp_path / "missing.env")

    assert config == DatabaseConfig(
        host="database.local",
        port=3307,
        database="vehicles",
        user="app",
        password="secret",
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


def test_complete_run_saves_observation_and_tracker_ids():
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

    repository.complete_run(
        run_id=7,
        processed_frames=100,
        source_fps=10.0,
        vehicle_ids={1},
        vehicles={1: state},
        frame_diagonal=100.0,
        image_paths={1: Path("results/vehicle_0001_car.jpg")},
    )

    statements = [statement for statement, _ in connection.cursor_instance.executed]
    assert any("INSERT INTO vehicle_observations" in sql for sql in statements)
    assert any("UPDATE processing_runs" in sql for sql in statements)
    assert connection.cursor_instance.executed_many[0][1] == [
        (42, 101, 1),
        (42, 202, 1),
    ]
    assert connection.committed is True
    assert connection.rolled_back is False


def test_fail_run_records_error_and_commits():
    connection = FakeConnection()
    repository = MySQLRepository(make_config(), lambda **arguments: connection)

    repository.fail_run(7, 25, RuntimeError("processing failed"))

    statement, parameters = connection.cursor_instance.executed[0]
    assert "status = 'failed'" in statement
    assert parameters == (25, "processing failed", 7)
    assert connection.committed is True
