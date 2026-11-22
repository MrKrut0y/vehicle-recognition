import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mysql.connector
from dotenv import load_dotenv

from vehicle_recognition.track_state import TrackState


@dataclass(frozen=True)
class DatabaseConfig:
    """Хранит параметры подключения к MySQL."""

    host: str
    port: int
    database: str
    user: str
    password: str
    retention_days: int | None = None

    @classmethod
    def from_environment(cls, env_path: Path = Path(".env")) -> "DatabaseConfig":
        """Загружает настройки из окружения и локального файла `.env`."""
        load_dotenv(env_path, override=False)

        required_values = {
            "MYSQL_DATABASE": os.getenv("MYSQL_DATABASE"),
            "MYSQL_USER": os.getenv("MYSQL_USER"),
            "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD"),
        }
        missing_names = [name for name, value in required_values.items() if not value]

        if missing_names:
            missing_text = ", ".join(missing_names)
            raise ValueError(f"Missing database settings: {missing_text}")

        port_text = os.getenv("MYSQL_PORT", "3306")

        try:
            port = int(port_text)
        except ValueError as error:
            raise ValueError("MYSQL_PORT must be an integer") from error

        retention_text = os.getenv("DATABASE_RETENTION_DAYS", "0")

        try:
            retention_value = int(retention_text)
        except ValueError as error:
            raise ValueError("DATABASE_RETENTION_DAYS must be an integer") from error

        if retention_value < 0:
            raise ValueError("DATABASE_RETENTION_DAYS cannot be negative")

        retention_days = retention_value or None

        return cls(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=port,
            database=required_values["MYSQL_DATABASE"] or "",
            user=required_values["MYSQL_USER"] or "",
            password=required_values["MYSQL_PASSWORD"] or "",
            retention_days=retention_days,
        )

    def connection_arguments(self) -> dict[str, object]:
        """Возвращает именованные аргументы для MySQL Connector/Python."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }


@dataclass(frozen=True)
class ClearResultsSummary:
    """Содержит число удалённых строк каждой таблицы результатов."""

    processing_runs: int
    vehicle_observations: int
    tracker_ids: int
    image_paths: tuple[Path, ...] = ()


class MySQLRepository:
    """Сохраняет запуски обработки и подтверждённые наблюдения в MySQL."""

    def __init__(
        self,
        config: DatabaseConfig,
        connection_factory: Callable[..., Any] = mysql.connector.connect,
    ) -> None:
        self.config = config
        self.connection_factory = connection_factory

    @classmethod
    def from_environment(cls) -> "MySQLRepository":
        """Создаёт репозиторий из настроек локального окружения."""
        return cls(DatabaseConfig.from_environment())

    def create_run(
        self,
        source_type: str,
        source_name: str,
        total_frames: int | None,
        source_fps: float | None,
        camera_id: int | None = None,
    ) -> int:
        """Создаёт запись о начавшейся обработке и возвращает её ID."""
        if source_type not in {"video", "stream"}:
            raise ValueError(f"Unsupported source type: {source_type}")

        connection = self.connection_factory(**self.config.connection_arguments())
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO processing_runs (
                    camera_id,
                    source_type,
                    source_name,
                    status,
                    total_frames,
                    source_fps
                )
                VALUES (%s, %s, %s, 'running', %s, %s)
                """,
                (camera_id, source_type, source_name, total_frames, source_fps),
            )
            run_id = cursor.lastrowid

            if run_id is None:
                raise RuntimeError("MySQL did not return the processing run ID")

            connection.commit()
            return int(run_id)
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def get_or_create_camera(self, name: str, stream_url: str) -> int:
        """Создаёт камеру или обновляет URL существующей камеры с тем же именем."""
        normalized_name = name.strip()

        if not normalized_name:
            raise ValueError("Camera name cannot be empty")

        if len(normalized_name) > 100:
            raise ValueError("Camera name cannot be longer than 100 characters")

        connection = self.connection_factory(**self.config.connection_arguments())
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO cameras (name, stream_url, is_active)
                VALUES (%s, %s, TRUE)
                ON DUPLICATE KEY UPDATE
                    camera_id = LAST_INSERT_ID(camera_id),
                    stream_url = VALUES(stream_url),
                    is_active = TRUE
                """,
                (normalized_name, stream_url),
            )
            camera_id = cursor.lastrowid

            if camera_id is None:
                raise RuntimeError("MySQL did not return the camera ID")

            connection.commit()
            return int(camera_id)
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def recover_interrupted_runs(self) -> int:
        """Завершает запуски, оставшиеся активными после остановки приложения."""
        connection = self.connection_factory(**self.config.connection_arguments())
        cursor = connection.cursor()

        try:
            cursor.execute("""
                UPDATE processing_runs AS run
                LEFT JOIN (
                    SELECT
                        run_id,
                        COUNT(*) AS observation_count,
                        MAX(last_frame) AS last_observation_frame
                    FROM vehicle_observations
                    GROUP BY run_id
                ) AS observation_summary
                    ON observation_summary.run_id = run.run_id
                SET run.status = 'failed',
                    run.finished_at = CURRENT_TIMESTAMP(6),
                    run.processed_frames = GREATEST(
                        run.processed_frames,
                        COALESCE(observation_summary.last_observation_frame, 0)
                    ),
                    run.accepted_observations = GREATEST(
                        run.accepted_observations,
                        COALESCE(observation_summary.observation_count, 0)
                    ),
                    run.error_message =
                        'Application stopped before completing the run'
                WHERE run.status = 'running'
                """)
            recovered_run_count = int(cursor.rowcount)
            connection.commit()
            return recovered_run_count
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def update_run_progress(
        self,
        run_id: int,
        processed_frames: int,
        accepted_observations: int,
    ) -> None:
        """Сохраняет промежуточный прогресс активного запуска."""
        connection = self.connection_factory(**self.config.connection_arguments())
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                UPDATE processing_runs
                SET processed_frames = %s,
                    accepted_observations = %s
                WHERE run_id = %s
                    AND status = 'running'
                """,
                (processed_frames, accepted_observations, run_id),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def complete_run(
        self,
        run_id: int,
        processed_frames: int,
        accepted_observations: int,
        status: str = "completed",
    ) -> None:
        """Завершает запуск после сохранения всех его наблюдений."""
        if status not in {"completed", "cancelled"}:
            raise ValueError(f"Unsupported final run status: {status}")

        connection = self.connection_factory(**self.config.connection_arguments())
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                UPDATE processing_runs
                SET status = %s,
                    finished_at = CURRENT_TIMESTAMP(6),
                    processed_frames = %s,
                    accepted_observations = %s,
                    error_message = NULL
                WHERE run_id = %s
                """,
                (status, processed_frames, accepted_observations, run_id),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def save_observations(
        self,
        run_id: int,
        vehicle_ids: set[int],
        vehicles: dict[int, TrackState],
        source_fps: float,
        frame_diagonal: float,
        image_paths: dict[int, Path],
    ) -> None:
        """Сохраняет завершённые наблюдения, не закрывая текущий запуск."""
        connection = self.connection_factory(**self.config.connection_arguments())
        cursor = connection.cursor()

        try:
            for vehicle_id in sorted(vehicle_ids):
                state = vehicles[vehicle_id]
                observation_id = self._insert_observation(
                    cursor,
                    run_id,
                    vehicle_id,
                    state,
                    source_fps,
                    frame_diagonal,
                    image_paths.get(vehicle_id),
                )
                self._replace_tracker_ids(cursor, observation_id, state)

            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def fail_run(self, run_id: int, processed_frames: int, error: Exception) -> None:
        """Помечает запуск как неуспешный и сохраняет текст ошибки."""
        connection = self.connection_factory(**self.config.connection_arguments())
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                UPDATE processing_runs
                SET status = 'failed',
                    finished_at = CURRENT_TIMESTAMP(6),
                    processed_frames = %s,
                    error_message = %s
                WHERE run_id = %s
                """,
                (processed_frames, str(error), run_id),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def clear_results(self) -> ClearResultsSummary:
        """Удаляет результаты обработки, сохраняя зарегистрированные камеры."""
        connection = self.connection_factory(**self.config.connection_arguments())
        cursor = connection.cursor()

        try:
            cursor.execute("""
                SELECT image_path
                FROM vehicle_observations
                WHERE image_path IS NOT NULL
                """)
            image_paths = tuple(Path(row[0]) for row in cursor.fetchall())

            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM processing_runs WHERE status = 'running'),
                    (SELECT COUNT(*) FROM processing_runs),
                    (SELECT COUNT(*) FROM vehicle_observations),
                    (SELECT COUNT(*) FROM observation_tracker_ids)
                """)
            counts = cursor.fetchone()

            if counts is None:
                raise RuntimeError("MySQL did not return result row counts")

            active_runs, run_count, observation_count, tracker_id_count = counts

            if active_runs:
                raise RuntimeError(
                    "Cannot clear results while a processing run is active"
                )

            # Удаление наблюдений каскадно удаляет связанные ID трекера.
            cursor.execute("DELETE FROM vehicle_observations")
            cursor.execute("DELETE FROM processing_runs")
            connection.commit()

            return ClearResultsSummary(
                processing_runs=int(run_count),
                vehicle_observations=int(observation_count),
                tracker_ids=int(tracker_id_count),
                image_paths=image_paths,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def clear_results_older_than(self, retention_days: int) -> ClearResultsSummary:
        """Удаляет завершённые результаты старше указанного числа дней."""
        if retention_days <= 0:
            raise ValueError("retention_days must be greater than zero")

        connection = self.connection_factory(**self.config.connection_arguments())
        cursor = connection.cursor()
        expired_run_condition = """
            status <> 'running'
            AND finished_at IS NOT NULL
            AND finished_at < DATE_SUB(CURRENT_TIMESTAMP(6), INTERVAL %s DAY)
        """
        joined_expired_run_condition = """
            run.status <> 'running'
            AND run.finished_at IS NOT NULL
            AND run.finished_at
                < DATE_SUB(CURRENT_TIMESTAMP(6), INTERVAL %s DAY)
        """

        try:
            cursor.execute(
                f"""
                SELECT observation.image_path
                FROM vehicle_observations AS observation
                JOIN processing_runs AS run
                    ON run.run_id = observation.run_id
                WHERE {joined_expired_run_condition}
                    AND observation.image_path IS NOT NULL
                """,
                (retention_days,),
            )
            image_paths = tuple(Path(row[0]) for row in cursor.fetchall())

            cursor.execute(
                f"""
                SELECT
                    (
                        SELECT COUNT(*)
                        FROM processing_runs
                        WHERE {expired_run_condition}
                    ),
                    (
                        SELECT COUNT(*)
                        FROM vehicle_observations AS observation
                        JOIN processing_runs AS run
                            ON run.run_id = observation.run_id
                        WHERE {joined_expired_run_condition}
                    ),
                    (
                        SELECT COUNT(*)
                        FROM observation_tracker_ids AS tracker
                        JOIN vehicle_observations AS observation
                            ON observation.observation_id = tracker.observation_id
                        JOIN processing_runs AS run
                            ON run.run_id = observation.run_id
                        WHERE {joined_expired_run_condition}
                    )
                """,
                (retention_days, retention_days, retention_days),
            )
            counts = cursor.fetchone()

            if counts is None:
                raise RuntimeError("MySQL did not return expired result row counts")

            run_count, observation_count, tracker_id_count = counts

            cursor.execute(
                f"""
                DELETE FROM vehicle_observations
                WHERE run_id IN (
                    SELECT run_id
                    FROM processing_runs
                    WHERE {expired_run_condition}
                )
                """,
                (retention_days,),
            )
            cursor.execute(
                f"DELETE FROM processing_runs WHERE {expired_run_condition}",
                (retention_days,),
            )
            connection.commit()

            return ClearResultsSummary(
                processing_runs=int(run_count),
                vehicle_observations=int(observation_count),
                tracker_ids=int(tracker_id_count),
                image_paths=image_paths,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    @staticmethod
    def _insert_observation(
        cursor: Any,
        run_id: int,
        vehicle_id: int,
        state: TrackState,
        source_fps: float,
        frame_diagonal: float,
        image_path: Path | None,
    ) -> int:
        """Добавляет итоговое наблюдение одной физической машины."""
        first_seen_seconds = (state.first_frame - 1) / source_fps
        last_seen_seconds = (state.last_seen_frame - 1) / source_fps

        cursor.execute(
            """
            INSERT INTO vehicle_observations (
                run_id,
                vehicle_id,
                vehicle_type,
                confidence,
                first_frame,
                last_frame,
                frames_seen,
                first_seen_seconds,
                last_seen_seconds,
                movement_ratio,
                pixel_motion_frames,
                max_pixel_motion_ratio,
                completion_reason,
                image_path
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                observation_id = LAST_INSERT_ID(observation_id),
                vehicle_type = VALUES(vehicle_type),
                confidence = VALUES(confidence),
                first_frame = VALUES(first_frame),
                last_frame = VALUES(last_frame),
                frames_seen = VALUES(frames_seen),
                first_seen_seconds = VALUES(first_seen_seconds),
                last_seen_seconds = VALUES(last_seen_seconds),
                movement_ratio = VALUES(movement_ratio),
                pixel_motion_frames = VALUES(pixel_motion_frames),
                max_pixel_motion_ratio = VALUES(max_pixel_motion_ratio),
                completion_reason = VALUES(completion_reason),
                image_path = COALESCE(VALUES(image_path), image_path)
            """,
            (
                run_id,
                vehicle_id,
                state.stable_class_name(),
                state.stable_confidence(),
                state.first_frame,
                state.last_seen_frame,
                state.frames_seen,
                first_seen_seconds,
                last_seen_seconds,
                state.displacement_ratio(frame_diagonal),
                state.pixel_motion_frames,
                state.max_pixel_motion_ratio,
                state.completion_reason or "video_ended",
                str(image_path) if image_path is not None else None,
            ),
        )
        observation_id = cursor.lastrowid

        if observation_id is None:
            raise RuntimeError("MySQL did not return the observation ID")

        return int(observation_id)

    @staticmethod
    def _replace_tracker_ids(
        cursor: Any,
        observation_id: int,
        state: TrackState,
    ) -> None:
        """Заменяет временные ID трекера актуальным набором наблюдения."""
        cursor.execute(
            "DELETE FROM observation_tracker_ids WHERE observation_id = %s",
            (observation_id,),
        )
        tracker_rows = [
            (observation_id, tracker_id, frames_seen)
            for tracker_id, frames_seen in sorted(state.tracker_id_frames.items())
        ]

        if tracker_rows:
            cursor.executemany(
                """
                INSERT INTO observation_tracker_ids (
                    observation_id,
                    tracker_id,
                    frames_seen
                )
                VALUES (%s, %s, %s)
                """,
                tracker_rows,
            )
