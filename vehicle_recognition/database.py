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

        return cls(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=port,
            database=required_values["MYSQL_DATABASE"] or "",
            user=required_values["MYSQL_USER"] or "",
            password=required_values["MYSQL_PASSWORD"] or "",
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
                    source_type,
                    source_name,
                    status,
                    total_frames,
                    source_fps
                )
                VALUES (%s, %s, 'running', %s, %s)
                """,
                (source_type, source_name, total_frames, source_fps),
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

    def complete_run(
        self,
        run_id: int,
        processed_frames: int,
        source_fps: float,
        vehicle_ids: set[int],
        vehicles: dict[int, TrackState],
        frame_diagonal: float,
        image_paths: dict[int, Path],
        status: str = "completed",
    ) -> None:
        """Транзакционно сохраняет результат успешно завершённой обработки."""
        if status not in {"completed", "cancelled"}:
            raise ValueError(f"Unsupported final run status: {status}")

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
                self._insert_tracker_ids(cursor, observation_id, state)

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
                (status, processed_frames, len(vehicle_ids), run_id),
            )
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
    def _insert_tracker_ids(
        cursor: Any,
        observation_id: int,
        state: TrackState,
    ) -> None:
        """Сохраняет все временные ID трекера, объединённые в наблюдение."""
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
