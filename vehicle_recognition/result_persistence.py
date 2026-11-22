import logging
from pathlib import Path

from vehicle_recognition.database import MySQLRepository
from vehicle_recognition.image_writer import save_vehicle_images
from vehicle_recognition.track_state import TrackState

logger = logging.getLogger(__name__)


def persist_new_observations(
    repository: MySQLRepository,
    run_id: int,
    counted_vehicle_ids: set[int],
    vehicles: dict[int, TrackState],
    persisted_vehicle_ids: set[int],
    image_output_directory: Path,
    source_fps: float,
    frame_diagonal: float,
) -> dict[int, Path]:
    """Сохраняет только новые завершённые наблюдения текущего запуска."""
    new_vehicle_ids = counted_vehicle_ids - persisted_vehicle_ids

    if not new_vehicle_ids:
        return {}

    image_paths = save_vehicle_images(
        image_output_directory,
        new_vehicle_ids,
        vehicles,
    )
    repository.save_observations(
        run_id=run_id,
        vehicle_ids=new_vehicle_ids,
        vehicles=vehicles,
        source_fps=source_fps,
        frame_diagonal=frame_diagonal,
        image_paths=image_paths,
    )
    persisted_vehicle_ids.update(new_vehicle_ids)

    logger.info(
        "Vehicle observations persisted: run_id=%d vehicle_ids=%s",
        run_id,
        sorted(new_vehicle_ids),
    )

    return image_paths
