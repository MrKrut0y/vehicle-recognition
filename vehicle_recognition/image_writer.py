import logging
from pathlib import Path

import cv2

from vehicle_recognition.track_state import TrackState

logger = logging.getLogger(__name__)


def save_vehicle_images(
    output_directory: Path,
    vehicle_ids: set[int],
    vehicles: dict[int, TrackState],
) -> dict[int, Path]:
    """Сохраняет лучший кадр каждой подтверждённой машины."""
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[int, Path] = {}

    for vehicle_id in sorted(vehicle_ids):
        state = vehicles.get(vehicle_id)

        if state is None or state.best_crop is None or state.best_crop.size == 0:
            continue

        class_name = state.stable_class_name().replace(" ", "_")
        output_path = output_directory / (f"vehicle_{vehicle_id:04d}_{class_name}.jpg")

        encoded_successfully, encoded_image = cv2.imencode(".jpg", state.best_crop)

        if not encoded_successfully:
            logger.warning(
                "Could not encode vehicle image: vehicle_id=%d path=%s",
                vehicle_id,
                output_path,
            )
            continue

        try:
            # tofile корректно работает с кириллицей в пути на Windows.
            encoded_image.tofile(str(output_path))
        except OSError:
            logger.exception(
                "Could not write vehicle image: vehicle_id=%d path=%s",
                vehicle_id,
                output_path,
            )
            continue

        saved_paths[vehicle_id] = output_path

    return saved_paths
