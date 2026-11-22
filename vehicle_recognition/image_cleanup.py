import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


def delete_result_images(image_paths: Iterable[Path], image_root: Path) -> int:
    """Удаляет сохранённые снимки, не выходя за пределы каталога результатов."""
    resolved_root = image_root.resolve()
    deleted_count = 0

    for image_path in set(image_paths):
        resolved_path = image_path.resolve()

        if not resolved_path.is_relative_to(resolved_root):
            logger.warning("Skipped image outside result directory: %s", image_path)
            continue

        if not resolved_path.is_file():
            continue

        try:
            resolved_path.unlink()
        except OSError:
            logger.exception("Could not delete expired vehicle image: %s", image_path)
            continue

        deleted_count += 1
        parent = resolved_path.parent

        while parent != resolved_root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    return deleted_count
