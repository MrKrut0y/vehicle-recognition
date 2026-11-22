import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class FileSource:
    """Последовательно получает кадры из локального видеофайла."""

    def __init__(self, source: Path):
        self.source = source
        self.fps: float | None = None
        self.total_frames: int | None = None

    def read_metadata(self) -> bool:
        """Читает параметры видео без последовательного получения кадров."""
        if not self.source.is_file():
            logger.error("Input video not found: %s", self.source)
            return False

        capture = cv2.VideoCapture(str(self.source))

        if not capture.isOpened():
            logger.error("Could not open input video: %s", self.source)
            capture.release()
            return False

        try:
            self.fps = capture.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

        return True

    def generate_frames(self):
        """Генерирует кадры видео и освобождает ресурс после завершения."""
        if not self.source.is_file():
            logger.error("Input video not found: %s", self.source)
            return

        capture = cv2.VideoCapture(str(self.source))

        if not capture.isOpened():
            logger.error("Could not open input video: %s", self.source)
            capture.release()
            return

        self.fps = capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                yield frame
        finally:
            capture.release()
