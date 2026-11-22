import logging
import time
from collections.abc import Iterator
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)

STREAM_SCHEMES = {"http", "https", "rtmp", "rtsp"}


def is_stream_url(source: str) -> bool:
    """Проверяет, является ли источник поддерживаемой сетевой трансляцией."""
    return urlparse(source).scheme.lower() in STREAM_SCHEMES


class StreamSource:
    """Читает кадры сетевой трансляции и восстанавливает соединение при обрыве."""

    def __init__(
        self,
        source: str,
        reconnect_attempts: int = 5,
        reconnect_delay_seconds: float = 2.0,
    ) -> None:
        self.source = source
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.fps: float | None = None
        self.total_frames: None = None

    @staticmethod
    def _open_capture(source: str) -> cv2.VideoCapture:
        """Открывает поток с ограниченным временем ожидания сети."""
        return cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                10_000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                10_000,
            ],
        )

    def read_metadata(self) -> bool:
        """Проверяет доступность потока и получает заявленную частоту кадров."""
        capture = self._open_capture(self.source)

        if not capture.isOpened():
            logger.error("Could not open stream: %s", self.source)
            capture.release()
            return False

        try:
            reported_fps = capture.get(cv2.CAP_PROP_FPS)
            self.fps = reported_fps if reported_fps > 0 else None
        finally:
            capture.release()

        return True

    def generate_frames(self) -> Iterator[np.ndarray]:
        """Выдаёт кадры и несколько раз переподключается после ошибки чтения."""
        failed_attempts = 0

        while failed_attempts <= self.reconnect_attempts:
            capture = self._open_capture(self.source)

            if not capture.isOpened():
                capture.release()
                failed_attempts += 1
            else:
                reported_fps = capture.get(cv2.CAP_PROP_FPS)
                if reported_fps > 0:
                    self.fps = reported_fps

                try:
                    while True:
                        success, frame = capture.read()

                        if not success:
                            break

                        failed_attempts = 0
                        yield frame
                finally:
                    capture.release()

                failed_attempts += 1

            if failed_attempts > self.reconnect_attempts:
                break

            logger.warning(
                "Stream connection lost; reconnecting: attempt=%d/%d",
                failed_attempts,
                self.reconnect_attempts,
            )
            time.sleep(self.reconnect_delay_seconds)

        raise RuntimeError(
            f"Stream unavailable after {self.reconnect_attempts} reconnect attempts"
        )
