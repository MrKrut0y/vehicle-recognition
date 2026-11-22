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
        reconnect_attempts: int | None = None,
        reconnect_delay_seconds: float = 2.0,
        max_reconnect_delay_seconds: float = 60.0,
    ) -> None:
        self.source = source
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.max_reconnect_delay_seconds = max_reconnect_delay_seconds
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
        failed_attempts = 0

        while True:
            capture = self._open_capture(self.source)

            if capture.isOpened():
                try:
                    reported_fps = capture.get(cv2.CAP_PROP_FPS)
                    self.fps = reported_fps if reported_fps > 0 else None
                finally:
                    capture.release()

                return True

            capture.release()
            failed_attempts += 1

            if not self._can_retry(failed_attempts):
                logger.error("Could not open stream: %s", self.source)
                return False

            self._wait_before_reconnect(failed_attempts)

    def generate_frames(self) -> Iterator[np.ndarray]:
        """Выдаёт кадры и несколько раз переподключается после ошибки чтения."""
        failed_attempts = 0

        while True:
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

            if not self._can_retry(failed_attempts):
                raise RuntimeError(
                    "Stream unavailable after "
                    f"{self.reconnect_attempts} reconnect attempts"
                )

            self._wait_before_reconnect(failed_attempts)

    def _can_retry(self, failed_attempts: int) -> bool:
        """Проверяет, разрешена ли следующая попытка подключения."""
        return (
            self.reconnect_attempts is None
            or failed_attempts <= self.reconnect_attempts
        )

    def _wait_before_reconnect(self, failed_attempts: int) -> None:
        """Ожидает перед подключением, постепенно увеличивая паузу."""
        exponent = min(max(failed_attempts - 1, 0), 10)
        delay_seconds = min(
            self.reconnect_delay_seconds * (2**exponent),
            self.max_reconnect_delay_seconds,
        )
        logger.warning(
            "Stream connection lost; reconnecting: attempt=%d delay=%.1fs",
            failed_attempts,
            delay_seconds,
        )
        time.sleep(delay_seconds)
