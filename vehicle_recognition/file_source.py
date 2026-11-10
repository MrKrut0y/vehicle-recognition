from pathlib import Path

import cv2


class FileSource:
    """Последовательно получает кадры из локального видеофайла."""

    def __init__(self, source: Path):
        self.source = source
        self.fps: float | None = None

    def generate_frames(self):
        """Генерирует кадры видео и освобождает ресурс после завершения."""
        if not self.source.is_file():
            print(f"Файл не найден: {self.source}")
            return

        capture = cv2.VideoCapture(str(self.source))

        if not capture.isOpened():
            print(f"Не удалось открыть видео {self.source}")
            capture.release()
            return

        self.fps = capture.get(cv2.CAP_PROP_FPS)

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                yield frame
        finally:
            capture.release()
