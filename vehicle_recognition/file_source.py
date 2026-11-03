from pathlib import Path

import cv2


class FileSource:
    def __init__(self, source: Path):
        self.source = source

    def generate_frames(self):
        if not self.source.is_file():
            print(f"Файл не найден: {self.source}")
            return

        capture = cv2.VideoCapture(str(self.source))

        if not capture.isOpened():
            print(f"Не удалось открыть видео {self.source}")
            capture.release()
            return

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                yield frame
        finally:
            capture.release()
