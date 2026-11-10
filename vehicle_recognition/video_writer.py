from pathlib import Path

import cv2


class VideoFileWriter:
    """Сохраняет обработанные кадры в видеофайл."""

    def __init__(
        self,
        output_path: Path,
        fps: float,
        frame_size: tuple[int, int],
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            frame_size,
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Не удалось создать видеофайл: {output_path}")

    def write(self, frame) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
