from collections.abc import Callable
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
        """Записывает один обработанный кадр."""
        self.writer.write(frame)

    def release(self) -> None:
        """Завершает запись и освобождает выходной файл."""
        self.writer.release()


class SegmentedVideoFileWriter:
    """Записывает поток в ограниченное число видеофрагментов."""

    def __init__(
        self,
        output_path: Path,
        fps: float,
        frame_size: tuple[int, int],
        segment_duration_seconds: float,
        max_segments: int,
        writer_factory: Callable[..., VideoFileWriter] = VideoFileWriter,
    ) -> None:
        if segment_duration_seconds <= 0:
            raise ValueError("segment_duration_seconds must be greater than zero")
        if max_segments <= 0:
            raise ValueError("max_segments must be greater than zero")

        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.segment_frame_limit = max(round(fps * segment_duration_seconds), 1)
        self.max_segments = max_segments
        self.writer_factory = writer_factory
        self.current_writer: VideoFileWriter | None = None
        self.current_segment_frames = 0
        self.segment_index = 0
        self.segment_paths: list[Path] = []
        self._load_existing_segments()

    def write(self, frame) -> None:
        """Записывает кадр, начиная новый сегмент при достижении лимита."""
        if (
            self.current_writer is None
            or self.current_segment_frames >= self.segment_frame_limit
        ):
            self._start_next_segment()

        if self.current_writer is None:
            raise RuntimeError("Video segment writer was not initialized")

        self.current_writer.write(frame)
        self.current_segment_frames += 1

    def release(self) -> None:
        """Завершает запись текущего сегмента."""
        if self.current_writer is not None:
            self.current_writer.release()
            self.current_writer = None

    def _start_next_segment(self) -> None:
        if self.current_writer is not None:
            self.current_writer.release()

        self.segment_index += 1
        segment_path = self.output_path.with_name(
            f"{self.output_path.stem}_{self.segment_index:04d}"
            f"{self.output_path.suffix}"
        )
        self.current_writer = self.writer_factory(
            output_path=segment_path,
            fps=self.fps,
            frame_size=self.frame_size,
        )
        self.current_segment_frames = 0
        self.segment_paths.append(segment_path)

        if len(self.segment_paths) > self.max_segments:
            expired_path = self.segment_paths.pop(0)
            expired_path.unlink(missing_ok=True)

    def _load_existing_segments(self) -> None:
        pattern = f"{self.output_path.stem}_*{self.output_path.suffix}"
        indexed_paths: list[tuple[int, Path]] = []

        for candidate in self.output_path.parent.glob(pattern):
            index_text = candidate.stem.removeprefix(f"{self.output_path.stem}_")

            if index_text.isdigit():
                indexed_paths.append((int(index_text), candidate))

        indexed_paths.sort()
        self.segment_paths = [path for _, path in indexed_paths]

        if indexed_paths:
            self.segment_index = indexed_paths[-1][0]

        while len(self.segment_paths) > self.max_segments:
            expired_path = self.segment_paths.pop(0)
            expired_path.unlink(missing_ok=True)
