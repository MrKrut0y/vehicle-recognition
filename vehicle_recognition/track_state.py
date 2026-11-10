import math
from dataclasses import dataclass


@dataclass
class TrackState:
    """Хранит состояние одного отслеживаемого транспортного средства."""

    frames_seen: int = 0
    max_displacement_pixels: float = 0.0
    first_center: tuple[float, float] | None = None

    def add_observation(self, center: tuple[float, float]) -> None:
        """Добавляет новое положение объекта и обновляет сведения о движении."""
        if self.frames_seen == 0:
            self.first_center = center

        self.frames_seen += 1

        if self.first_center is not None:
            displacement_pixels = math.dist(self.first_center, center)

            if displacement_pixels > self.max_displacement_pixels:
                self.max_displacement_pixels = displacement_pixels

    def displacement_ratio(self, frame_diagonal: float) -> float:
        """Возвращает максимальное перемещение относительно диагонали кадра."""
        if frame_diagonal <= 0.0:
            return 0.0

        return self.max_displacement_pixels / frame_diagonal
