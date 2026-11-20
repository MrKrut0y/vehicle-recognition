import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from vehicle_recognition.geometry import Box


@dataclass
class TrackState:
    """Хранит состояние одного отслеживаемого транспортного средства."""

    first_frame: int = 0
    last_seen_frame: int = 0
    frames_seen: int = 0
    max_displacement_pixels: float = 0.0
    first_center: tuple[float, float] | None = None
    last_center: tuple[float, float] | None = None
    last_box: Box | None = None
    appearance_histogram: np.ndarray | None = None
    # Короткая история нужна для оценки скорости и направления без хранения всего трека.
    recent_centers: deque[tuple[int, tuple[float, float]]] = field(
        default_factory=lambda: deque(maxlen=8)
    )

    def add_observation(
        self,
        center: tuple[float, float],
        frame_number: int,
        box: Box,
        new_histogram: np.ndarray | None = None,
    ) -> None:
        """Добавляет новое положение объекта и обновляет сведения о движении."""

        if self.frames_seen == 0:
            self.first_center = center
            self.first_frame = frame_number

        if new_histogram is not None:
            if self.appearance_histogram is None:
                self.appearance_histogram = new_histogram.copy()
            else:
                # Скользящее среднее снижает влияние бликов и случайных перекрытий.
                self.appearance_histogram = (
                    self.appearance_histogram * 0.9 + new_histogram * 0.1
                )
                cv2.normalize(
                    self.appearance_histogram,
                    self.appearance_histogram,
                    alpha=1.0,
                    norm_type=cv2.NORM_L1,
                )

        self.last_seen_frame = frame_number
        self.last_center = center
        self.last_box = box
        self.recent_centers.append((frame_number, center))

        self.frames_seen += 1

        if self.first_center is not None:
            displacement_pixels = math.dist(self.first_center, center)

            # Сохраняем максимум, чтобы поворот или возврат объекта не обнулил движение.
            if displacement_pixels > self.max_displacement_pixels:
                self.max_displacement_pixels = displacement_pixels

    def displacement_ratio(self, frame_diagonal: float) -> float:
        """Возвращает максимальное перемещение относительно диагонали кадра."""
        if frame_diagonal <= 0.0:
            return 0.0

        return self.max_displacement_pixels / frame_diagonal

    def velocity_vector(self) -> tuple[float, float] | None:
        """Возвращает среднее перемещение центра за один кадр."""
        if len(self.recent_centers) < 2:
            return None

        first_frame, first_center = self.recent_centers[0]
        last_frame, last_center = self.recent_centers[-1]

        frame_delta = max(last_frame - first_frame, 1)

        velocity_x = (last_center[0] - first_center[0]) / frame_delta
        velocity_y = (last_center[1] - first_center[1]) / frame_delta

        return (velocity_x, velocity_y)
