import math
from collections import defaultdict, deque
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
    last_box_area: float = 0.0
    appearance_histogram: np.ndarray | None = None
    pixel_motion_frames: int = 0
    last_pixel_motion_frame: int = 0
    max_pixel_motion_ratio: float = 0.0
    class_scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    confidence_sums: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    confidence_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tracker_id_frames: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    best_crop: np.ndarray | None = field(default=None, repr=False, compare=False)
    best_crop_score: float = 0.0
    finalized: bool = False
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
        *,
        tracker_id: int | None = None,
        class_name: str | None = None,
        confidence: float = 0.0,
        crop: np.ndarray | None = None,
        frame_area: float = 0.0,
        pixel_motion_ratio: float = 0.0,
        has_pixel_motion: bool = False,
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
        box_width = max(box[2] - box[0], 0.0)
        box_height = max(box[3] - box[1], 0.0)
        self.last_box_area = box_width * box_height
        self.recent_centers.append((frame_number, center))

        self.frames_seen += 1

        if self.first_center is not None:
            displacement_pixels = math.dist(self.first_center, center)

            # Сохраняем максимум, чтобы поворот или возврат объекта не обнулил движение.
            if displacement_pixels > self.max_displacement_pixels:
                self.max_displacement_pixels = displacement_pixels

        self.max_pixel_motion_ratio = max(
            self.max_pixel_motion_ratio,
            pixel_motion_ratio,
        )
        if has_pixel_motion and self.last_pixel_motion_frame != frame_number:
            self.pixel_motion_frames += 1
            self.last_pixel_motion_frame = frame_number

        if tracker_id is not None:
            self.tracker_id_frames[tracker_id] += 1

        observation_weight = 0.0
        if class_name is not None and frame_area > 0.0:
            area_ratio = max(self.last_box_area / frame_area, 1e-9)
            observation_weight = confidence * math.sqrt(area_ratio)
            self.class_scores[class_name] += observation_weight
            self.confidence_sums[class_name] += confidence
            self.confidence_counts[class_name] += 1

        if (
            crop is not None
            and crop.size > 0
            and observation_weight > self.best_crop_score
        ):
            self.best_crop = crop.copy()
            self.best_crop_score = observation_weight

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

        frame_delta = last_frame - first_frame

        if frame_delta <= 0:
            return None

        velocity_x = (last_center[0] - first_center[0]) / frame_delta
        velocity_y = (last_center[1] - first_center[1]) / frame_delta

        return (velocity_x, velocity_y)

    def stable_class_name(self) -> str:
        """Возвращает тип транспорта с наибольшим накопленным весом."""
        if not self.class_scores:
            return "unknown"

        return max(self.class_scores, key=self.class_scores.get)

    def stable_confidence(self) -> float:
        """Возвращает среднюю уверенность для стабильного типа транспорта."""
        class_name = self.stable_class_name()
        confidence_count = self.confidence_counts.get(class_name, 0)

        if confidence_count == 0:
            return 0.0

        return self.confidence_sums[class_name] / confidence_count

    def predicted_center(
        self,
        frame_number: int,
    ) -> tuple[float, float] | None:
        """Прогнозирует центр объекта на указанном кадре."""
        if self.last_center is None:
            return None

        velocity = self.velocity_vector()

        if velocity is None:
            return self.last_center

        missing_frames = max(
            frame_number - self.last_seen_frame,
            0,
        )

        predicted_x = self.last_center[0] + velocity[0] * missing_frames
        predicted_y = self.last_center[1] + velocity[1] * missing_frames

        return (predicted_x, predicted_y)
