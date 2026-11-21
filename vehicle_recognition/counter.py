import logging
import math
from itertools import combinations

import cv2
import numpy as np

from vehicle_recognition.appearance import calculate_histogram_similarity
from vehicle_recognition.detection import Detection
from vehicle_recognition.geometry import (
    Box,
    calculate_box_area,
    calculate_iou,
    calculate_smaller_coverage,
)
from vehicle_recognition.track_matching import concurrent_tracks_match
from vehicle_recognition.track_state import TrackState

logger = logging.getLogger(__name__)


class VehicleCounter:
    """Сопоставляет треки и считает подтверждённые движущиеся машины."""

    def __init__(
        self,
        min_confirmed_frames: int = 10,
        min_movement_ratio: float = 0.03,
        turn_recovery_max_gap: int = 60,
        turn_recovery_distance_ratio: float = 0.04,
        turn_recovery_confirm_frames: int = 5,
        concurrent_merge_frames: int = 6,
        large_vehicle_evidence_max_gap: int = 15,
        interrupted_bus_min_smaller_coverage: float = 0.75,
        interrupted_bus_min_size_ratio: float = 1.5,
        interrupted_bus_min_appearance_similarity: float = 0.75,
        min_pixel_motion_frames: int = 5,
        max_missing_frames: int = 90,
        stitch_max_gap: int = 45,
        stitch_distance_ratio: float = 0.08,
        stitch_appearance_threshold: float = 0.55,
        turn_recovery_appearance_threshold: float = 0.45,
    ):
        self.min_confirmed_frames = min_confirmed_frames
        # Множество не позволяет засчитать одну физическую машину повторно.
        self.counted_vehicle_ids: set[int] = set()
        # Новые внутренние ID ожидают проверки: не являются ли они продолжением старого трека.
        self.pending_recovery_vehicle_ids: set[int] = set()
        self.min_movement_ratio = min_movement_ratio
        # Связывает временный ID трекера с внутренним ID физической машины.
        self.raw_to_vehicle: dict[int, int] = {}
        # Хранит накопленное состояние каждой физической машины.
        self.vehicles: dict[int, TrackState] = {}
        self.next_vehicle_id: int = 1
        self.turn_recovery_max_gap = turn_recovery_max_gap
        self.turn_recovery_distance_ratio = turn_recovery_distance_ratio
        self.turn_recovery_confirm_frames = turn_recovery_confirm_frames
        # Для каждой пары храним число совпадений подряд и кадр последнего совпадения.
        self.concurrent_pair_evidence: dict[tuple[int, int], tuple[int, int]] = {}
        self.concurrent_merge_frames = concurrent_merge_frames
        self.large_vehicle_evidence_max_gap = large_vehicle_evidence_max_gap
        self.interrupted_bus_min_smaller_coverage = interrupted_bus_min_smaller_coverage
        self.interrupted_bus_min_size_ratio = interrupted_bus_min_size_ratio
        self.interrupted_bus_min_appearance_similarity = (
            interrupted_bus_min_appearance_similarity
        )
        self.min_pixel_motion_frames = min_pixel_motion_frames
        self.max_missing_frames = max_missing_frames
        self.stitch_max_gap = stitch_max_gap
        self.stitch_distance_ratio = stitch_distance_ratio
        self.stitch_appearance_threshold = stitch_appearance_threshold
        self.turn_recovery_appearance_threshold = turn_recovery_appearance_threshold

    def update(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int],
        frame_number: int,
    ) -> None:
        """Обновляет состояния машин по обнаружениям текущего кадра."""
        width, height = frame_size
        frame_diagonal = math.hypot(width, height)
        frame_area = float(width * height)

        # Машины, видимые на текущем кадре, не считаются потерянными.
        seen_vehicle_ids: set[int] = set()
        # Один внутренний ID должен получить не более одного наблюдения за кадр.
        best_detections_by_vehicle_id: dict[int, Detection] = {}

        for detection in detections:
            tracker_id = detection.track_id

            if tracker_id is None:
                continue

            center = (
                (detection.x1 + detection.x2) / 2,
                (detection.y1 + detection.y2) / 2,
            )
            box: Box = (
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
            )

            vehicle_id = self.raw_to_vehicle.get(tracker_id)

            if vehicle_id is None:
                vehicle_id = self._find_stitch_candidate(
                    center=center,
                    box=box,
                    appearance_histogram=detection.appearance_histogram,
                    frame_number=frame_number,
                    frame_diagonal=frame_diagonal,
                    seen_vehicle_ids=seen_vehicle_ids,
                )

                if vehicle_id is None:
                    # Новый ID трекера временно считаем новой физической машиной.
                    vehicle_id = self.next_vehicle_id
                    self.next_vehicle_id += 1

                    self.raw_to_vehicle[tracker_id] = vehicle_id
                    self.vehicles[vehicle_id] = TrackState()
                    self.pending_recovery_vehicle_ids.add(vehicle_id)

                    logger.info(
                        "Создан кандидат: vehicle_id=%d tracker_id=%d frame=%d",
                        vehicle_id,
                        tracker_id,
                        frame_number,
                    )
                else:
                    self.raw_to_vehicle[tracker_id] = vehicle_id
                    logger.info(
                        (
                            "Новый ID трекера привязан к машине: "
                            "vehicle_id=%d tracker_id=%d frame=%d"
                        ),
                        vehicle_id,
                        tracker_id,
                        frame_number,
                    )

            seen_vehicle_ids.add(vehicle_id)

            current_best_detection = best_detections_by_vehicle_id.get(vehicle_id)

            if (
                current_best_detection is None
                or detection.confidence > current_best_detection.confidence
            ):
                best_detections_by_vehicle_id[vehicle_id] = detection

        for vehicle_id, detection in best_detections_by_vehicle_id.items():
            center_x = (detection.x1 + detection.x2) / 2
            center_y = (detection.y1 + detection.y2) / 2
            center = (center_x, center_y)

            box: Box = (detection.x1, detection.y1, detection.x2, detection.y2)

            state = self.vehicles[vehicle_id]

            state.add_observation(
                center,
                frame_number,
                box,
                detection.appearance_histogram,
                tracker_id=detection.track_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                crop=detection.crop,
                frame_area=frame_area,
                pixel_motion_ratio=detection.pixel_motion_ratio,
                has_pixel_motion=detection.has_pixel_motion,
            )

            movement_ratio = state.displacement_ratio(frame_diagonal)

            # Засчитываем только достаточно долгий и действительно движущийся трек.
            if (
                state.frames_seen >= self.min_confirmed_frames
                and movement_ratio >= self.min_movement_ratio
                and state.pixel_motion_frames >= self.min_pixel_motion_frames
                and state.best_crop is not None
                and vehicle_id not in self.counted_vehicle_ids
            ):
                self.counted_vehicle_ids.add(vehicle_id)

                logger.info(
                    (
                        "Машина подтверждена: vehicle_id=%d "
                        "frames=%d movement_ratio=%.4f motion_frames=%d frame=%d"
                    ),
                    vehicle_id,
                    state.frames_seen,
                    movement_ratio,
                    state.pixel_motion_frames,
                    frame_number,
                )

        # Объединение может удалять ID из исходного множества во время проверки.
        pending_vehicle_ids_copy = self.pending_recovery_vehicle_ids.copy()

        for new_vehicle_id in pending_vehicle_ids_copy:
            if new_vehicle_id not in seen_vehicle_ids:
                continue

            state = self.vehicles[new_vehicle_id]

            if state.frames_seen < self.turn_recovery_confirm_frames:
                continue

            recovery_candidate_id = self._find_recovery_candidate(
                new_vehicle_id, frame_diagonal, seen_vehicle_ids
            )

            if recovery_candidate_id is not None:
                recovery_state = self.vehicles[recovery_candidate_id]
                recovery_gap = state.first_frame - recovery_state.last_seen_frame
                predicted_center = recovery_state.predicted_center(state.first_frame)
                recovery_distance = math.dist(
                    predicted_center or recovery_state.last_center,
                    state.first_center,
                )
                recovery_distance_ratio = recovery_distance / frame_diagonal

                self._merge_vehicle_ids(recovery_candidate_id, new_vehicle_id)

                logger.info(
                    (
                        "Восстановлена потерянная машина: vehicle_id=%d "
                        "removed_vehicle_id=%d gap=%d "
                        "distance_ratio=%.4f frame=%d"
                    ),
                    recovery_candidate_id,
                    new_vehicle_id,
                    recovery_gap,
                    recovery_distance_ratio,
                    frame_number,
                )

                seen_vehicle_ids.discard(new_vehicle_id)
                seen_vehicle_ids.add(recovery_candidate_id)
                self.pending_recovery_vehicle_ids.discard(new_vehicle_id)

        # Одного совпадения недостаточно: оно может возникнуть при случайном пересечении рамок.
        concurrent_merge_candidates: list[tuple[int, int]] = []

        for first_vehicle_id, second_vehicle_id in combinations(
            sorted(seen_vehicle_ids), 2
        ):
            first_state = self.vehicles[first_vehicle_id]
            second_state = self.vehicles[second_vehicle_id]

            tracks_match = concurrent_tracks_match(first_state, second_state)

            if not tracks_match:
                continue

            pair = (first_vehicle_id, second_vehicle_id)

            previous_count, previous_frame = self.concurrent_pair_evidence.get(
                pair, (0, -1)
            )

            evidence_max_gap = (
                self.large_vehicle_evidence_max_gap
                if self._allows_interrupted_bus_evidence(first_state, second_state)
                else 1
            )

            if previous_frame >= frame_number - evidence_max_gap:
                consecutive_count = previous_count + 1
            else:
                consecutive_count = 1

            self.concurrent_pair_evidence[pair] = (consecutive_count, frame_number)

            if consecutive_count >= self.concurrent_merge_frames:
                concurrent_merge_candidates.append(pair)

        for first_vehicle_id, second_vehicle_id in concurrent_merge_candidates:
            if (
                first_vehicle_id not in self.vehicles
                or second_vehicle_id not in self.vehicles
            ):
                continue

            first_state = self.vehicles[first_vehicle_id]
            second_state = self.vehicles[second_vehicle_id]

            # Сохраняем ID трека, который появился раньше и содержит более длинную историю.
            if first_state.first_frame <= second_state.first_frame:
                primary_vehicle_id = first_vehicle_id
                secondary_vehicle_id = second_vehicle_id
            else:
                primary_vehicle_id = second_vehicle_id
                secondary_vehicle_id = first_vehicle_id

            self._merge_vehicle_ids(
                primary_vehicle_id,
                secondary_vehicle_id,
            )

            logger.info(
                (
                    "Объединены одновременные треки: vehicle_id=%d "
                    "removed_vehicle_id=%d evidence_frames=%d frame=%d"
                ),
                primary_vehicle_id,
                secondary_vehicle_id,
                self.concurrent_merge_frames,
                frame_number,
            )

            seen_vehicle_ids.discard(secondary_vehicle_id)
            seen_vehicle_ids.add(primary_vehicle_id)

        stale_pairs: list[tuple[int, int]] = []

        for pair, (_, last_match_frame) in self.concurrent_pair_evidence.items():
            first_vehicle_id, second_vehicle_id = pair

            if (
                first_vehicle_id not in self.vehicles
                or second_vehicle_id not in self.vehicles
            ):
                stale_pairs.append(pair)
                continue

            evidence_max_gap = (
                self.large_vehicle_evidence_max_gap
                if self._allows_interrupted_bus_evidence(
                    self.vehicles[first_vehicle_id],
                    self.vehicles[second_vehicle_id],
                )
                else 1
            )

            if last_match_frame < frame_number - evidence_max_gap:
                stale_pairs.append(pair)

        for pair in stale_pairs:
            self.concurrent_pair_evidence.pop(pair)

        self._finalize_stale_tracks(frame_number)

    def get_total(self) -> int:
        """Возвращает количество подтверждённых транспортных средств."""
        return len(self.counted_vehicle_ids)

    def _allows_interrupted_bus_evidence(
        self,
        first_state: TrackState,
        second_state: TrackState,
    ) -> bool:
        """Проверяет, могут ли рамки быть большой и малой частями одного автобуса."""
        first_class = first_state.stable_class_name()
        second_class = second_state.stable_class_name()

        if first_class != "bus" or second_class != "bus":
            return False

        if first_state.last_box is None or second_state.last_box is None:
            return False

        smaller_coverage = calculate_smaller_coverage(
            first_state.last_box,
            second_state.last_box,
        )
        size_ratio = max(
            first_state.last_box_area,
            second_state.last_box_area,
        ) / max(
            min(first_state.last_box_area, second_state.last_box_area),
            1.0,
        )
        appearance_similarity = calculate_histogram_similarity(
            first_state.appearance_histogram,
            second_state.appearance_histogram,
        )

        return (
            smaller_coverage >= self.interrupted_bus_min_smaller_coverage
            and size_ratio >= self.interrupted_bus_min_size_ratio
            and appearance_similarity >= self.interrupted_bus_min_appearance_similarity
        )

    def _find_recovery_candidate(
        self,
        new_vehicle_id: int,
        frame_diagonal: float,
        seen_vehicle_ids: set[int],
    ) -> int | None:
        """Ищет единственную недавно потерянную машину для нового трека."""
        candidates: list[int] = []

        if frame_diagonal <= 0:
            return

        new_state = self.vehicles[new_vehicle_id]

        if new_state.frames_seen < self.turn_recovery_confirm_frames:
            return

        if new_state.first_center is None:
            return

        # Видимая сейчас машина не может быть потерянным предшественником нового трека.
        for old_vehicle_id, old_state in self.vehicles.items():
            if old_state.frames_seen < self.turn_recovery_confirm_frames:
                continue

            if old_state.finalized or old_vehicle_id in seen_vehicle_ids:
                continue

            if old_vehicle_id == new_vehicle_id:
                continue

            gap = new_state.first_frame - old_state.last_seen_frame

            if gap <= 0 or gap > self.turn_recovery_max_gap:
                continue

            if old_state.last_center is None or old_state.last_box_area <= 0.0:
                continue

            predicted_center = old_state.predicted_center(new_state.first_frame)
            direct_distance = math.dist(
                old_state.last_center,
                new_state.first_center,
            )
            predicted_distance = (
                math.dist(predicted_center, new_state.first_center)
                if predicted_center is not None
                else direct_distance
            )
            distance_ratio = min(direct_distance, predicted_distance) / frame_diagonal

            if distance_ratio > self.turn_recovery_distance_ratio:
                continue

            size_ratio = max(
                old_state.last_box_area,
                new_state.last_box_area,
            ) / max(min(old_state.last_box_area, new_state.last_box_area), 1.0)

            if size_ratio > 4.0:
                continue

            appearance_similarity = calculate_histogram_similarity(
                old_state.appearance_histogram,
                new_state.appearance_histogram,
            )

            if appearance_similarity < self.turn_recovery_appearance_threshold:
                continue

            candidates.append(old_vehicle_id)

        if len(candidates) == 1:
            return candidates[0]

        # При неоднозначности не объединяем треки, чтобы не склеить разные машины.
        return None

    def _find_stitch_candidate(
        self,
        center: tuple[float, float],
        box: Box,
        appearance_histogram: np.ndarray | None,
        frame_number: int,
        frame_diagonal: float,
        seen_vehicle_ids: set[int],
    ) -> int | None:
        """Ищет недавно потерянную машину для нового ID трекера."""
        if frame_diagonal <= 0.0:
            return None

        box_area = calculate_box_area(box)
        best_vehicle_id: int | None = None
        best_score = 0.0

        for vehicle_id, state in self.vehicles.items():
            if state.finalized or vehicle_id in seen_vehicle_ids:
                continue

            gap = frame_number - state.last_seen_frame

            if gap <= 0 or gap > self.stitch_max_gap:
                continue

            if (
                state.last_box is None
                or state.last_center is None
                or state.last_box_area <= 0.0
            ):
                continue

            predicted_center = state.predicted_center(frame_number)

            if predicted_center is None:
                continue

            predicted_distance = math.dist(predicted_center, center)
            direct_distance = math.dist(state.last_center, center)
            distance_ratio = min(predicted_distance, direct_distance) / frame_diagonal
            overlap = calculate_iou(state.last_box, box)
            appearance_similarity = calculate_histogram_similarity(
                state.appearance_histogram,
                appearance_histogram,
            )
            size_ratio = max(state.last_box_area, box_area) / max(
                min(state.last_box_area, box_area),
                1.0,
            )

            spatial_match = (
                overlap >= 0.05 or distance_ratio <= self.stitch_distance_ratio * 0.5
            )

            if not spatial_match:
                continue

            if distance_ratio > self.stitch_distance_ratio or size_ratio > 3.0:
                continue

            if appearance_similarity < self.stitch_appearance_threshold:
                continue

            proximity_score = 1.0 - min(
                distance_ratio / self.stitch_distance_ratio,
                1.0,
            )
            size_score = 1.0 - min((size_ratio - 1.0) / 2.0, 1.0)
            score = (
                0.35 * overlap
                + 0.35 * appearance_similarity
                + 0.20 * proximity_score
                + 0.10 * size_score
            )

            if score > best_score:
                best_score = score
                best_vehicle_id = vehicle_id

        return best_vehicle_id

    def _finalize_stale_tracks(self, frame_number: int) -> None:
        """Закрывает состояния, которые давно не появлялись в кадре."""
        for vehicle_id, state in self.vehicles.items():
            missing_frames = frame_number - state.last_seen_frame

            if state.finalized or missing_frames < self.max_missing_frames:
                continue

            state.finalized = True
            self.pending_recovery_vehicle_ids.discard(vehicle_id)
            logger.info(
                "Трек завершён: vehicle_id=%d missing_frames=%d frame=%d",
                vehicle_id,
                missing_frames,
                frame_number,
            )

    def finalize_all(self) -> None:
        """Закрывает оставшиеся активные состояния после окончания видео."""
        for state in self.vehicles.values():
            state.finalized = True

    def _merge_vehicle_states(
        self, primary_state: TrackState, secondary_state: TrackState
    ) -> None:
        """Переносит историю нового трека в состояние ранее потерянной машины."""
        primary_frames_before = primary_state.frames_seen
        secondary_frames_before = secondary_state.frames_seen
        tracks_overlap = not (
            primary_state.last_seen_frame < secondary_state.first_frame
            or secondary_state.last_seen_frame < primary_state.first_frame
        )

        if secondary_state.first_frame < primary_state.first_frame:
            primary_state.first_frame = secondary_state.first_frame
            primary_state.first_center = secondary_state.first_center

        if secondary_state.last_seen_frame >= primary_state.last_seen_frame:
            primary_state.last_seen_frame = secondary_state.last_seen_frame
            primary_state.last_center = secondary_state.last_center
            primary_state.last_box = secondary_state.last_box
            primary_state.last_box_area = secondary_state.last_box_area

        displacement_candidates = [
            primary_state.max_displacement_pixels,
            secondary_state.max_displacement_pixels,
        ]

        if primary_state.first_center is not None:
            if secondary_state.first_center is not None:
                distance_to_secondary_first = math.dist(
                    primary_state.first_center,
                    secondary_state.first_center,
                )
                displacement_candidates.append(distance_to_secondary_first)

            if secondary_state.last_center is not None:
                distance_to_secondary_last = math.dist(
                    primary_state.first_center,
                    secondary_state.last_center,
                )
                displacement_candidates.append(distance_to_secondary_last)

        primary_state.max_displacement_pixels = max(displacement_candidates)

        if tracks_overlap:
            primary_state.pixel_motion_frames = max(
                primary_state.pixel_motion_frames,
                secondary_state.pixel_motion_frames,
            )
        else:
            primary_state.pixel_motion_frames += secondary_state.pixel_motion_frames

        primary_state.last_pixel_motion_frame = max(
            primary_state.last_pixel_motion_frame,
            secondary_state.last_pixel_motion_frame,
        )
        primary_state.max_pixel_motion_ratio = max(
            primary_state.max_pixel_motion_ratio,
            secondary_state.max_pixel_motion_ratio,
        )

        for class_name, score in secondary_state.class_scores.items():
            primary_state.class_scores[class_name] += score

        for class_name, confidence_sum in secondary_state.confidence_sums.items():
            primary_state.confidence_sums[class_name] += confidence_sum

        for class_name, confidence_count in secondary_state.confidence_counts.items():
            primary_state.confidence_counts[class_name] += confidence_count

        for tracker_id, frame_count in secondary_state.tracker_id_frames.items():
            primary_state.tracker_id_frames[tracker_id] += frame_count

        total_frames = primary_frames_before + secondary_frames_before

        # Более длинный трек сильнее влияет на объединённое описание внешнего вида.
        if secondary_state.appearance_histogram is not None:
            if primary_state.appearance_histogram is None:
                primary_state.appearance_histogram = (
                    secondary_state.appearance_histogram.copy()
                )
            elif total_frames > 0:
                primary_state.appearance_histogram = (
                    primary_state.appearance_histogram * primary_frames_before
                    + secondary_state.appearance_histogram * secondary_frames_before
                ) / total_frames

                cv2.normalize(
                    primary_state.appearance_histogram,
                    primary_state.appearance_histogram,
                    alpha=1.0,
                    norm_type=cv2.NORM_L1,
                )

        # Для общего кадра усредняем два центра, а не сохраняем дубли наблюдений.
        centers_by_frame: dict[int, list[tuple[float, float]]] = {}

        for observed_frame, center in (
            *primary_state.recent_centers,
            *secondary_state.recent_centers,
        ):
            centers_by_frame.setdefault(observed_frame, []).append(center)

        merged_recent_centers: list[tuple[int, tuple[float, float]]] = []

        for observed_frame in sorted(centers_by_frame):
            centers = centers_by_frame[observed_frame]

            average_center_x = sum(center[0] for center in centers) / len(centers)
            average_center_y = sum(center[1] for center in centers) / len(centers)

            merged_recent_centers.append(
                (
                    observed_frame,
                    (average_center_x, average_center_y),
                )
            )

        primary_state.recent_centers.clear()
        primary_state.recent_centers.extend(merged_recent_centers[-8:])
        primary_state.frames_seen = total_frames

        if secondary_state.best_crop_score > primary_state.best_crop_score:
            primary_state.best_crop = secondary_state.best_crop
            primary_state.best_crop_score = secondary_state.best_crop_score

        primary_state.finalized = primary_state.finalized and secondary_state.finalized

    def _merge_vehicle_ids(
        self, primary_vehicle_id: int, secondary_vehicle_id: int
    ) -> None:
        """Переносит все связи вторичного ID на основной и удаляет вторичный ID."""
        if primary_vehicle_id == secondary_vehicle_id:
            return

        primary_state = self.vehicles[primary_vehicle_id]
        secondary_state = self.vehicles[secondary_vehicle_id]
        self._merge_vehicle_states(primary_state, secondary_state)

        for tracker_id, vehicle_id in self.raw_to_vehicle.items():
            if vehicle_id == secondary_vehicle_id:
                self.raw_to_vehicle[tracker_id] = primary_vehicle_id

        was_counted = (
            primary_vehicle_id in self.counted_vehicle_ids
            or secondary_vehicle_id in self.counted_vehicle_ids
        )

        if was_counted:
            self.counted_vehicle_ids.add(primary_vehicle_id)
            self.counted_vehicle_ids.discard(secondary_vehicle_id)

        self.pending_recovery_vehicle_ids.discard(secondary_vehicle_id)

        for pair in list(self.concurrent_pair_evidence):
            if secondary_vehicle_id in pair:
                self.concurrent_pair_evidence.pop(pair)

        self.vehicles.pop(secondary_vehicle_id)
