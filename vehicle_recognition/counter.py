import math
from itertools import combinations

import cv2

from vehicle_recognition.detection import Detection
from vehicle_recognition.geometry import Box
from vehicle_recognition.track_matching import concurrent_tracks_match
from vehicle_recognition.track_state import TrackState


class VehicleCounter:
    """Сопоставляет треки и считает подтверждённые движущиеся машины."""

    def __init__(
        self,
        min_confirmed_frames: int = 10,
        min_movement_ratio: float = 0.03,
        turn_recovery_max_gap: int = 60,
        turn_recovery_distance_ratio: float = 0.04,
        turn_recovery_confirm_frames: int = 5,
        concurrent_merge_frames: int = 5,
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

    def update(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int],
        frame_number: int,
    ) -> None:
        """Обновляет состояния машин по обнаружениям текущего кадра."""
        width, height = frame_size
        frame_diagonal = math.hypot(width, height)

        # Машины, видимые на текущем кадре, не считаются потерянными.
        seen_vehicle_ids: set[int] = set()
        # Один внутренний ID должен получить не более одного наблюдения за кадр.
        best_detections_by_vehicle_id: dict[int, Detection] = {}

        for detection in detections:
            tracker_id = detection.track_id

            if tracker_id is None:
                continue

            vehicle_id = self.raw_to_vehicle.get(tracker_id)

            if vehicle_id is None:
                # Новый ID трекера временно считаем новой физической машиной.
                vehicle_id = self.next_vehicle_id
                self.next_vehicle_id += 1

                self.raw_to_vehicle[tracker_id] = vehicle_id
                self.vehicles[vehicle_id] = TrackState()
                self.pending_recovery_vehicle_ids.add(vehicle_id)

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
            )

            movement_ratio = state.displacement_ratio(frame_diagonal)

            # Засчитываем только достаточно долгий и действительно движущийся трек.
            if (
                state.frames_seen >= self.min_confirmed_frames
                and movement_ratio >= self.min_movement_ratio
            ):
                self.counted_vehicle_ids.add(vehicle_id)

        # Объединение может удалять ID из исходного множества во время проверки.
        pending_vehicle_ids_copy = self.pending_recovery_vehicle_ids.copy()

        for new_vehicle_id in pending_vehicle_ids_copy:
            state = self.vehicles[new_vehicle_id]

            if state.frames_seen < self.turn_recovery_confirm_frames:
                continue

            recovery_candidate_id = self._find_recovery_candidate(
                new_vehicle_id, frame_diagonal, seen_vehicle_ids
            )

            if recovery_candidate_id is not None:
                self._merge_vehicle_ids(recovery_candidate_id, new_vehicle_id)
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

            if previous_frame == frame_number - 1:
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

            seen_vehicle_ids.discard(secondary_vehicle_id)
            seen_vehicle_ids.add(primary_vehicle_id)

        # Пропуск кадра разрывает последовательность подтверждений для пары.
        stale_pairs = [
            pair
            for pair, (_, last_match_frame) in self.concurrent_pair_evidence.items()
            if last_match_frame < frame_number
        ]

        for pair in stale_pairs:
            self.concurrent_pair_evidence.pop(pair)

    def get_total(self) -> int:
        """Возвращает количество подтверждённых транспортных средств."""
        return len(self.counted_vehicle_ids)

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

            if old_vehicle_id in seen_vehicle_ids:
                continue

            if old_vehicle_id == new_vehicle_id:
                continue

            gap = new_state.first_frame - old_state.last_seen_frame

            if gap <= 0 or gap > self.turn_recovery_max_gap:
                continue

            if old_state.last_center is None:
                continue

            distance = math.dist(old_state.last_center, new_state.first_center)
            distance_ratio = distance / frame_diagonal

            if distance_ratio <= self.turn_recovery_distance_ratio:
                candidates.append(old_vehicle_id)

        if len(candidates) == 1:
            return candidates[0]

        # При неоднозначности не объединяем треки, чтобы не склеить разные машины.
        return None

    def _merge_vehicle_states(
        self, primary_state: TrackState, secondary_state: TrackState
    ) -> None:
        """Переносит историю нового трека в состояние ранее потерянной машины."""
        primary_frames_before = primary_state.frames_seen
        secondary_frames_before = secondary_state.frames_seen
        primary_state.last_seen_frame = secondary_state.last_seen_frame
        primary_state.last_center = secondary_state.last_center
        primary_state.frames_seen += secondary_state.frames_seen
        primary_state.last_box = secondary_state.last_box

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
