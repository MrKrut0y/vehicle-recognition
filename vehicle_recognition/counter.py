import math

from vehicle_recognition.detection import Detection
from vehicle_recognition.track_state import TrackState


class VehicleCounter:
    """Считает транспортные средства по уникальным идентификаторам."""

    def __init__(
        self, min_confirmed_frames: int = 10, min_movement_ratio: float = 0.03
    ):
        self.min_confirmed_frames = min_confirmed_frames
        self.counted_track_ids: set[int] = set()
        self.track_states: dict[int, TrackState] = {}
        self.min_movement_ratio = min_movement_ratio

    def update(self, detections: list[Detection], frame_size: tuple[int, int]) -> None:
        width, height = frame_size
        diagonal = math.hypot(width, height)

        for detection in detections:
            if detection.track_id is None:
                continue

            state = self.track_states.get(detection.track_id)

            if state is None:
                state = TrackState()
                self.track_states[detection.track_id] = state

            center_x = (detection.x1 + detection.x2) / 2
            center_y = (detection.y1 + detection.y2) / 2

            center = (center_x, center_y)

            state.add_observation(center)
            displacement_ratio = state.displacement_ratio(diagonal)

            if (
                state.frames_seen >= self.min_confirmed_frames
                and displacement_ratio >= self.min_movement_ratio
            ):
                self.counted_track_ids.add(detection.track_id)

    def get_total(self) -> int:
        return len(self.counted_track_ids)
