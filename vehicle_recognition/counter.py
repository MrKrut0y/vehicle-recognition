from vehicle_recognition.detection import Detection


class VehicleCounter:
    """Считает транспортные средства по уникальным идентификаторам."""

    def __init__(self):
        self.counted_track_ids: set[int] = set()

    def update(self, detections: list[Detection]) -> None:
        for detection in detections:
            if detection.track_id is not None:
                self.counted_track_ids.add(detection.track_id)

    def get_total(self) -> int:
        return len(self.counted_track_ids)
