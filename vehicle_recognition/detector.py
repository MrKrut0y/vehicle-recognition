from ultralytics import YOLO

from vehicle_recognition.appearance import calculate_appearance_histogram
from vehicle_recognition.detection import Detection

# 2 — car, 3 — motorcycle, 5 — bus, 7 — truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]


class VehicleDetector:
    """Обнаруживает и отслеживает транспортные средства на кадре с помощью YOLO."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame) -> list[Detection]:
        """Возвращает обнаруженные транспортные средства с идентификаторами."""
        results = self.model.track(
            source=frame,
            verbose=False,
            classes=VEHICLE_CLASS_IDS,
            # Удаляет совпадающие рамки, даже если YOLO назначила им разные классы.
            agnostic_nms=True,
            # Сохраняет состояние трекера между последовательными кадрами.
            persist=True,
            tracker="tracker_static_reid.yaml",
        )

        result = results[0]
        detections = []

        frame_height, frame_width = frame.shape[:2]

        for box in result.boxes:

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0].item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            confidence = box.conf[0].item()

            # Координаты YOLO вещественные и могут выходить за границы кадра.
            crop_x1 = min(max(int(x1), 0), frame_width)
            crop_y1 = min(max(int(y1), 0), frame_height)
            crop_x2 = min(max(int(x2), 0), frame_width)
            crop_y2 = min(max(int(y2), 0), frame_height)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            appearance_histogram = calculate_appearance_histogram(crop)

            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                track_id=track_id,
                appearance_histogram=appearance_histogram,
            )

            detections.append(detection)

        return detections
