from ultralytics import YOLO

# 2 — car, 3 — motorcycle, 5 — bus, 7 — truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]


class VehicleDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model.predict(
            source=frame, verbose=False, classes=VEHICLE_CLASS_IDS
        )

        return results[0]
