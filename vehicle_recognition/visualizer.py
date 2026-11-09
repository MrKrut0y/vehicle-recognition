import cv2

from vehicle_recognition.detection import Detection

CLASS_COLORS = {
    2: (0, 255, 0),  # car — зелёный
    3: (0, 255, 255),  # motorcycle — жёлтый
    5: (255, 0, 0),  # bus — синий
    7: (0, 0, 255),  # truck — красный
}


def draw_detections(frame, detections: list[Detection]):
    """Рисует рамки и подписи обнаруженных транспортных средств на кадре."""
    annotated_frame = frame.copy()

    for detection in detections:
        # Прямоугольники
        point1 = (int(detection.x1), int(detection.y1))
        point2 = (int(detection.x2), int(detection.y2))

        color = CLASS_COLORS.get(detection.class_id, (255, 255, 255))

        cv2.rectangle(annotated_frame, point1, point2, color, 2)

        # Подписи
        label = f"{detection.class_name} {detection.confidence:.2f}"

        text_position = (
            int(detection.x1),
            max(int(detection.y1) - 10, 20),
        )

        cv2.putText(
            annotated_frame,
            label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated_frame
