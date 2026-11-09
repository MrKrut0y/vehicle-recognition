import argparse
from pathlib import Path

import cv2

from vehicle_recognition.file_source import FileSource
from vehicle_recognition.detector import VehicleDetector
from vehicle_recognition.visualizer import draw_detections


def parse_arguments() -> argparse.Namespace:
    """Получает параметры запуска программы из командной строки."""
    parser = argparse.ArgumentParser(
        description="Обработка видео с транспортными средствами."
    )

    parser.add_argument("source", type=Path, help="Путь к исходному видеофайлу.")

    return parser.parse_args()


def main() -> None:
    """Запускает чтение видео и распознавание транспорта."""
    arguments = parse_arguments()
    file_source = FileSource(arguments.source)

    model_path = "yolov8s.pt"
    detector = VehicleDetector(model_path)

    for frame in file_source.generate_frames():
        detections = detector.detect(frame)

        print(f"Количество обнаруженных объектов: {len(detections)}")

        annotated_frame = draw_detections(frame, detections)

        cv2.imshow("Vehicle detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
