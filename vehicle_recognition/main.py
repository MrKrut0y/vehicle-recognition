import argparse
from pathlib import Path

from vehicle_recognition.file_source import FileSource
from vehicle_recognition.detector import VehicleDetector


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

        for detection in detections:
            print(f"Класс: {detection.class_name}")
            print(f"Уверенность: {detection.confidence:.2f}")
            print(
                f"Координаты: "
                f"{detection.x1}, {detection.y1}, "
                f"{detection.x2}, {detection.y2}"
            )
            print(f"ID: {detection.class_id}")

        # На текущем этапе обрабатываем только первый кадр.
        break


if __name__ == "__main__":
    main()
