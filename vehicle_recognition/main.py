import argparse
from pathlib import Path

from vehicle_recognition.file_source import FileSource
from vehicle_recognition.detector import VehicleDetector


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обработка видео с транспортными средствами."
    )

    parser.add_argument("source", type=Path, help="Путь к исходному видеофайлу.")

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    file_source = FileSource(arguments.source)

    frame_count = 0

    model_path = "yolov8s.pt"
    detector = VehicleDetector(model_path)

    for frame in file_source.generate_frames():
        result = detector.detect(frame)
        print(f"Количество обнаруженных объектов: {len(result.boxes)}")
        frame_count += 1
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            confidence = box.conf[0].item()
            print(f"Класс: {class_name}")
            print(f"Уверенность: {confidence:.2f}")
            print(f"Координаты: {coords}")
            print(f"ID: {class_id}")
        break
    print(f"Количество кадров: {frame_count}")


if __name__ == "__main__":
    main()
