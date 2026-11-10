import argparse
from pathlib import Path

import cv2

from vehicle_recognition.counter import VehicleCounter
from vehicle_recognition.detector import VehicleDetector
from vehicle_recognition.file_source import FileSource
from vehicle_recognition.video_writer import VideoFileWriter
from vehicle_recognition.visualizer import draw_detections


def parse_arguments() -> argparse.Namespace:
    """Получает параметры запуска программы из командной строки."""
    parser = argparse.ArgumentParser(
        description="Обработка видео с транспортными средствами."
    )

    parser.add_argument("source", type=Path, help="Путь к исходному видеофайлу.")

    parser.add_argument(
        "--output", type=Path, help="Путь для сохранения обработанного видео."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Показывать обработанные кадры во время работы.",
    )

    return parser.parse_args()


def main() -> None:
    """Запускает чтение видео и распознавание транспорта."""
    arguments = parse_arguments()
    file_source = FileSource(arguments.source)

    model_path = "yolov8s.pt"
    detector = VehicleDetector(model_path)

    counter = VehicleCounter()

    video_writer: VideoFileWriter | None = None

    try:
        for frame in file_source.generate_frames():
            detections = detector.detect(frame)

            height, width = frame.shape[:2]
            frame_size = (width, height)

            counter.update(detections, frame_size)
            total = counter.get_total()

            annotated_frame = draw_detections(frame, detections, total)

            if arguments.output is not None:
                if video_writer is None:
                    height, width = annotated_frame.shape[:2]

                    fps = file_source.fps
                    if fps is None or fps <= 0:
                        fps = 30.0

                    video_writer = VideoFileWriter(
                        output_path=arguments.output,
                        fps=fps,
                        frame_size=(width, height),
                    )

                video_writer.write(annotated_frame)

            if arguments.show:
                cv2.imshow("Vehicle detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if video_writer is not None:
            video_writer.release()

        if arguments.show:
            cv2.destroyAllWindows()

    print(f"Итоговое количество транспортных средств: {counter.get_total()}")


if __name__ == "__main__":
    main()
