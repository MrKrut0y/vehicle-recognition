import argparse
import logging
import time
from pathlib import Path

import cv2

from vehicle_recognition.counter import VehicleCounter
from vehicle_recognition.detector import VehicleDetector
from vehicle_recognition.file_source import FileSource
from vehicle_recognition.logging_config import configure_logging
from vehicle_recognition.progress import build_progress_message, print_progress
from vehicle_recognition.video_writer import VideoFileWriter
from vehicle_recognition.visualizer import draw_detections

logger = logging.getLogger(__name__)


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

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Выводить подробный журнал обработки.",
    )

    return parser.parse_args()


def main() -> None:
    """Запускает чтение видео и распознавание транспорта."""
    arguments = parse_arguments()

    configure_logging(arguments.verbose)

    logger.info(
        "Запуск обработки: source=%s output=%s show=%s",
        arguments.source,
        arguments.output,
        arguments.show,
    )

    file_source = FileSource(arguments.source)

    model_path = "yolov8s.pt"
    detector = VehicleDetector(model_path)

    counter = VehicleCounter()

    video_writer: VideoFileWriter | None = None

    processed_frames = 0
    started_at = time.perf_counter()

    try:
        for frame_number, frame in enumerate(file_source.generate_frames(), start=1):
            processed_frames = frame_number

            detections = detector.detect(frame)

            height, width = frame.shape[:2]
            frame_size = (width, height)

            counter.update(detections, frame_size, frame_number)
            total = counter.get_total()

            is_first_frame = frame_number == 1
            is_update_frame = frame_number % 10 == 0
            is_last_frame = frame_number == file_source.total_frames

            if is_first_frame or is_update_frame or is_last_frame:
                elapsed_seconds = time.perf_counter() - started_at

                if arguments.verbose:
                    progress_message = build_progress_message(
                        frame_number,
                        file_source.total_frames,
                        total,
                        elapsed_seconds,
                    )
                    logger.info("Прогресс: %s", progress_message)
                else:
                    print_progress(
                        frame_number,
                        file_source.total_frames,
                        total,
                        elapsed_seconds,
                    )

            annotated_frame = draw_detections(frame, detections, total)

            if arguments.output is not None:
                # Размер выходного видео становится известен после первого кадра.
                if video_writer is None:
                    height, width = annotated_frame.shape[:2]

                    fps = file_source.fps
                    if fps is None or fps <= 0:
                        # Некоторые источники не сообщают FPS, поэтому используем запасное значение.
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
        # Освобождаем ресурсы даже при ошибке или досрочном завершении обработки.
        if video_writer is not None:
            video_writer.release()

        if arguments.show:
            cv2.destroyAllWindows()

    elapsed_seconds = time.perf_counter() - started_at
    average_fps = processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0

    logger.info(
        (
            "Обработка завершена: frames=%d vehicles=%d "
            "elapsed=%.2fs average_fps=%.2f output=%s"
        ),
        processed_frames,
        counter.get_total(),
        elapsed_seconds,
        average_fps,
        arguments.output,
    )

    if not arguments.verbose:
        print()

    print(f"Итоговое количество транспортных средств: {counter.get_total()}")


if __name__ == "__main__":
    main()
