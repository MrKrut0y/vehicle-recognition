import argparse
import logging
import time
from pathlib import Path

import cv2

from vehicle_recognition.counter import VehicleCounter
from vehicle_recognition.detector import VehicleDetector
from vehicle_recognition.file_source import FileSource
from vehicle_recognition.image_writer import save_vehicle_images
from vehicle_recognition.logging_config import configure_logging
from vehicle_recognition.motion import PixelMotionAnalyzer
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

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("results/vehicle_images"),
        help="Каталог для лучших снимков подтверждённых машин.",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("results/vehicle_recognition.log"),
        help="Файл для подробного журнала обработки.",
    )

    return parser.parse_args()


def main() -> None:
    """Запускает чтение видео и распознавание транспорта."""
    arguments = parse_arguments()

    configure_logging(arguments.verbose, arguments.log_file)

    logger.info(
        "Запуск обработки: source=%s output=%s images_dir=%s log_file=%s show=%s",
        arguments.source,
        arguments.output,
        arguments.images_dir,
        arguments.log_file,
        arguments.show,
    )

    file_source = FileSource(arguments.source)

    if not file_source.read_metadata():
        return

    model_path = "yolov8s.pt"
    detector = VehicleDetector(model_path)
    motion_analyzer = PixelMotionAnalyzer()

    counter = VehicleCounter()

    video_writer: VideoFileWriter | None = None

    processed_frames = 0
    started_at = time.perf_counter()

    try:
        for frame_number, (frame, detections) in enumerate(
            detector.track_video(arguments.source),
            start=1,
        ):
            processed_frames = frame_number

            motion_analyzer.update(frame, detections, frame_number)

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

    counter.finalize_all()
    saved_image_paths = save_vehicle_images(
        arguments.images_dir,
        counter.counted_vehicle_ids,
        counter.vehicles,
    )

    logger.info(
        (
            "Обработка завершена: frames=%d vehicles=%d "
            "elapsed=%.2fs average_fps=%.2f output=%s images=%d"
        ),
        processed_frames,
        counter.get_total(),
        elapsed_seconds,
        average_fps,
        arguments.output,
        len(saved_image_paths),
    )

    if not arguments.verbose:
        print()

    print(f"Итоговое количество транспортных средств: {counter.get_total()}")
    print(f"Сохранено снимков транспортных средств: {len(saved_image_paths)}")


if __name__ == "__main__":
    main()
