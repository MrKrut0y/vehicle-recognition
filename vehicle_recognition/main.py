import argparse
import logging
import time
from pathlib import Path

import cv2

from vehicle_recognition.counter import VehicleCounter
from vehicle_recognition.database import MySQLRepository
from vehicle_recognition.detector import VehicleDetector
from vehicle_recognition.file_source import FileSource
from vehicle_recognition.image_writer import save_vehicle_images
from vehicle_recognition.logging_config import configure_logging
from vehicle_recognition.motion import PixelMotionAnalyzer
from vehicle_recognition.progress import build_progress_message, print_progress
from vehicle_recognition.stream_source import StreamSource, is_stream_url
from vehicle_recognition.video_writer import VideoFileWriter
from vehicle_recognition.visualizer import draw_detections

logger = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    """Преобразует положительный аргумент командной строки в число."""
    parsed_value = int(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")

    return parsed_value


def parse_arguments() -> argparse.Namespace:
    """Получает параметры запуска программы из командной строки."""
    parser = argparse.ArgumentParser(
        description="Обработка видео с транспортными средствами."
    )

    parser.add_argument(
        "source",
        help="Путь к видеофайлу или URL сетевой трансляции.",
    )

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

    parser.add_argument(
        "--save-to-database",
        action="store_true",
        help="Сохранять результаты обработки в MySQL.",
    )

    parser.add_argument(
        "--max-frames",
        type=positive_int,
        help="Остановить обработку после указанного числа кадров.",
    )

    return parser.parse_args()


def main() -> None:
    """Запускает чтение видео и распознавание транспорта."""
    arguments = parse_arguments()

    configure_logging(arguments.verbose, arguments.log_file)

    source_text = arguments.source
    source_is_stream = is_stream_url(source_text)
    source_type = "stream" if source_is_stream else "video"

    logger.info("Source: %s", source_text)

    if source_is_stream:
        media_source = StreamSource(source_text)
        detector_source: Path | None = None
        database_source_name = source_text
    else:
        detector_source = Path(source_text)
        media_source = FileSource(detector_source)
        database_source_name = str(detector_source.resolve())

    if not media_source.read_metadata():
        return

    progress_total_frames = media_source.total_frames
    if arguments.max_frames is not None:
        if progress_total_frames is None:
            progress_total_frames = arguments.max_frames
        else:
            progress_total_frames = min(
                progress_total_frames,
                arguments.max_frames,
            )

    logger.info(
        "Frames: %s | FPS: %.2f",
        progress_total_frames if progress_total_frames else "unknown",
        media_source.fps or 0.0,
    )
    logger.info("Tracker: tracker_static_reid.yaml")
    logger.info(
        "Output: video=%s images=%s log=%s database=%s show=%s",
        arguments.output,
        arguments.images_dir,
        arguments.log_file,
        arguments.save_to_database,
        arguments.show,
    )

    model_path = "yolov8s.pt"
    detector = VehicleDetector(model_path)
    motion_analyzer = PixelMotionAnalyzer()

    counter = VehicleCounter()

    database_repository: MySQLRepository | None = None
    database_run_id: int | None = None

    if arguments.save_to_database:
        database_repository = MySQLRepository.from_environment()

        if database_repository.config.retention_days is not None:
            cleanup_summary = database_repository.clear_results_older_than(
                database_repository.config.retention_days
            )
            logger.info(
                (
                    "Database retention applied: days=%d runs=%d "
                    "observations=%d tracker_ids=%d"
                ),
                database_repository.config.retention_days,
                cleanup_summary.processing_runs,
                cleanup_summary.vehicle_observations,
                cleanup_summary.tracker_ids,
            )

        database_run_id = database_repository.create_run(
            source_type=source_type,
            source_name=database_source_name,
            total_frames=progress_total_frames,
            source_fps=media_source.fps,
        )
        logger.info("Database run created: run_id=%d", database_run_id)

    video_writer: VideoFileWriter | None = None

    processed_frames = 0
    started_at = time.perf_counter()
    run_cancelled = False

    if source_is_stream:
        tracked_frames = (
            (frame, detector.detect(frame)) for frame in media_source.generate_frames()
        )
    else:
        if detector_source is None:
            raise RuntimeError("Local video path was not initialized")
        tracked_frames = detector.track_video(detector_source)

    try:
        for frame_number, (frame, detections) in enumerate(tracked_frames, start=1):
            processed_frames = frame_number

            motion_analyzer.update(frame, detections, frame_number)

            height, width = frame.shape[:2]
            frame_size = (width, height)

            counter.update(detections, frame_size, frame_number)
            total = counter.get_total()
            stitched_count = counter.get_stitched_id_count()

            is_first_frame = frame_number == 1
            is_update_frame = frame_number % 10 == 0
            is_last_frame = frame_number == progress_total_frames

            if is_first_frame or is_update_frame or is_last_frame:
                elapsed_seconds = time.perf_counter() - started_at

                progress_message = build_progress_message(
                    frame_number,
                    progress_total_frames,
                    total,
                    elapsed_seconds,
                    stitched_count,
                )
                logger.info("%s", progress_message)

                if not arguments.verbose:
                    print_progress(
                        frame_number,
                        progress_total_frames,
                        total,
                        elapsed_seconds,
                        stitched_count,
                    )

            annotated_frame = draw_detections(frame, detections, total)

            if arguments.output is not None:
                # Размер выходного видео становится известен после первого кадра.
                if video_writer is None:
                    height, width = annotated_frame.shape[:2]

                    fps = media_source.fps
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
                    run_cancelled = True
                    break
            if (
                arguments.max_frames is not None
                and frame_number >= arguments.max_frames
            ):
                logger.info("Frame limit reached: max_frames=%d", arguments.max_frames)
                break
    except KeyboardInterrupt:
        run_cancelled = True
        logger.info("Processing stopped by user")
    except Exception as error:
        if database_repository is not None and database_run_id is not None:
            try:
                database_repository.fail_run(
                    database_run_id,
                    processed_frames,
                    error,
                )
            except Exception:
                logger.exception(
                    "Could not mark database run as failed: run_id=%d",
                    database_run_id,
                )
        raise
    finally:
        close_tracked_frames = getattr(tracked_frames, "close", None)
        if close_tracked_frames is not None:
            close_tracked_frames()

        # Освобождаем ресурсы даже при ошибке или досрочном завершении обработки.
        if video_writer is not None:
            video_writer.release()

        if arguments.show:
            cv2.destroyAllWindows()

    try:
        elapsed_seconds = time.perf_counter() - started_at
        average_fps = processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0

        completion_reason = "stream_stopped" if source_is_stream else "video_ended"
        counter.finalize_all(completion_reason)

        image_output_directory = arguments.images_dir
        if database_run_id is not None:
            image_output_directory /= f"run_{database_run_id:06d}"

        saved_image_paths = save_vehicle_images(
            image_output_directory,
            counter.counted_vehicle_ids,
            counter.vehicles,
        )

        if database_repository is not None and database_run_id is not None:
            database_repository.complete_run(
                run_id=database_run_id,
                processed_frames=processed_frames,
                source_fps=media_source.fps or 30.0,
                vehicle_ids=counter.counted_vehicle_ids,
                vehicles=counter.vehicles,
                frame_diagonal=counter.frame_diagonal,
                image_paths=saved_image_paths,
                status="cancelled" if run_cancelled else "completed",
            )
            logger.info("Database run completed: run_id=%d", database_run_id)
    except Exception as error:
        if database_repository is not None and database_run_id is not None:
            try:
                database_repository.fail_run(
                    database_run_id,
                    processed_frames,
                    error,
                )
            except Exception:
                logger.exception(
                    "Could not mark database run as failed: run_id=%d",
                    database_run_id,
                )
        raise

    logger.info(
        (
            "Processing completed: frames=%d vehicles=%d "
            "elapsed=%.2fs average_fps=%.2f output=%s images=%d"
        ),
        processed_frames,
        counter.get_total(),
        elapsed_seconds,
        average_fps,
        arguments.output,
        len(saved_image_paths),
    )
    logger.info("Accepted observations: %d", counter.get_total())
    logger.info("Saved vehicle images: %d", len(saved_image_paths))

    if not arguments.verbose:
        print()

    print(f"Accepted observations: {counter.get_total()}")
    print(f"Saved vehicle images: {len(saved_image_paths)}")


if __name__ == "__main__":
    main()
