import argparse
import logging
import time
from pathlib import Path

import cv2

from vehicle_recognition.counter import VehicleCounter
from vehicle_recognition.database import MySQLRepository
from vehicle_recognition.database_retry import retry_database_operation
from vehicle_recognition.detector import VehicleDetector
from vehicle_recognition.file_source import FileSource
from vehicle_recognition.frame_pacer import FramePacer
from vehicle_recognition.image_cleanup import delete_result_images
from vehicle_recognition.image_writer import save_vehicle_images
from vehicle_recognition.logging_config import configure_logging
from vehicle_recognition.motion import PixelMotionAnalyzer
from vehicle_recognition.periodic_trigger import PeriodicTrigger
from vehicle_recognition.progress import build_progress_message, print_progress
from vehicle_recognition.result_persistence import persist_new_observations
from vehicle_recognition.stream_source import StreamSource, is_stream_url
from vehicle_recognition.video_writer import SegmentedVideoFileWriter, VideoFileWriter
from vehicle_recognition.visualizer import draw_detections

logger = logging.getLogger(__name__)
DATABASE_PROGRESS_UPDATE_INTERVAL = 100
DATABASE_RETENTION_INTERVAL_SECONDS = 24 * 60 * 60


def positive_int(value: str) -> int:
    """Преобразует положительный аргумент командной строки в число."""
    parsed_value = int(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")

    return parsed_value


def apply_database_retention(
    repository: MySQLRepository,
    image_root: Path,
) -> None:
    """Удаляет устаревшие результаты, если для базы настроен срок хранения."""
    retention_days = repository.config.retention_days

    if retention_days is None:
        return

    cleanup_summary = retry_database_operation(
        "apply retention",
        lambda: repository.clear_results_older_than(retention_days),
    )
    deleted_image_count = delete_result_images(
        cleanup_summary.image_paths,
        image_root,
    )
    logger.info(
        (
            "Database retention applied: days=%d runs=%d "
            "observations=%d tracker_ids=%d images=%d"
        ),
        retention_days,
        cleanup_summary.processing_runs,
        cleanup_summary.vehicle_observations,
        cleanup_summary.tracker_ids,
        deleted_image_count,
    )


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
        "--output-segment-minutes",
        type=positive_int,
        default=60,
        help="Длительность одного файла потокового видео в минутах.",
    )

    parser.add_argument(
        "--output-max-segments",
        type=positive_int,
        default=24,
        help="Максимальное число хранимых сегментов потокового видео.",
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
        "--camera-name",
        help="Имя сетевой камеры для сохранения её истории в MySQL.",
    )

    parser.add_argument(
        "--max-frames",
        type=positive_int,
        help="Остановить обработку после указанного числа кадров.",
    )

    arguments = parser.parse_args()
    source_is_stream = is_stream_url(arguments.source)

    if arguments.camera_name is not None and not source_is_stream:
        parser.error("--camera-name can only be used with a stream URL")

    if source_is_stream and arguments.save_to_database and not arguments.camera_name:
        parser.error("--camera-name is required for a database-backed stream")

    return arguments


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
    retention_trigger: PeriodicTrigger | None = None

    if arguments.save_to_database:
        database_repository = MySQLRepository.from_environment()

        recovered_run_count = retry_database_operation(
            "recover interrupted runs",
            database_repository.recover_interrupted_runs,
        )

        if recovered_run_count:
            logger.warning(
                "Recovered interrupted database runs: count=%d",
                recovered_run_count,
            )

        apply_database_retention(database_repository, arguments.images_dir)

        if database_repository.config.retention_days is not None:
            retention_trigger = PeriodicTrigger(DATABASE_RETENTION_INTERVAL_SECONDS)

        camera_id: int | None = None

        if source_is_stream:
            camera_id = retry_database_operation(
                "register camera",
                lambda: database_repository.get_or_create_camera(
                    name=arguments.camera_name,
                    stream_url=source_text,
                ),
            )
            logger.info(
                "Database camera registered: camera_id=%d name=%s",
                camera_id,
                arguments.camera_name,
            )

        database_run_id = retry_database_operation(
            "create processing run",
            lambda: database_repository.create_run(
                source_type=source_type,
                source_name=database_source_name,
                total_frames=progress_total_frames,
                source_fps=media_source.fps,
                camera_id=camera_id,
            ),
        )
        logger.info("Database run created: run_id=%d", database_run_id)

    video_writer: VideoFileWriter | SegmentedVideoFileWriter | None = None
    frame_pacer = FramePacer(media_source.fps or 30.0)
    persisted_vehicle_ids: set[int] = set()
    saved_image_paths: dict[int, Path] = {}
    image_output_directory = arguments.images_dir

    if database_run_id is not None:
        image_output_directory /= f"run_{database_run_id:06d}"

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

            if database_repository is not None and database_run_id is not None:
                new_image_paths = retry_database_operation(
                    "persist observations",
                    lambda: persist_new_observations(
                        repository=database_repository,
                        run_id=database_run_id,
                        counted_vehicle_ids=counter.counted_vehicle_ids,
                        vehicles=counter.vehicles,
                        persisted_vehicle_ids=persisted_vehicle_ids,
                        image_output_directory=image_output_directory,
                        source_fps=media_source.fps or 30.0,
                        frame_diagonal=counter.frame_diagonal,
                    ),
                )
                saved_image_paths.update(new_image_paths)

            total = counter.get_total()
            stitched_count = counter.get_stitched_id_count()

            if (
                database_repository is not None
                and database_run_id is not None
                and frame_number % DATABASE_PROGRESS_UPDATE_INTERVAL == 0
            ):
                retry_database_operation(
                    "update run progress",
                    lambda: database_repository.update_run_progress(
                        run_id=database_run_id,
                        processed_frames=frame_number,
                        accepted_observations=total,
                    ),
                )

            if (
                database_repository is not None
                and retention_trigger is not None
                and retention_trigger.is_due()
            ):
                apply_database_retention(database_repository, arguments.images_dir)

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

                    if source_is_stream:
                        video_writer = SegmentedVideoFileWriter(
                            output_path=arguments.output,
                            fps=fps,
                            frame_size=(width, height),
                            segment_duration_seconds=(
                                arguments.output_segment_minutes * 60
                            ),
                            max_segments=arguments.output_max_segments,
                        )
                    else:
                        video_writer = VideoFileWriter(
                            output_path=arguments.output,
                            fps=fps,
                            frame_size=(width, height),
                        )

                video_writer.write(annotated_frame)

            if arguments.show:
                cv2.imshow("Vehicle detection", annotated_frame)
                if cv2.waitKey(frame_pacer.next_delay_ms()) & 0xFF == ord("q"):
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
                retry_database_operation(
                    "mark run as failed",
                    lambda: database_repository.fail_run(
                        database_run_id,
                        processed_frames,
                        error,
                    ),
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

        if database_repository is not None and database_run_id is not None:
            new_image_paths = retry_database_operation(
                "persist final observations",
                lambda: persist_new_observations(
                    repository=database_repository,
                    run_id=database_run_id,
                    counted_vehicle_ids=counter.counted_vehicle_ids,
                    vehicles=counter.vehicles,
                    persisted_vehicle_ids=persisted_vehicle_ids,
                    image_output_directory=image_output_directory,
                    source_fps=media_source.fps or 30.0,
                    frame_diagonal=counter.frame_diagonal,
                ),
            )
            saved_image_paths.update(new_image_paths)

            retry_database_operation(
                "complete processing run",
                lambda: database_repository.complete_run(
                    run_id=database_run_id,
                    processed_frames=processed_frames,
                    accepted_observations=len(counter.counted_vehicle_ids),
                    status="cancelled" if run_cancelled else "completed",
                ),
            )
            logger.info("Database run completed: run_id=%d", database_run_id)
        else:
            saved_image_paths = save_vehicle_images(
                image_output_directory,
                counter.counted_vehicle_ids,
                counter.vehicles,
            )
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
