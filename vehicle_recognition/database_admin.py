import argparse
from pathlib import Path

from vehicle_recognition.database import MySQLRepository
from vehicle_recognition.image_cleanup import delete_result_images


def positive_int(value: str) -> int:
    """Преобразует положительное число дней из командной строки."""
    parsed_value = int(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("days must be greater than zero")

    return parsed_value


def parse_arguments() -> argparse.Namespace:
    """Получает команду обслуживания базы данных."""
    parser = argparse.ArgumentParser(
        description="Обслуживание базы данных распознавания транспорта."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clear_parser = subparsers.add_parser(
        "clear-results",
        help="Удалить результаты обработки, сохранив камеры.",
    )
    clear_parser.add_argument(
        "--yes",
        action="store_true",
        help="Не запрашивать интерактивное подтверждение.",
    )
    clear_parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("results/vehicle_images"),
        help="Каталог сохранённых снимков машин.",
    )

    old_results_parser = subparsers.add_parser(
        "clear-old-results",
        help="Удалить завершённые результаты старше указанного срока.",
    )
    old_results_parser.add_argument(
        "--days",
        type=positive_int,
        required=True,
        help="Срок хранения завершённых результатов в днях.",
    )
    old_results_parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("results/vehicle_images"),
        help="Каталог сохранённых снимков машин.",
    )
    old_results_parser.add_argument(
        "--yes",
        action="store_true",
        help="Не запрашивать интерактивное подтверждение.",
    )

    return parser.parse_args()


def confirm_clear(message: str) -> bool:
    """Запрашивает явное подтверждение необратимого удаления."""
    print(message)
    return input("Type CLEAR to continue: ").strip() == "CLEAR"


def main() -> None:
    """Выполняет выбранную административную команду."""
    arguments = parse_arguments()

    if arguments.command == "clear-results":
        if not arguments.yes and not confirm_clear(
            "This will permanently delete all processing results."
        ):
            print("Database cleanup cancelled.")
            return

        summary = MySQLRepository.from_environment().clear_results()
        deleted_images = delete_result_images(summary.image_paths, arguments.images_dir)
        print(
            "Database results cleared: "
            f"runs={summary.processing_runs}, "
            f"observations={summary.vehicle_observations}, "
            f"tracker_ids={summary.tracker_ids}, images={deleted_images}."
        )
        print("Camera records were preserved.")

    if arguments.command == "clear-old-results":
        if not arguments.yes and not confirm_clear(
            f"This will delete processing results older than {arguments.days} days."
        ):
            print("Database cleanup cancelled.")
            return

        summary = MySQLRepository.from_environment().clear_results_older_than(
            arguments.days
        )
        deleted_images = delete_result_images(summary.image_paths, arguments.images_dir)
        print(
            "Expired database results cleared: "
            f"runs={summary.processing_runs}, "
            f"observations={summary.vehicle_observations}, "
            f"tracker_ids={summary.tracker_ids}, images={deleted_images}."
        )
        print("Recent results, active runs, and camera records were preserved.")


if __name__ == "__main__":
    main()
