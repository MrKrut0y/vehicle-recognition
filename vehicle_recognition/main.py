import argparse
from pathlib import Path

from vehicle_recognition.file_source import FileSource


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
    for frame in file_source.generate_frames():
        frame_count += 1
    print(frame_count)


if __name__ == "__main__":
    main()
