import logging
from pathlib import Path


def configure_logging(verbose: bool, log_file: Path) -> None:
    """Настраивает вывод журнала в консоль и файл."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_handler.setFormatter(formatter)

    # Файл всегда содержит подробные события, даже без --verbose.
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Ultralytics мог настроить корневой логгер до запуска приложения.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
