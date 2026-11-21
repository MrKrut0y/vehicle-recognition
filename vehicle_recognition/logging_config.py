import logging


def configure_logging(verbose: bool) -> None:
    """Настраивает уровень и формат журналирования приложения."""
    log_level = logging.INFO if verbose else logging.WARNING

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        # Сторонние библиотеки могли настроить корневой логгер раньше приложения.
        force=True,
    )
