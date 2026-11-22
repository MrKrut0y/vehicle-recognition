import logging
from logging.handlers import RotatingFileHandler

from vehicle_recognition.logging_config import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    configure_logging,
)


def test_configure_logging_writes_info_messages_to_file(tmp_path):
    log_file = tmp_path / "processing.log"

    configure_logging(verbose=False, log_file=log_file)
    logging.getLogger("vehicle_recognition.test").info("Test event")

    for handler in logging.getLogger().handlers:
        handler.flush()

    log_contents = log_file.read_text(encoding="utf-8")

    assert "INFO" in log_contents
    assert "vehicle_recognition.test" in log_contents
    assert "Test event" in log_contents


def test_configure_logging_limits_log_file_size(tmp_path):
    log_file = tmp_path / "processing.log"

    configure_logging(verbose=False, log_file=log_file)

    file_handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, RotatingFileHandler)
    ]

    assert len(file_handlers) == 1
    assert file_handlers[0].maxBytes == LOG_MAX_BYTES
    assert file_handlers[0].backupCount == LOG_BACKUP_COUNT
