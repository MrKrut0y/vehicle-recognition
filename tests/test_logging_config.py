import logging

from vehicle_recognition.logging_config import configure_logging


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
