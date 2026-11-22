import logging
import time
from collections.abc import Callable
from typing import TypeVar

import mysql.connector

logger = logging.getLogger(__name__)
Result = TypeVar("Result")


def retry_database_operation(
    operation_name: str,
    operation: Callable[[], Result],
    initial_delay_seconds: float = 2.0,
    max_delay_seconds: float = 60.0,
    max_attempts: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Result:
    """Повторяет операцию MySQL после временных ошибок соединения."""
    failed_attempts = 0

    while True:
        try:
            return operation()
        except mysql.connector.Error:
            failed_attempts += 1

            if max_attempts is not None and failed_attempts >= max_attempts:
                raise

            exponent = min(failed_attempts - 1, 10)
            delay_seconds = min(
                initial_delay_seconds * (2**exponent),
                max_delay_seconds,
            )
            logger.warning(
                "Database unavailable; retrying operation=%s attempt=%d delay=%.1fs",
                operation_name,
                failed_attempts,
                delay_seconds,
            )
            sleep(delay_seconds)
