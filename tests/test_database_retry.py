import mysql.connector
import pytest

from vehicle_recognition.database_retry import retry_database_operation


def test_database_operation_retries_with_exponential_delay():
    attempts = iter(
        (
            mysql.connector.Error("first failure"),
            mysql.connector.Error("second failure"),
            "saved",
        )
    )
    delays: list[float] = []

    def operation():
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    result = retry_database_operation(
        "save",
        operation,
        sleep=delays.append,
    )

    assert result == "saved"
    assert delays == [2.0, 4.0]


def test_database_operation_stops_at_configured_attempt_limit():
    def operation():
        raise mysql.connector.Error("database unavailable")

    with pytest.raises(mysql.connector.Error):
        retry_database_operation(
            "save",
            operation,
            max_attempts=2,
            sleep=lambda _: None,
        )
