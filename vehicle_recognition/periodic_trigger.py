import time
from collections.abc import Callable


class PeriodicTrigger:
    """Сообщает, когда наступило время очередной периодической операции."""

    def __init__(
        self,
        interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")

        self.interval_seconds = interval_seconds
        self.clock = clock
        self.next_run_at = clock() + interval_seconds

    def is_due(self) -> bool:
        """Возвращает True один раз при наступлении очередного интервала."""
        current_time = self.clock()

        if current_time < self.next_run_at:
            return False

        self.next_run_at = current_time + self.interval_seconds
        return True
