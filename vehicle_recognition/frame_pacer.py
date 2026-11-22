import time
from collections.abc import Callable


class FramePacer:
    """Рассчитывает задержку для показа кадров с частотой исходного видео."""

    def __init__(
        self,
        fps: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        effective_fps = fps if fps > 0 else 30.0
        self.frame_interval = 1.0 / effective_fps
        self.clock = clock
        self.next_deadline: float | None = None

    def next_delay_ms(self) -> int:
        """Возвращает задержку до показа следующего кадра в миллисекундах."""
        current_time = self.clock()

        # После долгой сетевой паузы начинаем отсчёт заново, не ускоряя показ.
        if (
            self.next_deadline is None
            or current_time >= self.next_deadline + self.frame_interval
        ):
            self.next_deadline = current_time + self.frame_interval

        delay_seconds = max(self.next_deadline - current_time, 0.0)
        self.next_deadline += self.frame_interval

        # waitKey(0) ожидал бы нажатия клавиши без ограничения времени.
        return max(round(delay_seconds * 1000), 1)
