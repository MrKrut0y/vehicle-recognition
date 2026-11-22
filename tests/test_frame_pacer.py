from vehicle_recognition.frame_pacer import FramePacer


class FakeClock:
    def __init__(self, *timestamps: float) -> None:
        self.timestamps = iter(timestamps)

    def __call__(self) -> float:
        return next(self.timestamps)


def test_pacer_waits_for_source_frame_interval():
    pacer = FramePacer(10.0, clock=FakeClock(0.0))

    assert pacer.next_delay_ms() == 100


def test_pacer_subtracts_processing_time_from_delay():
    pacer = FramePacer(10.0, clock=FakeClock(0.0, 0.14))

    pacer.next_delay_ms()

    assert pacer.next_delay_ms() == 60


def test_pacer_does_not_add_delay_when_processing_is_slow():
    pacer = FramePacer(10.0, clock=FakeClock(0.0, 0.25))

    pacer.next_delay_ms()

    assert pacer.next_delay_ms() == 1


def test_pacer_resets_schedule_after_stream_stall():
    pacer = FramePacer(10.0, clock=FakeClock(0.0, 1.0))

    pacer.next_delay_ms()

    assert pacer.next_delay_ms() == 100


def test_pacer_uses_fallback_for_unknown_fps():
    pacer = FramePacer(0.0, clock=FakeClock(0.0))

    assert pacer.next_delay_ms() == 33
