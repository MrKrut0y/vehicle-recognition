import pytest

from vehicle_recognition.periodic_trigger import PeriodicTrigger


class FakeClock:
    def __init__(self, *timestamps: float) -> None:
        self.timestamps = iter(timestamps)

    def __call__(self) -> float:
        return next(self.timestamps)


def test_trigger_is_not_due_before_interval():
    trigger = PeriodicTrigger(10.0, clock=FakeClock(100.0, 109.9))

    assert trigger.is_due() is False


def test_trigger_is_due_at_interval_boundary():
    trigger = PeriodicTrigger(10.0, clock=FakeClock(100.0, 110.0))

    assert trigger.is_due() is True


def test_trigger_starts_new_interval_after_it_is_due():
    trigger = PeriodicTrigger(
        10.0,
        clock=FakeClock(100.0, 115.0, 124.9, 125.0),
    )

    assert trigger.is_due() is True
    assert trigger.is_due() is False
    assert trigger.is_due() is True


def test_trigger_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="greater than zero"):
        PeriodicTrigger(0.0)
