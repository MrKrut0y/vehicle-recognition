import math


def format_duration(seconds: float) -> str:
    """Преобразует продолжительность в формат ЧЧ:ММ:СС."""
    total_seconds = max(math.ceil(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def estimate_remaining_seconds(
    frame_number: int,
    total_frames: int | None,
    elapsed_seconds: float,
) -> float | None:
    """Оценивает оставшееся время по средней скорости обработки кадров."""
    if (
        frame_number <= 0
        or total_frames is None
        or total_frames <= 0
        or elapsed_seconds < 0
    ):
        return None

    seconds_per_frame = elapsed_seconds / frame_number
    remaining_frames = max(total_frames - frame_number, 0)

    return remaining_frames * seconds_per_frame


def build_progress_message(
    frame_number: int,
    total_frames: int | None,
    vehicle_count: int,
    elapsed_seconds: float,
) -> str:
    """Формирует одинаковое сообщение для обычного и подробного режимов."""
    remaining_seconds = estimate_remaining_seconds(
        frame_number,
        total_frames,
        elapsed_seconds,
    )
    remaining_text = (
        format_duration(remaining_seconds)
        if remaining_seconds is not None
        else "неизвестно"
    )

    if total_frames is None or total_frames <= 0:
        return (
            f"Кадр: {frame_number} | Машин: {vehicle_count} | "
            f"Осталось: {remaining_text}"
        )

    progress_percent = min(
        frame_number / total_frames * 100,
        100.0,
    )

    return (
        f"Обработано: {progress_percent:6.2f}% | "
        f"Кадр: {frame_number}/{total_frames} | "
        f"Машин: {vehicle_count} | "
        f"Осталось: {remaining_text}"
    )


def print_progress(
    frame_number: int,
    total_frames: int | None,
    vehicle_count: int,
    elapsed_seconds: float,
) -> None:
    """Выводит прогресс обработки в одной строке терминала."""
    message = build_progress_message(
        frame_number,
        total_frames,
        vehicle_count,
        elapsed_seconds,
    )

    print(f"\r{message}", end="", flush=True)
