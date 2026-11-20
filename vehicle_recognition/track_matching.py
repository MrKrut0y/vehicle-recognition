import math

from vehicle_recognition.appearance import calculate_histogram_similarity
from vehicle_recognition.geometry import (
    calculate_box_diagonal,
    calculate_iou,
    calculate_smaller_coverage,
)
from vehicle_recognition.track_state import TrackState

MIN_CONCURRENT_IOU = 0.08
MIN_CONCURRENT_SMALLER_COVERAGE = 0.40
MAX_CONCURRENT_CENTER_DISTANCE_RATIO = 0.75
MIN_CONCURRENT_APPEARANCE_SIMILARITY = 0.55
MIN_RELIABLE_SPEED = 0.5
MIN_DIRECTION_COSINE = 0.20


def concurrent_tracks_match(first_state: TrackState, second_state: TrackState) -> bool:
    """Проверяет, могут ли два одновременных трека описывать одну машину."""
    if first_state.last_box is None or second_state.last_box is None:
        return False

    if first_state.last_center is None or second_state.last_center is None:
        return False

    first_box_diagonal = calculate_box_diagonal(first_state.last_box)
    second_box_diagonal = calculate_box_diagonal(second_state.last_box)

    largest_box_diagonal = max(first_box_diagonal, second_box_diagonal)

    # Нормализация по размеру рамки одинаково работает на близких и далёких машинах.
    max_center_distance = largest_box_diagonal * MAX_CONCURRENT_CENTER_DISTANCE_RATIO
    center_distance = math.dist(first_state.last_center, second_state.last_center)

    iou = calculate_iou(first_state.last_box, second_state.last_box)
    smaller_coverage = calculate_smaller_coverage(
        first_state.last_box, second_state.last_box
    )

    # Покрытие меньшей рамки учитывает случай, когда она почти целиком вложена в большую.
    if iou < MIN_CONCURRENT_IOU and smaller_coverage < MIN_CONCURRENT_SMALLER_COVERAGE:
        return False

    if center_distance > max_center_distance:
        return False

    appearance_similarity = calculate_histogram_similarity(
        first_state.appearance_histogram, second_state.appearance_histogram
    )

    if appearance_similarity < MIN_CONCURRENT_APPEARANCE_SIMILARITY:
        return False

    first_velocity = first_state.velocity_vector()
    second_velocity = second_state.velocity_vector()

    if first_velocity is not None and second_velocity is not None:
        first_speed = math.hypot(first_velocity[0], first_velocity[1])
        second_speed = math.hypot(second_velocity[0], second_velocity[1])

        # У почти неподвижных треков направление нестабильно и не используется как признак.
        if first_speed >= MIN_RELIABLE_SPEED and second_speed >= MIN_RELIABLE_SPEED:
            dot_product = (
                first_velocity[0] * second_velocity[0]
                + first_velocity[1] * second_velocity[1]
            )
            direction_cosine = dot_product / (first_speed * second_speed)

            if direction_cosine < MIN_DIRECTION_COSINE:
                return False

    return True
