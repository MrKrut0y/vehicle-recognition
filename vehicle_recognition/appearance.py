import cv2
import numpy as np


def calculate_appearance_histogram(crop: np.ndarray | None) -> np.ndarray | None:
    """Возвращает нормализованную HSV-гистограмму фрагмента изображения."""
    if crop is None or crop.size == 0:
        return None

    # Hue и saturation описывают цвет устойчивее к изменению яркости, чем BGR.
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist([hsv_crop], [0, 1], None, [16, 16], [0, 180, 0, 256])

    # L1-нормализация делает сравнение независимым от размера вырезанного участка.
    cv2.normalize(
        histogram,
        histogram,
        alpha=1.0,
        norm_type=cv2.NORM_L1,
    )

    return histogram


def calculate_histogram_similarity(
    first_histogram: np.ndarray | None, second_histogram: np.ndarray | None
) -> float:
    """Возвращает сходство двух гистограмм в диапазоне от 0 до 1."""
    if first_histogram is None or second_histogram is None:
        return 0.0

    histogram_distance = cv2.compareHist(
        first_histogram, second_histogram, cv2.HISTCMP_BHATTACHARYYA
    )

    similarity = 1.0 - histogram_distance

    similarity = min(similarity, 1.0)
    similarity = max(similarity, 0.0)

    return similarity
