import numpy as np
import pytest

from vehicle_recognition.appearance import (
    calculate_appearance_histogram,
    calculate_histogram_similarity,
)


def test_appearance_histogram_is_none_for_missing_crop():
    assert calculate_appearance_histogram(None) is None


def test_appearance_histogram_is_none_for_empty_crop():
    empty_crop = np.empty((0, 0, 3), dtype=np.uint8)

    assert calculate_appearance_histogram(empty_crop) is None


def test_appearance_histogram_is_normalized():
    crop = np.zeros((10, 10, 3), dtype=np.uint8)

    histogram = calculate_appearance_histogram(crop)

    assert histogram is not None
    assert histogram.shape == (16, 16)
    assert float(histogram.sum()) == pytest.approx(1.0)


def test_histogram_similarity_is_zero_when_histogram_is_missing():
    histogram = np.array([[1.0, 0.0]], dtype=np.float32)

    assert calculate_histogram_similarity(histogram, None) == pytest.approx(0.0)


def test_histogram_similarity_is_one_for_identical_histograms():
    histogram = np.array([[0.25, 0.75]], dtype=np.float32)

    assert calculate_histogram_similarity(histogram, histogram) == pytest.approx(1.0)


def test_histogram_similarity_is_zero_for_opposite_distributions():
    first_histogram = np.array([[1.0, 0.0]], dtype=np.float32)
    second_histogram = np.array([[0.0, 1.0]], dtype=np.float32)

    assert calculate_histogram_similarity(
        first_histogram,
        second_histogram,
    ) == pytest.approx(0.0)
