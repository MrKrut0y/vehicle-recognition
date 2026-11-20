import pytest

from vehicle_recognition.geometry import (
    calculate_box_area,
    calculate_box_diagonal,
    calculate_intersection_area,
    calculate_iou,
    calculate_smaller_coverage,
)


def test_box_diagonal_uses_width_and_height():
    assert calculate_box_diagonal((0, 0, 3, 4)) == pytest.approx(5.0)


def test_box_diagonal_is_zero_for_reversed_coordinates():
    assert calculate_box_diagonal((10, 10, 0, 0)) == pytest.approx(0.0)


def test_box_area_for_regular_box():
    assert calculate_box_area((0, 0, 10, 10)) == pytest.approx(100.0)


def test_box_area_is_zero_for_zero_width():
    assert calculate_box_area((5, 0, 5, 10)) == pytest.approx(0.0)


def test_box_area_is_zero_for_reversed_horizontal_coordinates():
    assert calculate_box_area((10, 0, 0, 10)) == pytest.approx(0.0)


def test_intersection_area_for_partially_overlapping_boxes():
    assert calculate_intersection_area(
        (0, 0, 10, 10),
        (5, 5, 15, 15),
    ) == pytest.approx(25.0)


def test_intersection_area_is_zero_for_distant_boxes():
    assert calculate_intersection_area(
        (0, 0, 10, 10),
        (100, 100, 110, 110),
    ) == pytest.approx(0.0)


def test_intersection_area_equals_inner_box_area():
    assert calculate_intersection_area(
        (0, 0, 10, 10),
        (2, 2, 4, 4),
    ) == pytest.approx(4.0)


def test_intersection_area_is_zero_for_touching_boxes():
    assert calculate_intersection_area(
        (0, 0, 10, 10),
        (10, 0, 20, 10),
    ) == pytest.approx(0.0)


def test_intersection_area_does_not_depend_on_argument_order():
    first_result = calculate_intersection_area((0, 0, 10, 10), (5, 5, 15, 15))
    second_result = calculate_intersection_area((5, 5, 15, 15), (0, 0, 10, 10))

    assert first_result == pytest.approx(second_result)


def test_iou_for_partially_overlapping_boxes():
    assert calculate_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_iou_is_zero_for_distant_boxes():
    assert calculate_iou((0, 0, 10, 10), (100, 100, 110, 110)) == pytest.approx(0.0)


def test_iou_is_one_for_identical_boxes():
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_is_zero_for_zero_area_boxes():
    assert calculate_iou((0, 0, 0, 0), (0, 0, 0, 0)) == pytest.approx(0.0)


def test_iou_for_box_inside_another_box():
    assert calculate_iou((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(4 / 100)


def test_iou_is_zero_for_boxes_touching_by_edge():
    assert calculate_iou((0, 0, 10, 10), (10, 0, 20, 10)) == pytest.approx(0.0)


def test_iou_does_not_depend_on_argument_order():
    first_result = calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))
    second_result = calculate_iou((5, 5, 15, 15), (0, 0, 10, 10))

    assert first_result == pytest.approx(second_result)


def test_smaller_coverage_is_one_for_inner_box():
    assert calculate_smaller_coverage(
        (0, 0, 10, 10),
        (2, 2, 4, 4),
    ) == pytest.approx(1.0)


def test_smaller_coverage_for_partially_overlapping_boxes():
    assert calculate_smaller_coverage(
        (0, 0, 10, 10),
        (5, 5, 15, 15),
    ) == pytest.approx(0.25)


def test_smaller_coverage_is_zero_for_distant_boxes():
    assert calculate_smaller_coverage(
        (0, 0, 10, 10),
        (100, 100, 110, 110),
    ) == pytest.approx(0.0)


def test_smaller_coverage_is_zero_for_zero_area_box():
    assert calculate_smaller_coverage(
        (0, 0, 10, 10),
        (0, 0, 0, 0),
    ) == pytest.approx(0.0)


def test_smaller_coverage_does_not_depend_on_argument_order():
    first_result = calculate_smaller_coverage((0, 0, 10, 10), (2, 2, 4, 4))
    second_result = calculate_smaller_coverage((2, 2, 4, 4), (0, 0, 10, 10))

    assert first_result == pytest.approx(second_result)
