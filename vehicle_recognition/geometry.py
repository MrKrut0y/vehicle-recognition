import math

Box = tuple[float, float, float, float]


def calculate_box_area(box: Box) -> float:
    """Возвращает площадь прямоугольной рамки."""
    x1, y1, x2, y2 = box

    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)

    return width * height


def calculate_intersection_area(first_box: Box, second_box: Box) -> float:
    """Возвращает площадь пересечения двух рамок."""
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box

    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)

    intersection_width = max(intersection_x2 - intersection_x1, 0)
    intersection_height = max(intersection_y2 - intersection_y1, 0)

    return intersection_width * intersection_height


def calculate_iou(first_box: Box, second_box: Box) -> float:
    """Возвращает отношение пересечения рамок к их объединению."""
    first_box_area = calculate_box_area(first_box)
    second_box_area = calculate_box_area(second_box)

    intersection_area = calculate_intersection_area(first_box, second_box)

    union_area = first_box_area + second_box_area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def calculate_smaller_coverage(first_box: Box, second_box: Box) -> float:
    """Возвращает долю меньшей рамки, покрытую пересечением."""
    first_box_area = calculate_box_area(first_box)
    second_box_area = calculate_box_area(second_box)

    intersection_area = calculate_intersection_area(first_box, second_box)

    min_area = min(first_box_area, second_box_area)

    if min_area == 0:
        return 0.0

    return intersection_area / min_area


def calculate_box_diagonal(box: Box) -> float:
    """Возвращает длину диагонали прямоугольной рамки."""
    x1, y1, x2, y2 = box

    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)

    return math.hypot(width, height)
