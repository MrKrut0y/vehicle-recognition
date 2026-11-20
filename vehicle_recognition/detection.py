from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """Хранит информацию об одном обнаруженном транспортном средстве."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None
    appearance_histogram: np.ndarray | None = None
