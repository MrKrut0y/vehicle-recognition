import cv2

from vehicle_recognition.detection import Detection


class PixelMotionAnalyzer:
    """Подтверждает реальное движение по изменениям пикселей внутри рамок."""

    def __init__(
        self,
        motion_threshold: float = 0.08,
        warmup_frames: int = 30,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.warmup_frames = warmup_frames
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,
            detectShadows=False,
        )
        self.motion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

    def update(
        self,
        frame,
        detections: list[Detection],
        frame_number: int,
    ) -> None:
        """Записывает долю движущихся пикселей в каждое обнаружение."""
        foreground_mask = self.background_subtractor.apply(frame)
        foreground_mask = cv2.morphologyEx(
            foreground_mask,
            cv2.MORPH_OPEN,
            self.motion_kernel,
        )
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            x1 = min(max(int(detection.x1), 0), frame_width)
            y1 = min(max(int(detection.y1), 0), frame_height)
            x2 = min(max(int(detection.x2), 0), frame_width)
            y2 = min(max(int(detection.y2), 0), frame_height)

            if x2 <= x1 or y2 <= y1:
                detection.pixel_motion_ratio = 0.0
                detection.has_pixel_motion = False
                continue

            # Центральная половина рамки меньше захватывает соседний транспорт и фон.
            margin_x = max(int((x2 - x1) * 0.25), 1)
            margin_y = max(int((y2 - y1) * 0.25), 1)
            motion_x1 = min(x1 + margin_x, x2 - 1)
            motion_y1 = min(y1 + margin_y, y2 - 1)
            motion_x2 = max(x2 - margin_x, motion_x1 + 1)
            motion_y2 = max(y2 - margin_y, motion_y1 + 1)

            motion_crop = foreground_mask[
                motion_y1:motion_y2,
                motion_x1:motion_x2,
            ]
            motion_area = max(motion_crop.size, 1)
            motion_ratio = cv2.countNonZero(motion_crop) / motion_area

            detection.pixel_motion_ratio = motion_ratio
            detection.has_pixel_motion = (
                frame_number > self.warmup_frames
                and motion_ratio >= self.motion_threshold
            )
