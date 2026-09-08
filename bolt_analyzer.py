import base64
import cv2
import logging
import math
import os
from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

QUARTER_DIAMETER_MM = 24.26

# Metric bolt specifications: shank diameter, hex head across-flats (wrench), standard coarse pitch
METRIC_BOLT_SPECS = {
    "M3":  {"shank": 3.0,  "wrench": 5.5,  "pitch": 0.5},
    "M4":  {"shank": 4.0,  "wrench": 7.0,  "pitch": 0.7},
    "M5":  {"shank": 5.0,  "wrench": 8.0,  "pitch": 0.8},
    "M6":  {"shank": 6.0,  "wrench": 10.0, "pitch": 1.0},
    "M7":  {"shank": 7.0,  "wrench": 11.0, "pitch": 1.0},
    "M8":  {"shank": 8.0,  "wrench": 13.0, "pitch": 1.25},
    "M10": {"shank": 10.0, "wrench": 17.0, "pitch": 1.5},
    "M12": {"shank": 12.0, "wrench": 19.0, "pitch": 1.75},
    "M14": {"shank": 14.0, "wrench": 22.0, "pitch": 2.0},
    "M16": {"shank": 16.0, "wrench": 24.0, "pitch": 2.0},
    "M20": {"shank": 20.0, "wrench": 30.0, "pitch": 2.5},
    "M24": {"shank": 24.0, "wrench": 36.0, "pitch": 3.0},
}

# SAE/imperial bolt specifications (dimensions in mm for internal matching)
# shank = nominal diameter in mm, wrench = hex head across-flats in mm, tpi = threads per inch
SAE_BOLT_SPECS = {
    "#6":    {"shank": 3.505, "wrench": 7.938,  "tpi": 32},
    "#8":    {"shank": 4.166, "wrench": 8.731,  "tpi": 32},
    "#10":   {"shank": 4.826, "wrench": 9.525,  "tpi": 24},
    "1/4\"":  {"shank": 6.35,  "wrench": 11.113, "tpi": 20},
    "5/16\"": {"shank": 7.938, "wrench": 12.7,   "tpi": 18},
    "3/8\"":  {"shank": 9.525, "wrench": 14.288, "tpi": 16},
    "7/16\"": {"shank": 11.113,"wrench": 15.875, "tpi": 14},
    "1/2\"":  {"shank": 12.7,  "wrench": 19.05,  "tpi": 13},
    "9/16\"": {"shank": 14.288,"wrench": 20.638, "tpi": 12},
    "5/8\"":  {"shank": 15.875,"wrench": 23.813, "tpi": 11},
    "3/4\"":  {"shank": 19.05, "wrench": 28.575, "tpi": 10},
    "7/8\"":  {"shank": 22.225,"wrench": 33.338, "tpi": 9},
    "1\"":    {"shank": 25.4,  "wrench": 38.1,   "tpi": 8},
}

MM_PER_INCH = 25.4
MAX_IMAGE_PIXELS = 32_000_000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions are too large.")
    return image


@dataclass(frozen=True)
class ScaleCalibration:
    """Pixel-to-mm calibration inferred from the quarter silhouette.

    A tilted camera turns a circular quarter into an ellipse.  Keeping a scale
    for both ellipse axes lets us correct measurements in any image direction
    under the usual (top-down, approximately affine) camera model.
    """

    axis_x: tuple
    axis_y: tuple
    mm_per_px_x: float
    mm_per_px_y: float
    quarter_axis_ratio: float
    method: str

    @property
    def isotropic_mm_per_px(self):
        return math.sqrt(self.mm_per_px_x * self.mm_per_px_y)

    def scale_for_direction(self, direction):
        vector = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("Measurement direction has zero length.")
        vector /= norm
        x_component = float(np.dot(vector, self.axis_x))
        y_component = float(np.dot(vector, self.axis_y))
        return math.hypot(
            x_component * self.mm_per_px_x,
            y_component * self.mm_per_px_y,
        )


def find_latest_model(base_dir):
    shipped = os.path.join(base_dir, "model", "best.pt")
    if os.path.isfile(shipped):
        return shipped
    runs_dir = os.path.join(base_dir, "runs", "detect")
    if not os.path.isdir(runs_dir):
        return None
    candidates = []
    for name in os.listdir(runs_dir):
        weights = os.path.join(runs_dir, name, "weights", "best.pt")
        if os.path.isfile(weights):
            candidates.append(weights)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def compute_calibration(quarter_box, image):
    """Build a direction-aware calibration from the quarter's outer edge."""
    x1, y1, x2, y2 = map(int, quarter_box)
    h, w = image.shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        raise ValueError("Quarter bounding box has zero size.")

    pad = max(3, int(max(box_w, box_h) * 0.08))
    crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
    crop_x2, crop_y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median = float(np.median(blurred))
        edges = cv2.Canny(
            blurred,
            max(20, int(median * 0.5)),
            min(255, max(60, int(median * 1.35))),
        )
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        expected_center = np.array(
            [((x1 + x2) / 2) - crop_x1, ((y1 + y2) / 2) - crop_y1],
            dtype=float,
        )
        expected_diameter = float(max(box_w, box_h))
        best_ellipse = None
        best_score = -1.0

        for contour in contours:
            if len(contour) < 20:
                continue
            (center, axes, angle) = cv2.fitEllipse(contour)
            axis_a, axis_b = map(float, axes)
            if min(axis_a, axis_b) < min(box_w, box_h) * 0.55:
                continue
            if max(axis_a, axis_b) > expected_diameter * 1.25:
                continue
            center_error = np.linalg.norm(np.asarray(center) - expected_center) / expected_diameter
            size_error = abs(max(axis_a, axis_b) - expected_diameter) / expected_diameter
            score = 1.0 - (0.65 * size_error + 0.35 * center_error)
            if score > best_score:
                best_score = score
                best_ellipse = (axis_a, axis_b, float(angle))

        if best_ellipse is not None and best_score > 0.55:
            axis_a, axis_b, angle = best_ellipse
            theta = math.radians(angle)
            vector_a = np.array([math.cos(theta), math.sin(theta)])
            vector_b = np.array([-math.sin(theta), math.cos(theta)])
            if axis_a >= axis_b:
                major, minor = axis_a, axis_b
                major_vector, minor_vector = vector_a, vector_b
            else:
                major, minor = axis_b, axis_a
                major_vector, minor_vector = vector_b, vector_a
            return ScaleCalibration(
                axis_x=tuple(major_vector),
                axis_y=tuple(minor_vector),
                mm_per_px_x=QUARTER_DIAMETER_MM / major,
                mm_per_px_y=QUARTER_DIAMETER_MM / minor,
                quarter_axis_ratio=minor / major,
                method="ellipse",
            )

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(crop.shape[:2]),
            param1=100,
            param2=30,
            minRadius=max(1, int(min(box_w, box_h) * 0.3)),
            maxRadius=max(2, int(max(box_w, box_h) * 0.6)),
        )
        if circles is not None:
            diameter = float(max(circles[0], key=lambda circle: circle[2])[2]) * 2
            if diameter > 0:
                scale = QUARTER_DIAMETER_MM / diameter
                return ScaleCalibration(
                    axis_x=(1.0, 0.0),
                    axis_y=(0.0, 1.0),
                    mm_per_px_x=scale,
                    mm_per_px_y=scale,
                    quarter_axis_ratio=1.0,
                    method="circle",
                )

    diameter = (box_w + box_h) / 2
    scale = QUARTER_DIAMETER_MM / diameter
    return ScaleCalibration(
        axis_x=(1.0, 0.0),
        axis_y=(0.0, 1.0),
        mm_per_px_x=scale,
        mm_per_px_y=scale,
        quarter_axis_ratio=min(box_w, box_h) / max(box_w, box_h),
        method="bounding_box",
    )


def compute_scale(quarter_box, image):
    """Backward-compatible scalar scale; prefer ``compute_calibration``."""
    return compute_calibration(quarter_box, image).isotropic_mm_per_px


def _median_smooth(values, radius):
    if radius <= 0 or len(values) < 3:
        return values.astype(float)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray([
        np.median(padded[index:index + 2 * radius + 1])
        for index in range(len(values))
    ])


def estimate_bolt_dimensions(bolt_box, calibration, image, return_details=False):
    """Measure shank diameter and nominal under-head length from a bolt silhouette.

    The old implementation used the short and long sides of one rectangle around
    the entire bolt.  That measures the head (often across corners) as the bolt
    diameter and includes head thickness in length.  A longitudinal width
    profile separates the narrower shank from the wider head instead.
    """
    x1, y1, x2, y2 = map(int, bolt_box)
    bb_w = x2 - x1
    bb_h = y2 - y1
    if bb_w <= 0 or bb_h <= 0:
        raise ValueError("Bolt bounding box has zero size.")
    pad = max(5, int(min(bb_w, bb_h) * 0.05))
    h, w = image.shape[:2]
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    bb_short = min(bb_w, bb_h)
    bb_long = max(bb_w, bb_h)
    bb_area = bb_w * bb_h

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        scalar = calibration if isinstance(calibration, (int, float)) else calibration.isotropic_mm_per_px
        result = (bb_short * scalar, bb_long * scalar)
        return (*result, {"method": "bounding_box"}) if return_details else result

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    masks = []
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.extend((otsu, cv2.bitwise_not(otsu)))
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8,
    )
    masks.extend((adaptive, cv2.bitwise_not(adaptive)))
    median_val = int(np.median(blurred))
    lower = max(0, int(median_val * 0.4))
    upper = min(255, int(median_val * 1.4))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    masks.append(closed)

    best_contours = None
    best_score = -1.0
    min_area = bb_area * 0.15
    expected_center = np.array([(x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1])

    for mask in masks:
        clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                 iterations=1)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= min_area * 0.1 or len(contour) < 5:
                continue
            contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
            touches_crop_edge = (
                contour_x <= 1
                or contour_y <= 1
                or contour_x + contour_w >= crop.shape[1] - 1
                or contour_y + contour_h >= crop.shape[0] - 1
            )
            if touches_crop_edge or area > bb_area * 0.9:
                continue
            significant.append(contour)

        candidates = [[contour] for contour in significant]
        if len(significant) > 1:
            candidates.append(significant)

        for candidate in candidates:
            points = np.vstack(candidate)
            area = sum(cv2.contourArea(contour) for contour in candidate)
            if area > bb_area * 0.9:
                continue
            rect = cv2.minAreaRect(points)
            short_px, long_px = sorted(rect[1])
            if short_px <= 0 or long_px <= 0:
                continue
            fill_ratio = (short_px * long_px) / bb_area
            if fill_ratio < 0.12 or fill_ratio > 1.6:
                continue
            if len(candidate) > 1:
                # This candidate unions several disjoint contours into one
                # minAreaRect. That rect's fill/aspect/center stats describe
                # its enclosing rectangle, not the pixels inside it -- three
                # unrelated fragments scattered across the bounding box can
                # span it neatly (high fill, great aspect match) while the
                # silhouette drawn from them is mostly empty gap, corrupting
                # the width profile downstream. A real object's own contour
                # (single or fragmented-by-a-thin-gap) measured this way
                # consistently comes out at solidity >= ~0.4; require the
                # merge not fall far below that before trusting it as one
                # object rather than noise that happens to bracket the box.
                solidity = area / (short_px * long_px)
                if solidity < 0.35:
                    continue
            bb_aspect = bb_long / bb_short
            contour_aspect = long_px / short_px
            aspect_agreement = 1.0 - min(abs(bb_aspect - contour_aspect) / bb_aspect, 1.0)
            size_agreement = 1.0 - min(abs(1.0 - fill_ratio), 1.0)
            center_error = np.linalg.norm(np.asarray(rect[0]) - expected_center) / bb_long
            center_agreement = 1.0 - min(center_error, 1.0)
            score = 0.5 * aspect_agreement + 0.3 * size_agreement + 0.2 * center_agreement
            if score > best_score:
                best_score = score
                best_contours = candidate

    if best_contours is None:
        scalar = calibration if isinstance(calibration, (int, float)) else calibration.isotropic_mm_per_px
        result = (bb_short * scalar, bb_long * scalar)
        return (*result, {"method": "bounding_box"}) if return_details else result

    rect = cv2.minAreaRect(np.vstack(best_contours))
    rect_w, rect_h = rect[1]
    theta = math.radians(rect[2])
    width_axis = np.array([math.cos(theta), math.sin(theta)])
    height_axis = np.array([-math.sin(theta), math.cos(theta)])
    long_axis = width_axis if rect_w >= rect_h else height_axis
    short_axis = np.array([-long_axis[1], long_axis[0]])

    silhouette = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(silhouette, best_contours, -1, 255, thickness=cv2.FILLED)
    ys, xs = np.nonzero(silhouette)
    points = np.column_stack((xs, ys)).astype(float)
    longitudinal = points @ long_axis
    transverse = points @ short_axis
    start = math.floor(float(longitudinal.min()))
    bins = np.floor(longitudinal - start).astype(int)
    profile_size = int(bins.max()) + 1
    low = np.full(profile_size, np.inf)
    high = np.full(profile_size, -np.inf)
    np.minimum.at(low, bins, transverse)
    np.maximum.at(high, bins, transverse)
    profile = high - low + 1.0
    valid = np.isfinite(profile)
    if valid.sum() < 5:
        scalar = calibration if isinstance(calibration, (int, float)) else calibration.isotropic_mm_per_px
        result = (min(rect[1]) * scalar, max(rect[1]) * scalar)
        return (*result, {"method": "rotated_box"}) if return_details else result
    profile = np.interp(np.arange(profile_size), np.flatnonzero(valid), profile[valid])
    profile = _median_smooth(profile, max(1, min(5, profile_size // 80)))

    window = max(3, int(profile_size * 0.15))
    left_width = float(np.percentile(profile[:window], 80))
    right_width = float(np.percentile(profile[-window:], 80))
    if right_width > left_width:
        profile = profile[::-1]
        long_axis = -long_axis

    sample_start = int(profile_size * 0.45)
    sample_end = max(sample_start + 1, int(profile_size * 0.95))
    shank_sample = profile[sample_start:sample_end]
    shank_px = float(np.percentile(shank_sample, 80))

    head_window_end = max(3, int(profile_size * 0.5))
    head_peak = int(np.argmax(profile[:head_window_end]))
    transition = 0
    threshold = max(shank_px * 1.18, shank_px + 1.5)
    head_is_distinct = float(np.percentile(profile[:window], 80)) > threshold
    if head_is_distinct:
        run = max(2, int(profile_size * 0.025))
        for index in range(head_peak + 1, head_window_end - run + 1):
            if np.all(profile[index:index + run] <= threshold):
                transition = index
                break

    # A valid hex/flange head is short relative to shank diameter.  This guard
    # prevents texture or shadows farther down the threads from being mistaken
    # for the under-head boundary.
    transition = min(transition, int(round(shank_px * 1.15)))

    length_px = float(profile_size - transition)
    if isinstance(calibration, (int, float)):
        width_scale = length_scale = float(calibration)
    else:
        width_scale = calibration.scale_for_direction(short_axis)
        length_scale = calibration.scale_for_direction(long_axis)
    width_mm = shank_px * width_scale
    length_mm = length_px * length_scale
    overall_length_mm = profile_size * length_scale
    head_width_mm = float(np.percentile(profile[:max(window, transition)], 90)) * width_scale
    # A hex/flange/socket head can never be narrower than the shank it sits on.
    # If the region before the transition doesn't measure wider than the shank,
    # the transition point did not land on a real head boundary (texture,
    # shadow, or a straight-shank fastener) -- fall back the same as when no
    # head-like step was found at all, instead of reporting an impossible pair
    # of numbers (head thinner than shank).
    head_detected = head_is_distinct and transition > 0 and head_width_mm > width_mm
    if not head_detected:
        # Preserve the useful shank measurement when thresholding found only the
        # shaft.  The detector box is a safer low-confidence overall length than
        # a partial contour, although it can include head thickness.
        length_mm = bb_long * length_scale
    details = {
        "method": "silhouette_profile",
        "head_detected": head_detected,
        "head_width_mm": head_width_mm,
        "overall_length_mm": overall_length_mm,
        "segmentation_score": round(best_score, 3),
    }
    result = (width_mm, length_mm)
    return (*result, details) if return_details else result


def classify_bolt(width_mm, unit_system="metric", measurement="shank"):
    """Match a measured shank diameter (or legacy wrench width) to a bolt spec."""

    if unit_system == "imperial":
        specs_table = SAE_BOLT_SPECS
    else:
        specs_table = METRIC_BOLT_SPECS

    dimension = "shank" if measurement == "shank" else "wrench"
    name, specs = min(
        specs_table.items(),
        key=lambda kv: abs(kv[1][dimension] - width_mm),
    )

    tolerance = max(0.5, specs[dimension] * 0.15)
    if abs(specs[dimension] - width_mm) > tolerance:
        result = {
            "bolt_type": "Unknown",
            "unit_system": unit_system,
            "shank_diameter_mm": None,
            "wrench_size_mm": None,
        }
        if unit_system == "imperial":
            result["shank_diameter_in"] = None
            result["wrench_size_in"] = None
            result["tpi"] = None
        else:
            result["thread_pitch_mm"] = None
        result["size_candidates"] = []
        return result

    candidates = [
        candidate_name
        for candidate_name, candidate_specs in specs_table.items()
        if abs(candidate_specs[dimension] - width_mm)
        <= max(0.5, candidate_specs[dimension] * 0.15)
    ]

    if unit_system == "imperial":
        return {
            "bolt_type": name,
            "unit_system": "imperial",
            "shank_diameter_mm": specs["shank"],
            "shank_diameter_in": round(specs["shank"] / MM_PER_INCH, 4),
            "wrench_size_mm": specs["wrench"],
            "wrench_size_in": round(specs["wrench"] / MM_PER_INCH, 4),
            "tpi": specs["tpi"],
            "size_candidates": candidates,
        }

    return {
        "bolt_type": name,
        "unit_system": "metric",
        "shank_diameter_mm": specs["shank"],
        "wrench_size_mm": specs["wrench"],
        "thread_pitch_mm": specs["pitch"],
        "size_candidates": candidates,
    }


def _draw_label(img, text, x, y, color):
    """Draw text with a filled background pill."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + 2), color, -1)
    cv2.putText(img, text, (x + 4, y - 3), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def generate_annotated_image(image, bolt_box=None, quarter_box=None, bolt_info=None):
    """Draw detection boxes and labels, return base64-encoded JPEG."""
    annotated = image.copy()

    if quarter_box:
        qx1, qy1, qx2, qy2 = map(int, quarter_box)
        cv2.rectangle(annotated, (qx1, qy1), (qx2, qy2), (0, 165, 255), 2)
        _draw_label(annotated, "Quarter (24.26mm)", qx1, qy1 - 4, (0, 165, 255))

    if bolt_box:
        bx1, by1, bx2, by2 = map(int, bolt_box)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 200, 0), 2)

        if bolt_info and "bolt_type" in bolt_info:
            length_label = bolt_info.get("bolt_length_mm")
            length_label = "?" if length_label is None else length_label
            label = (
                f"{bolt_info['bolt_type']}  "
                f"{bolt_info.get('bolt_width_mm', '?')}x{length_label}mm"
            )
        else:
            label = "Bolt"

        label_y = by1 - 4 if by1 > 30 else by2 + 18
        _draw_label(annotated, label, bx1, label_y, (0, 200, 0))

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


class BoltAnalyzer:
    """Detects a bolt and a US quarter in an image, then estimates bolt size."""

    def __init__(self, model_path, min_confidence=0.25):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if YOLO is None:
            raise RuntimeError("ultralytics is required to use the local model")
        self.detector = YOLO(model_path)
        self.min_confidence = float(min_confidence)

        names_lower = {v.lower(): k for k, v in self.detector.names.items()}
        self._bolt_id = names_lower.get("bolt", 0)
        self._quarter_id = names_lower.get("quarter", 1)
        logger.info(f"Model loaded. bolt_id={self._bolt_id}  quarter_id={self._quarter_id}")

    def detect_objects(self, img_path, unit_system="metric"):
        """
        Run detection on img_path.

        Returns a dict with detection flags, confidence scores,
        measurements, classification, and a base64 annotated image.
        """
        if not os.path.isfile(img_path):
            return {"bolt": False, "quarter": False}

        image = _read_image(img_path)
        if image is None:
            return {"bolt": False, "quarter": False}

        results = self.detector(img_path, conf=self.min_confidence)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return {"bolt": False, "quarter": False}

        raw = results.boxes.data.detach().cpu().numpy()

        best_bolt = None
        best_quarter = None
        bolt_count = 0

        for x1, y1, x2, y2, conf, cid in raw:
            box = (float(x1), float(y1), float(x2), float(y2))
            if int(cid) == self._bolt_id:
                bolt_count += 1
                if best_bolt is None or conf > best_bolt[0]:
                    best_bolt = (float(conf), box)
            elif int(cid) == self._quarter_id:
                if best_quarter is None or conf > best_quarter[0]:
                    best_quarter = (float(conf), box)

        bolt_box = best_bolt[1] if best_bolt else None
        quarter_box = best_quarter[1] if best_quarter else None

        result = {
            "bolt": bolt_box is not None,
            "quarter": quarter_box is not None,
            "bolt_confidence": round(best_bolt[0] * 100, 1) if best_bolt else None,
            "quarter_confidence": round(best_quarter[0] * 100, 1) if best_quarter else None,
            "bolts_detected": bolt_count,
        }

        if bolt_box is not None and quarter_box is not None:
            calibration = compute_calibration(quarter_box, image)
            width_mm, length_mm, measurement = estimate_bolt_dimensions(
                bolt_box, calibration, image, return_details=True
            )
            classification = classify_bolt(width_mm, unit_system, measurement="shank")

            result.update({
                "bolt_type": classification["bolt_type"],
                "unit_system": classification["unit_system"],
                "bolt_width_mm": round(width_mm, 2),
                "bolt_length_mm": round(length_mm, 2) if measurement["head_detected"] else None,
                "wrench_size_mm": classification["wrench_size_mm"],
                "shank_diameter_mm": classification["shank_diameter_mm"],
                "size_candidates": classification["size_candidates"],
                "measured_shank_mm": round(width_mm, 2),
                "measured_head_width_mm": round(measurement["head_width_mm"], 2),
                "measurement_method": measurement["method"],
                "calibration_method": calibration.method,
                "quarter_axis_ratio": round(calibration.quarter_axis_ratio, 3),
            })

            warnings = []
            if calibration.quarter_axis_ratio < 0.82:
                warnings.append("Camera angle is steep; retake the photo closer to straight overhead.")
            if not measurement["head_detected"]:
                warnings.append("The head boundary was unclear, so the length may include part of the head.")
            if warnings:
                result["measurement_warnings"] = warnings

            if unit_system == "imperial":
                result.update({
                    "bolt_width_in": round(width_mm / MM_PER_INCH, 3),
                    "bolt_length_in": (
                        round(length_mm / MM_PER_INCH, 3)
                        if measurement["head_detected"] else None
                    ),
                    "wrench_size_in": classification["wrench_size_in"],
                    "shank_diameter_in": classification["shank_diameter_in"],
                    "tpi": classification["tpi"],
                })
            else:
                result["thread_pitch_mm"] = classification["thread_pitch_mm"]

        bolt_info = result if bolt_box and quarter_box else None
        result["annotated_image"] = generate_annotated_image(
            image, bolt_box, quarter_box, bolt_info
        )

        return result


class RoboflowAnalyzer:
    """Detects a bolt and a US quarter via the Roboflow inference SDK."""

    def __init__(self, api_key, workspace, workflow_id, min_confidence=0.25):
        from inference_sdk import InferenceHTTPClient
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.min_confidence = float(min_confidence)
        logger.info(f"Roboflow analyzer ready. workspace={workspace} workflow={workflow_id}")

    def _extract_predictions(self, results):
        """Extract predictions list from the workflow response."""
        if not results:
            return []
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    preds = item.get("predictions", [])
                    if isinstance(preds, list) and preds:
                        return preds
                    if isinstance(preds, dict):
                        inner = preds.get("predictions", [])
                        if isinstance(inner, list) and inner:
                            return inner
        if isinstance(results, dict):
            if "predictions" in results:
                return results["predictions"]
        return []

    def detect_objects(self, img_path, unit_system="metric"):
        if not os.path.isfile(img_path):
            return {"bolt": False, "quarter": False}

        image = _read_image(img_path)
        if image is None:
            return {"bolt": False, "quarter": False}

        try:
            results = self.client.run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow_id,
                images={"image": img_path},
                use_cache=True,
            )
            logger.debug("Roboflow workflow returned %s", type(results).__name__)
        except Exception as e:
            logger.error(f"Roboflow API error: {e}")
            return {"bolt": False, "quarter": False, "service_error": True}

        predictions = self._extract_predictions(results)
        if not predictions:
            return {"bolt": False, "quarter": False}

        best_bolt = None
        best_quarter = None
        bolt_count = 0

        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            try:
                cx = float(pred["x"])
                cy = float(pred["y"])
                pw = float(pred["width"])
                ph = float(pred["height"])
                conf = float(pred["confidence"])
                cls = str(pred["class"]).lower()
            except (KeyError, TypeError, ValueError):
                logger.warning("Ignoring malformed Roboflow prediction: %r", pred)
                continue
            if conf < self.min_confidence or pw <= 0 or ph <= 0:
                continue
            x1 = cx - pw / 2
            y1 = cy - ph / 2
            x2 = cx + pw / 2
            y2 = cy + ph / 2
            box = (float(x1), float(y1), float(x2), float(y2))
            if cls == "bolt":
                bolt_count += 1
                if best_bolt is None or conf > best_bolt[0]:
                    best_bolt = (float(conf), box)
            elif cls == "quarter":
                if best_quarter is None or conf > best_quarter[0]:
                    best_quarter = (float(conf), box)

        bolt_box = best_bolt[1] if best_bolt else None
        quarter_box = best_quarter[1] if best_quarter else None

        result = {
            "bolt": bolt_box is not None,
            "quarter": quarter_box is not None,
            "bolt_confidence": round(best_bolt[0] * 100, 1) if best_bolt else None,
            "quarter_confidence": round(best_quarter[0] * 100, 1) if best_quarter else None,
            "bolts_detected": bolt_count,
        }

        if bolt_box is not None and quarter_box is not None:
            calibration = compute_calibration(quarter_box, image)
            width_mm, length_mm, measurement = estimate_bolt_dimensions(
                bolt_box, calibration, image, return_details=True
            )
            classification = classify_bolt(width_mm, unit_system, measurement="shank")

            result.update({
                "bolt_type": classification["bolt_type"],
                "unit_system": classification["unit_system"],
                "bolt_width_mm": round(width_mm, 2),
                "bolt_length_mm": round(length_mm, 2) if measurement["head_detected"] else None,
                "wrench_size_mm": classification["wrench_size_mm"],
                "shank_diameter_mm": classification["shank_diameter_mm"],
                "size_candidates": classification["size_candidates"],
                "measured_shank_mm": round(width_mm, 2),
                "measured_head_width_mm": round(measurement["head_width_mm"], 2),
                "measurement_method": measurement["method"],
                "calibration_method": calibration.method,
                "quarter_axis_ratio": round(calibration.quarter_axis_ratio, 3),
            })

            warnings = []
            if calibration.quarter_axis_ratio < 0.82:
                warnings.append("Camera angle is steep; retake the photo closer to straight overhead.")
            if not measurement["head_detected"]:
                warnings.append("The head boundary was unclear, so the length may include part of the head.")
            if warnings:
                result["measurement_warnings"] = warnings

            if unit_system == "imperial":
                result.update({
                    "bolt_width_in": round(width_mm / MM_PER_INCH, 3),
                    "bolt_length_in": (
                        round(length_mm / MM_PER_INCH, 3)
                        if measurement["head_detected"] else None
                    ),
                    "wrench_size_in": classification["wrench_size_in"],
                    "shank_diameter_in": classification["shank_diameter_in"],
                    "tpi": classification["tpi"],
                })
            else:
                result["thread_pitch_mm"] = classification["thread_pitch_mm"]

        bolt_info = result if bolt_box and quarter_box else None
        result["annotated_image"] = generate_annotated_image(
            image, bolt_box, quarter_box, bolt_info
        )

        return result


if __name__ == "__main__":
    import argparse

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_MODEL = find_latest_model(BASE_DIR)
    DEFAULT_IMAGE = os.path.join(BASE_DIR, "test_images", "bqm10.jpg")

    parser = argparse.ArgumentParser(description="BoltMetrics: detect and measure a bolt.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLO weights (.pt)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    args = parser.parse_args()

    if args.model is None:
        print("No trained model found. Run: python scripts/train.py")
        exit(1)

    analyzer = BoltAnalyzer(args.model, min_confidence=args.conf)
    result = analyzer.detect_objects(args.image)
    display = {k: v for k, v in result.items() if k != "annotated_image"}
    print("Result:", display)
