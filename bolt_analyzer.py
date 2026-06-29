import base64
import cv2
import logging
import os

import numpy as np
import requests

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
    "7/16\"": {"shank": 11.113,"wrench": 16.669, "tpi": 14},
    "1/2\"":  {"shank": 12.7,  "wrench": 19.05,  "tpi": 13},
    "9/16\"": {"shank": 14.288,"wrench": 20.638, "tpi": 12},
    "5/8\"":  {"shank": 15.875,"wrench": 23.813, "tpi": 11},
    "3/4\"":  {"shank": 19.05, "wrench": 28.575, "tpi": 10},
    "7/8\"":  {"shank": 22.225,"wrench": 33.338, "tpi": 9},
    "1\"":    {"shank": 25.4,  "wrench": 38.1,   "tpi": 8},
}

MM_PER_INCH = 25.4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def compute_scale(quarter_box, image):
    """Compute mm-per-pixel scale using the known quarter diameter.
    Uses HoughCircles to detect the circular edge for precision,
    falls back to bounding-box average."""
    x1, y1, x2, y2 = map(int, quarter_box)
    h, w = image.shape[:2]
    crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    pixel_diameter = None

    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        crop_h, crop_w = crop.shape[:2]
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(crop_h, crop_w),
            param1=100, param2=30,
            minRadius=int(min(crop_h, crop_w) * 0.3),
            maxRadius=int(max(crop_h, crop_w) * 0.55),
        )
        if circles is not None:
            pixel_diameter = float(circles[0][0][2]) * 2

    if pixel_diameter is None or pixel_diameter <= 0:
        pixel_diameter = ((x2 - x1) + (y2 - y1)) / 2

    if pixel_diameter <= 0:
        raise ValueError("Quarter bounding box has zero size.")
    return QUARTER_DIAMETER_MM / pixel_diameter


def estimate_bolt_dimensions(bolt_box, scale_mm_per_px, image):
    """ Estimate bolt width and length in mm. Uses multiple segmentation strategies (Otsu, adaptive threshold, Canny)
    and picks the result whose contour best fills the bounding box.
    Short axis -> width, long axis -> length. """
    x1, y1, x2, y2 = map(int, bolt_box)
    bb_w = x2 - x1
    bb_h = y2 - y1
    pad = max(5, int(min(bb_w, bb_h) * 0.05))
    h, w = image.shape[:2]
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    bb_short = min(bb_w, bb_h)
    bb_long = max(bb_w, bb_h)
    bb_area = bb_w * bb_h

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return bb_short * scale_mm_per_px, bb_long * scale_mm_per_px

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    masks = []

    # Otsu's thresholding
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(otsu)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8,
    )
    masks.append(adaptive)

    # Canny with morphological closing
    median_val = int(np.median(blurred))
    lower = max(0, int(median_val * 0.4))
    upper = min(255, int(median_val * 1.4))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    masks.append(closed)

    best_short, best_long = None, None
    best_score = 0.0
    min_area = bb_area * 0.15

    for mask in masks:
        # Clean up the mask
        clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                 iterations=1)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Merge all significant contours into one point set
        significant = [c for c in contours if cv2.contourArea(c) > min_area * 0.1]
        if not significant:
            continue
        merged = np.vstack(significant)

        if len(merged) < 5:
            continue

        rect = cv2.minAreaRect(merged)
        dims = rect[1]
        short_px = min(dims)
        long_px = max(dims)

        if short_px <= 0 or long_px <= 0:
            continue

        # Score: prefer contours that closely match the bounding box aspect ratio
        # and occupy a reasonable fraction of the bounding box area
        contour_area = short_px * long_px
        fill_ratio = contour_area / bb_area if bb_area > 0 else 0
        bb_aspect = bb_long / bb_short if bb_short > 0 else 1
        contour_aspect = long_px / short_px if short_px > 0 else 1
        aspect_agreement = 1.0 - min(abs(bb_aspect - contour_aspect) / bb_aspect, 1.0)

        score = fill_ratio * 0.4 + aspect_agreement * 0.6

        if score > best_score and fill_ratio > 0.15:
            best_score = score
            best_short = short_px
            best_long = long_px

    if best_short is not None:
        return best_short * scale_mm_per_px, best_long * scale_mm_per_px

    return bb_short * scale_mm_per_px, bb_long * scale_mm_per_px


def classify_bolt(width_mm, unit_system="metric"):
    """Match measured width to the closest bolt by hex head size (across-flats).
    Returns 'Unknown' if the best match is more than 20% off."""

    if unit_system == "imperial":
        specs_table = SAE_BOLT_SPECS
    else:
        specs_table = METRIC_BOLT_SPECS

    name, specs = min(
        specs_table.items(),
        key=lambda kv: abs(kv[1]["wrench"] - width_mm),
    )

    tolerance = specs["wrench"] * 0.20
    if abs(specs["wrench"] - width_mm) > tolerance:
        result = {
            "bolt_type": "Unknown",
            "unit_system": unit_system,
            "shank_diameter_mm": None,
            "wrench_size_mm": round(width_mm, 2),
        }
        if unit_system == "imperial":
            result["shank_diameter_in"] = None
            result["wrench_size_in"] = round(width_mm / MM_PER_INCH, 4)
            result["tpi"] = None
        else:
            result["thread_pitch_mm"] = None
        return result

    if unit_system == "imperial":
        return {
            "bolt_type": name,
            "unit_system": "imperial",
            "shank_diameter_mm": specs["shank"],
            "shank_diameter_in": round(specs["shank"] / MM_PER_INCH, 4),
            "wrench_size_mm": specs["wrench"],
            "wrench_size_in": round(specs["wrench"] / MM_PER_INCH, 4),
            "tpi": specs["tpi"],
        }

    return {
        "bolt_type": name,
        "unit_system": "metric",
        "shank_diameter_mm": specs["shank"],
        "wrench_size_mm": specs["wrench"],
        "thread_pitch_mm": specs["pitch"],
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
            label = (
                f"{bolt_info['bolt_type']}  "
                f"{bolt_info.get('bolt_width_mm', '?')}x{bolt_info.get('bolt_length_mm', '?')}mm"
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

        image = cv2.imread(img_path)
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
            scale = compute_scale(quarter_box, image)
            width_mm, length_mm = estimate_bolt_dimensions(bolt_box, scale, image)
            classification = classify_bolt(width_mm, unit_system)

            result.update({
                "bolt_type": classification["bolt_type"],
                "unit_system": classification["unit_system"],
                "bolt_width_mm": round(width_mm, 2),
                "bolt_length_mm": round(length_mm, 2),
                "wrench_size_mm": classification["wrench_size_mm"],
                "shank_diameter_mm": classification["shank_diameter_mm"],
            })

            if unit_system == "imperial":
                result.update({
                    "bolt_width_in": round(width_mm / MM_PER_INCH, 3),
                    "bolt_length_in": round(length_mm / MM_PER_INCH, 3),
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
    """Detects a bolt and a US quarter via the Roboflow Workflows API."""

    def __init__(self, api_key, workflow_url, min_confidence=0.25):
        self.api_key = api_key
        self.workflow_url = workflow_url
        self.min_confidence = float(min_confidence)
        logger.info(f"Roboflow analyzer ready. workflow={workflow_url}")

    def _extract_predictions(self, data):
        """Extract predictions list from the Roboflow API response."""
        if "predictions" in data and isinstance(data["predictions"], list):
            return data["predictions"]
        outputs = data.get("outputs", [])
        if not outputs:
            return []
        output = outputs[0] if isinstance(outputs, list) else outputs
        preds = output.get("predictions", output.get("result", []))
        if isinstance(preds, dict):
            preds = preds.get("predictions", [])
        return preds if isinstance(preds, list) else []

    def detect_objects(self, img_path, unit_system="metric"):
        if not os.path.isfile(img_path):
            return {"bolt": False, "quarter": False}

        image = cv2.imread(img_path)
        if image is None:
            return {"bolt": False, "quarter": False}

        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": {"type": "base64", "value": img_b64},
            },
        }

        try:
            resp = requests.post(
                self.workflow_url, json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Roboflow API error: {e}")
            return {"bolt": False, "quarter": False}

        predictions = self._extract_predictions(data)
        if not predictions:
            return {"bolt": False, "quarter": False}

        best_bolt = None
        best_quarter = None
        bolt_count = 0

        for pred in predictions:
            cx = pred["x"]
            cy = pred["y"]
            pw = pred["width"]
            ph = pred["height"]
            x1 = cx - pw / 2
            y1 = cy - ph / 2
            x2 = cx + pw / 2
            y2 = cy + ph / 2
            box = (float(x1), float(y1), float(x2), float(y2))
            conf = pred["confidence"]
            cls = pred["class"].lower()

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
            scale = compute_scale(quarter_box, image)
            width_mm, length_mm = estimate_bolt_dimensions(bolt_box, scale, image)
            classification = classify_bolt(width_mm, unit_system)

            result.update({
                "bolt_type": classification["bolt_type"],
                "unit_system": classification["unit_system"],
                "bolt_width_mm": round(width_mm, 2),
                "bolt_length_mm": round(length_mm, 2),
                "wrench_size_mm": classification["wrench_size_mm"],
                "shank_diameter_mm": classification["shank_diameter_mm"],
            })

            if unit_system == "imperial":
                result.update({
                    "bolt_width_in": round(width_mm / MM_PER_INCH, 3),
                    "bolt_length_in": round(length_mm / MM_PER_INCH, 3),
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
