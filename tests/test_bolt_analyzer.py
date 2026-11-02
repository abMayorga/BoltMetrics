import unittest

import cv2
import numpy as np

from bolt_analyzer import (
    QUARTER_DIAMETER_MM,
    RoboflowAnalyzer,
    classify_bolt,
    compute_calibration,
    estimate_bolt_dimensions,
)


class CalibrationTests(unittest.TestCase):
    def test_circle_produces_known_scale(self):
        image = np.full((320, 320, 3), 255, dtype=np.uint8)
        cv2.circle(image, (160, 160), 100, (20, 20, 20), -1)

        calibration = compute_calibration((60, 60, 260, 260), image)

        self.assertAlmostEqual(calibration.isotropic_mm_per_px, QUARTER_DIAMETER_MM / 200, delta=0.005)
        self.assertGreater(calibration.quarter_axis_ratio, 0.95)

    def test_ellipse_records_camera_foreshortening(self):
        image = np.full((360, 420, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (210, 180), (100, 70), 25, 0, 360, (20, 20, 20), -1)
        points = cv2.ellipse2Poly((210, 180), (100, 70), 25, 0, 360, 2)
        x, y, width, height = cv2.boundingRect(points)

        calibration = compute_calibration((x, y, x + width, y + height), image)

        self.assertEqual(calibration.method, "ellipse")
        self.assertAlmostEqual(calibration.quarter_axis_ratio, 0.70, delta=0.04)


class MeasurementTests(unittest.TestCase):
    def test_profile_separates_shank_from_head(self):
        image = np.full((240, 620, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (80, 110), (480, 130), (20, 20, 20), -1)
        cv2.rectangle(image, (480, 90), (495, 150), (20, 20, 20), -1)

        width, length, details = estimate_bolt_dimensions(
            (75, 85, 500, 155), 1.0, image, return_details=True
        )

        self.assertAlmostEqual(width, 21.0, delta=2.0)
        self.assertAlmostEqual(length, 401.0, delta=12.0)
        self.assertTrue(details["head_detected"])
        self.assertGreater(details["head_width_mm"], width * 2)


class ClassificationTests(unittest.TestCase):
    def test_metric_uses_shank_diameter(self):
        result = classify_bolt(6.1, "metric")
        self.assertEqual(result["bolt_type"], "M6")
        self.assertEqual(result["wrench_size_mm"], 10.0)

    def test_m7_is_supported(self):
        self.assertEqual(classify_bolt(7.1)["bolt_type"], "M7")

    def test_imperial_reference_size(self):
        result = classify_bolt(6.35, "imperial")
        self.assertEqual(result["bolt_type"], '1/4"')
        self.assertEqual(result["tpi"], 20)

    def test_unknown_does_not_claim_wrench_measurement(self):
        result = classify_bolt(50.0)
        self.assertEqual(result["bolt_type"], "Unknown")
        self.assertIsNone(result["wrench_size_mm"])


class RoboflowResponseTests(unittest.TestCase):
    def test_extracts_nested_workflow_predictions(self):
        analyzer = RoboflowAnalyzer.__new__(RoboflowAnalyzer)
        predictions = [{"class": "Bolt", "x": 10, "y": 10, "width": 5, "height": 5, "confidence": 0.9}]
        payload = [{"predictions": {"predictions": predictions}}]
        self.assertEqual(analyzer._extract_predictions(payload), predictions)


if __name__ == "__main__":
    unittest.main()
