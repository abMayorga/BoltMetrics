import io
import unittest

import app as webapp


class StubAnalyzer:
    def __init__(self, result=None, error=None):
        self.result = result or {"bolt": False, "quarter": False}
        self.error = error

    def detect_objects(self, _path, unit_system="metric"):
        if self.error:
            raise self.error
        result = dict(self.result)
        result["requested_units"] = unit_system
        return result


class AppRouteTests(unittest.TestCase):
    def setUp(self):
        self.original_analyzer = webapp.analyzer
        webapp.app.config.update(TESTING=True)
        self.client = webapp.app.test_client()

    def tearDown(self):
        webapp.analyzer = self.original_analyzer

    def test_get_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"BoltMetrics", response.data)

    def test_rejects_unsupported_extension(self):
        webapp.analyzer = StubAnalyzer()
        response = self.client.post(
            "/",
            data={"bolt_image": (io.BytesIO(b"not an image"), "sample.gif")},
            content_type="multipart/form-data",
        )
        self.assertIn(b"Invalid file", response.data)

    def test_analysis_exception_is_not_exposed(self):
        webapp.analyzer = StubAnalyzer(error=RuntimeError("private service detail"))
        response = self.client.post(
            "/",
            data={"bolt_image": (io.BytesIO(b"image bytes"), "sample.jpg")},
            content_type="multipart/form-data",
        )
        self.assertIn(b"Analysis failed. Please try another photo.", response.data)
        self.assertNotIn(b"private service detail", response.data)


if __name__ == "__main__":
    unittest.main()
