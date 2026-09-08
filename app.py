import os
import uuid
import logging

from dotenv import load_dotenv
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from bolt_analyzer import BoltAnalyzer, RoboflowAnalyzer, find_latest_model

load_dotenv()

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32).hex()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_UPLOAD_MB = 16
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE")
ROBOFLOW_WORKFLOW_ID = os.environ.get("ROBOFLOW_WORKFLOW_ID")
ANALYZER_BACKEND = os.environ.get("ANALYZER_BACKEND", "auto").strip().lower()
if ANALYZER_BACKEND not in {"auto", "local", "roboflow"}:
    logger.warning("Unknown ANALYZER_BACKEND=%r; using auto", ANALYZER_BACKEND)
    ANALYZER_BACKEND = "auto"

roboflow_configured = ROBOFLOW_API_KEY and ROBOFLOW_WORKSPACE and ROBOFLOW_WORKFLOW_ID
use_roboflow = ANALYZER_BACKEND == "roboflow" or (
    ANALYZER_BACKEND == "auto" and roboflow_configured
)

if use_roboflow and roboflow_configured:
    analyzer = RoboflowAnalyzer(ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID, min_confidence=0.25)
    print(f"Using Roboflow API: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_WORKFLOW_ID}")
elif use_roboflow:
    analyzer = None
    logger.error("Roboflow backend selected but its API settings are incomplete")
else:
    MODEL_PATH = os.environ.get("MODEL_PATH") or find_latest_model(BASE_DIR)
    if MODEL_PATH:
        try:
            analyzer = BoltAnalyzer(MODEL_PATH, min_confidence=0.25)
            print(f"Local model loaded: {MODEL_PATH}")
        except FileNotFoundError:
            analyzer = None
            print(f"[WARNING] Model not found at: {MODEL_PATH}")
            print("Run: python scripts/train.py")
    else:
        analyzer = None
        print("[WARNING] No trained model found in runs/detect/")
        print("Run: python scripts/train.py")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        if analyzer is None:
            result = {"error": "Model not loaded. Train the model first: python scripts/train.py"}
            return render_template("index.html", result=result)

        file = request.files.get("bolt_image")

        if file and allowed_file(file.filename):
            safe_name = secure_filename(file.filename)
            ext = safe_name.rsplit(".", 1)[1].lower() if "." in safe_name else "jpg"
            filename = f"{uuid.uuid4().hex}.{ext}"
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            unit_system = request.form.get("unit_system", "metric")
            if unit_system not in ("metric", "imperial"):
                unit_system = "metric"
            try:
                result = analyzer.detect_objects(path, unit_system=unit_system)
                if result.get("service_error"):
                    result = {"error": "Detection service is temporarily unavailable. Please try again in a moment."}
            except Exception:
                logger.exception("Bolt analysis failed")
                result = {"error": "Analysis failed. Please try another photo."}
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        else:
            result = {"error": "Invalid file. Please upload a JPG, PNG, or GIF."}

    return render_template("index.html", result=result)


@app.errorhandler(413)
def upload_too_large(_error):
    return render_template(
        "index.html",
        result={"error": f"Image is too large. Maximum upload size is {MAX_UPLOAD_MB} MB."},
    ), 413


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)
