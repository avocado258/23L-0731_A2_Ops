import logging

from flask import Flask, render_template, request
from sqlalchemy import text
from werkzeug.utils import secure_filename

from app import config
from app.db import Prediction, db
from app.inference import ImageDecodeError, get_predictor

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

def create_app():
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=config.DATABASE_URL,
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_BYTES,
    )
    db.init_app(app)
    with app.app_context():
        db.create_all()
    app.predictor = get_predictor()          # loaded once at startup

    def error(code, title, msg):
        return render_template("error.html", code=code, title=title, msg=msg), code

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        f = request.files.get("image")
        if f is None or not f.filename:
            return error(400, "No file selected", "Please choose an image to upload.")
        name = secure_filename(f.filename) or "upload"
        if name.rsplit(".", 1)[-1].lower() not in ALLOWED_EXT:
            return error(415, "Unsupported file type",
                         "Allowed types: " + ", ".join(sorted(ALLOWED_EXT)) + ".")
        data = f.stream.read()                   # in memory only; nothing written to disk
        try:
            res = app.predictor.predict(data)
        except ImageDecodeError:
            return error(422, "Image could not be read",
                         "The file looks like an image but is corrupt or cannot be decoded.")
        db.session.add(Prediction(filename=name, label=res["label"],
                                  confidence=res["confidence"], p_real=res["p_real"],
                                  threshold=res["threshold"], model_name=res["model"]))
        db.session.commit()
        return render_template("result.html", r=res, filename=name)

    @app.get("/history")
    def history():
        rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(200).all()
        return render_template("history.html", rows=rows)

    @app.get("/health")
    def health():
        try:
            db.session.execute(text("SELECT 1"))
            return {"status": "ok", "model": app.predictor.model_name, "db": "ok"}, 200
        except Exception:
            app.logger.exception("health check failed")
            return {"status": "unhealthy", "db": "down"}, 503

    @app.errorhandler(413)
    def too_large(_):
        return error(413, "File too large",
                     f"Maximum size is {config.MAX_UPLOAD_BYTES // (1024*1024)} MB.")

    @app.errorhandler(404)
    def nf(_):
        return error(404, "Page not found", "That page does not exist.")

    @app.errorhandler(500)
    def ise(_):
        return error(500, "Something went wrong", "Please try again later.")

    logging.basicConfig(level=logging.INFO)
    return app