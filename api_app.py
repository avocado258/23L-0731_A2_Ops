import hmac
import os

from flask import Flask, jsonify, request

from app.inference import ImageDecodeError, get_predictor

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
API_KEY = os.environ.get("API_KEY", "")


def create_api():
    api = Flask(__name__)
    api.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    predictor = get_predictor()                     # loaded once at startup

    @api.get("/health")
    def health():
        return {"status": "ok", "model": predictor.model_name}

    @api.post("/predict")
    def predict():
        key = request.headers.get("X-API-Key", "")
        if not API_KEY or not hmac.compare_digest(key.encode(), API_KEY.encode()):
            return jsonify(error="invalid or missing API key"), 401
        f = request.files.get("image")
        if f is None or not f.filename:
            return jsonify(error="send a file in form-data field 'image'"), 400
        if f.filename.rsplit(".", 1)[-1].lower() not in ALLOWED_EXT:
            return jsonify(error="unsupported file type"), 415
        try:
            return jsonify(predictor.predict(f.stream.read()))
        except ImageDecodeError:
            return jsonify(error="image could not be decoded"), 422

    @api.errorhandler(413)
    def too_large(_):
        return jsonify(error="file too large"), 413

    return api


if __name__ == "__main__":
    create_api().run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=False)