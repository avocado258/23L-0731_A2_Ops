import io
import os

os.environ["DATABASE_URL"] = "sqlite://"   # in-memory DB; set before app.config is imported

import pytest
from PIL import Image

import app as app_module
from app.inference import preprocess_bytes


class FakePredictor:
    model_name = "visionguard-model:vTEST"

    def predict(self, data):
        preprocess_bytes(data)             # real decode + resize path, no model needed
        return {"label": "Real", "confidence": 0.9, "p_real": 0.9,
                "threshold": 0.5, "model": self.model_name}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "get_predictor", lambda: FakePredictor())
    application = app_module.create_app()
    application.config.update(TESTING=True)
    with application.test_client() as c:
        yield c


@pytest.fixture
def png():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 80, 200)).save(buf, "PNG")
    return buf.getvalue()