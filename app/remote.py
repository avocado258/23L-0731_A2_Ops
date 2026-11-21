import requests

from app import config
from app.inference import ImageDecodeError


class RemoteError(Exception):
    pass


class RemotePredictor:
    def __init__(self, url, key, timeout):
        self.url, self.key, self.timeout = url, key, timeout
        self.model_name = f"remote:{url}"          # used only by /health

    def predict(self, data: bytes) -> dict:
        try:
            r = requests.post(
                f"{self.url}/predict",
                headers={"X-API-Key": self.key},
                files={"image": ("upload.jpg", data)},   # the API decodes by content, not by name
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise RemoteError(f"API unreachable: {e}") from e
        if r.status_code == 422:
            raise ImageDecodeError("API could not decode the image")
        if r.status_code != 200:
            raise RemoteError(f"API returned {r.status_code}")
        return r.json()


def build_remote():
    return RemotePredictor(config.INFERENCE_API_URL, config.INFERENCE_API_KEY,
                           config.INFERENCE_API_TIMEOUT)