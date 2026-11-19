import io

from app.inference import preprocess_bytes


def post(client, name, data):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_predict_and_history(client, png):
    r = post(client, "cat pic.png", png)
    assert r.status_code == 200 and b"Real" in r.data
    assert b"cat_pic.png" in client.get("/history").data     # sanitized name logged


def test_no_file(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_wrong_type(client):
    assert post(client, "notes.txt", b"hello").status_code == 415


def test_undecodable(client):
    assert post(client, "fake.jpg", b"\x00" * 500).status_code == 422


def test_too_large(client, png):
    client.application.config["MAX_CONTENT_LENGTH"] = 100
    assert post(client, "big.png", png).status_code == 413


def test_preprocess_shape(png):
    assert preprocess_bytes(png).shape == (224, 224, 3)