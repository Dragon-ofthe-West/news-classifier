from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "metrics" in r.json()


def test_predict_ok():
    r = client.post("/predict", json={"text": "Stocks rally as oil prices fall"})
    assert r.status_code == 200
    body = r.json()
    assert body["label_name"] in ["World", "Sports", "Business", "Sci-Tech"]
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_invalid_empty():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_invalid_type():
    r = client.post("/predict", json={"text": 123})
    assert r.status_code == 422
