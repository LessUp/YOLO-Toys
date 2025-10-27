from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert "info" in data
    assert "recommended_models" in data


def test_models():
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data.get("models"), list)
