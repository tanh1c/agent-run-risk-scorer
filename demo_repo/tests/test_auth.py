from fastapi.testclient import TestClient

from app.main import app


def test_login_accepts_valid_token():
    client = TestClient(app)
    response = client.post("/login", json={"token": "valid-token"})
    assert response.status_code == 200
    assert response.json() == {"message": "Login successful"}


def test_login_rejects_invalid_token():
    client = TestClient(app)
    response = client.post("/login", json={"token": "bad-token"})
    assert response.status_code == 401
