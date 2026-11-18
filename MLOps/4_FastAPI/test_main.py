from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_read_and_update_item():
    payload = {"name": "Daisy Pot", "price": 12.5, "is_offer": True}
    put_res = client.put("/items/3", json=payload)
    assert put_res.status_code == 200
    assert put_res.json()["item_name"] == "Daisy Pot"

    get_res = client.get("/items/3?q=fresh")
    assert get_res.status_code == 200
    assert get_res.json()["q"] == "fresh"
    assert get_res.json()["item"]["name"] == "Daisy Pot"


def test_predict_flower():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_species"] == "setosa"
