from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_predict_endpoint():
    # Test with a sample image
    image_path = "path/to/sample_image.jpg"
    with open(image_path, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("filename.jpg", image_file, "image/jpeg")}
        )
    assert response.status_code == 200
    assert "results" in response.json()
