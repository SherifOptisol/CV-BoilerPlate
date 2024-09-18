import cv2
from src.preprocessing.image_preprocessing import resize_image

def test_resize_image():
    image_path = "tests/test_image.jpg"  # Add a sample image in tests
    resized_image = resize_image(image_path, (224, 224))
    assert resized_image.shape == (224, 224, 3)
