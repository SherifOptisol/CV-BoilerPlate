import cv2
import numpy as np
from src.utils import helper

logger = helper.setup_logger()
logger.info("Starting Image Preprocessing")

def load_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or invalid format: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise

def resize_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, size)
    return resized

def normalize_image(image):
    return image / 255.0
