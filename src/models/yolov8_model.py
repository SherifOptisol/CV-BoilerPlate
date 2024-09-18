from ultralytics import YOLO
import torch
import os
from src.utils import helper

logger = helper.setup_logger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YOLOv8Model:
    def __init__(self, model_name="yolov8n.pt", pretrained=True, epochs=10, data_path="data.yaml"):
        logger.info("Initializing YOLOv8 Model")
        self.model = YOLO(model_name)  # YOLOv8n is a small model; you can choose others like yolov8s, yolov8m, etc.
        self.epochs = epochs
        self.data_path = data_path

    def train(self):
        logger.info(f"Starting YOLOv8 training on {self.data_path} for {self.epochs} epochs.")
        try:
            self.model.train(data=self.data_path, epochs=self.epochs, device=device)
            logger.info("YOLOv8 training complete.")
        except Exception as e:
            logger.error(f"Error during YOLOv8 training: {e}")
            raise

    def validate(self):
        logger.info("Validating YOLOv8 model.")
        try:
            self.model.val()
            logger.info("YOLOv8 validation complete.")
        except Exception as e:
            logger.error(f"Error during YOLOv8 validation: {e}")
            raise

    def predict(self, image_path):
        logger.info(f"Running prediction on {image_path}")
        try:
            results = self.model(image_path)
            results.show()  # Displays the result with bounding boxes
            return results
        except Exception as e:
            logger.error(f"Error during YOLOv8 prediction: {e}")
            raise

# Example Usage:
if __name__ == "__main__":
    yolo_model = YOLOv8Model(model_name="yolov8n.pt", epochs=5, data_path="path/to/your/data.yaml")
    yolo_model.train()
    yolo_model.validate()
    yolo_model.predict("path/to/sample_image.jpg")
