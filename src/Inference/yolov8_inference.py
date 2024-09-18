from ultralytics import YOLO
from src.utils import helper
import torch
import cv2

logger = helper.setup_logger()

class YOLOInference:
    def __init__(self, model_path="best.pt", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model from {model_path} on {self.device}")
        self.model = YOLO(model_path)

    def predict(self, image_path, save_results=True, output_path="inference_results"):
        logger.info(f"Running inference on {image_path}")
        try:
            # Load the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not open or find the image: {image_path}")
            
            # Perform inference
            results = self.model(image)
            
            # Display results (bounding boxes and labels)
            results.show()

            if save_results:
                results.save(output_path)  # Save results to a folder
            
            return results
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

# Example usage:
if __name__ == "__main__":
    infer = YOLOInference(model_path="runs/train/exp/weights/best.pt")
    infer.predict("data/sample_image.jpg")
