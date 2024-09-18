from src.utils import helper
from ultralytics import YOLO

logger = helper.setup_logger()

class YOLOv8Model:
    def __init__(self, model_name="yolov8n.pt", pretrained=True, epochs=10, data_path="data.yaml"):
        self.model = YOLO(model_name)
        self.epochs = epochs
        self.data_path = data_path

    def export(self, format="onnx"):
        """Export the trained YOLOv8 model to a different format."""
        logger.info(f"Exporting model to {format} format")
        try:
            # Supported formats: "onnx", "torchscript", "coreml", "saved_model", "tflite", etc.
            export_path = self.model.export(format=format)
            logger.info(f"Model exported successfully to {export_path}")
        except Exception as e:
            logger.error(f"Error during model export: {e}")
            raise

# Example usage:
if __name__ == "__main__":
    yolo_model = YOLOv8Model(model_name="yolov8n.pt", epochs=5, data_path="path/to/data.yaml")
    yolo_model.export(format="onnx")  # Export to ONNX format


# Supported Formats: onnx, torchscript, coreml, saved_model (TensorFlow), tflite, etc.