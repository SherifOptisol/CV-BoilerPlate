from src.models.yolov8_model import YOLOv8Model

def main():
    # Initialize YOLOv8 model
    yolo_model = YOLOv8Model(model_name="yolov8n.pt", epochs=10, data_path="path/to/your/data.yaml")
    
    # Train the model
    yolo_model.train()
    
    # Validate the model
    yolo_model.validate()

    # Test on a sample image
    yolo_model.predict("path/to/sample_image.jpg")

if __name__ == "__main__":
    main()
