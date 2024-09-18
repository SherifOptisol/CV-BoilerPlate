from fastapi import FastAPI, UploadFile, File
import uvicorn
from src.inference.yolov8_inference import YOLOInference
import shutil
import os

app = FastAPI()

# Path to save uploaded images
UPLOAD_FOLDER = "uploaded_images/"

# Initialize YOLOv8 Inference class
yolo_infer = YOLOInference(model_path="runs/train/exp/weights/best.pt")

# Ensure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Perform inference using YOLOv8
        results = yolo_infer.predict(image_path)

        # Return results (for simplicity, returning string representation of results)
        return {"filename": file.filename, "results": str(results)}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn src.api.app:app --reload

# python app.py

# curl -X 'POST' \
#   'http://127.0.0.1:8000/predict/' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: multipart/form-data' \
#   -F 'file=@path_to_your_image.jpg'
