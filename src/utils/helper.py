import os
import cv2
import logging

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def setup_logger():
    # Setup logger
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("cv_boilerplate.log"),  # Logs to file
            logging.StreamHandler()  # Logs to console
        ]
    )
    return logging.getLogger()

if __name__ == "__main__":
    # Example Usage
    logger = setup_logger()
    logger.info("Starting Image Augmentation Process")
