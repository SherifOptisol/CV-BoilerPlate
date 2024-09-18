import cv2
import numpy as np
import torch
from torchvision import transforms

def augment_image(image):
    # Define the augmentations
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    ])
    
    # Convert image to PIL format for torchvision transforms
    image_pil = transforms.ToPILImage()(image)
    
    # Apply augmentation
    image_augmented = augmentation(image_pil)
    
    # Convert back to tensor for model processing
    return transforms.ToTensor()(image_augmented)

# Example Usage
def load_and_augment_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = augment_image(torch.tensor(image).permute(2, 0, 1))  # Convert to tensor
    return image

if __name__ == "__main__":
    image = load_and_augment_image("data/sample_image.jpg")
