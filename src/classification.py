from src.preprocessing.image_preprocessing import resize_image, normalize_image
from src.models.cnn_model import SimpleCNN
import torch

def main():
    # Load an example image
    image = resize_image("data/sample_image.jpg")
    image = normalize_image(image)

    # Convert image to tensor for PyTorch
    image_tensor = torch.tensor(image).float().unsqueeze(0).permute(0, 3, 1, 2)

    # Initialize model
    model = SimpleCNN()
    output = model(image_tensor)
    print(f"Model Output: {output}")

if __name__ == "__main__":
    main()
