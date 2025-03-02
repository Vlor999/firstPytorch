#! /usr/bin/env python3 

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from chargementModele import chargementModel

IMAGE_PATH = "src/data/images/image-cinq.jpg"

def load_and_preprocess_image(image_path, is_grey):
    """
    Charge une image et applique les transformations nécessaires pour le modèle.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1) if is_grey else transforms.Lambda(lambda x: x),  # Convertir en niveaux de gris si nécessaire
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if is_grey else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert("L" if is_grey else "RGB")  # Convertir en niveaux de gris ou couleur
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def predict(image_path):
    """
    Charge le modèle, traite l'image et effectue une prédiction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, is_grey = chargementModel()

    image = load_and_preprocess_image(image_path, is_grey)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    print(f"Le modèle prédit : {predicted.item()}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    predict(IMAGE_PATH)
