#! /usr/bin/env python3 

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import sys
from chargementModele import chargementModel
from subprocess import run

IMAGE_PATH_FORMAT = "src/data/images/"
IMAGE_PATH = "src/data/images/Figure_2.jpg"

def foundInfo(format):
    listElem = format.split("-")
    try:
        expected, pos = int(listElem[2]), int(listElem[3][:-4])
        return expected, pos
    except ValueError :
        print("erreur conversion")
        return None, None

def goodFormat(elem):
    format = elem.split("-")
    return len(format) == 4 and format[0] == "pixil" and format[1] == "frame" 

def foundImages():
    output = run(["ls", "src/data/images"], capture_output=True)
    listImage = {}
    if output.returncode == 0:
        for elem in output.stdout.decode().split('\n'):
            if goodFormat(elem):
                numExpected, pos = foundInfo(elem)
                if numExpected is None:
                    continue
                if numExpected not in listImage:
                    listImage[numExpected] = [(elem, pos)]
                else:
                    listImage[numExpected].append((elem, pos))
    return listImage

def affiche(path):
    image = Image.open(path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

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

def predictDict(dicoImages):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, is_grey = chargementModel()

    for key in dicoImages:
        for tupleTitlePos in dicoImages[key]:
            pathImage, pos = tupleTitlePos
            image = load_and_preprocess_image(IMAGE_PATH_FORMAT + pathImage, is_grey)
            image = image.to(device)

            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
            
            print(f"Le modèle prédit : {predicted.item()} et la valeur attendu est {key}")


if __name__ == "__main__":
    dicoImages = foundImages()
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    predict(IMAGE_PATH)
    predictDict(dicoImages)
