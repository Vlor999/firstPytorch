#! /usr/bin/env python3 

import torch
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelisationNet.model import Net

DIR = "src/model/"
MODEL_CHIFFRE = "model-chiffre.pth"
MODEL_IMAGE = "model-image.pth"

def detect_model_file():
    """
    Détecte automatiquement le modèle disponible et retourne son chemin et son type.
    """
    model_files = [MODEL_CHIFFRE, MODEL_IMAGE]
    for model_file in model_files:
        path = os.path.join(DIR, model_file)
        if os.path.exists(path):
            is_grey = model_file == MODEL_CHIFFRE
            return path, is_grey
    print("Aucun modèle trouvé dans le dossier 'model/'.")
    exit(1)

def chargementModel():
    """
    Charge automatiquement le modèle en fonction du fichier détecté.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path, is_grey = detect_model_file()
    num_classes = 10 if is_grey else 10

    print(f"Chargement du modèle depuis {model_path}...")
    model = Net(num_classes=num_classes, isGrey=is_grey)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modèle chargé avec succès ({'Chiffres' if is_grey else 'Images'}) !")

    return model, is_grey

def main():
    model, is_grey = chargementModel()
    print("Modèle prêt à être utilisé.")

if __name__ == "__main__":
    main()
