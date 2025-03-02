#! /usr/bin/env python3

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Gestion des arguments pour le modèle d'IA.")
    
    parser.add_argument("-c", action="store_true", help="Modèle pour les chiffres")
    parser.add_argument("-s", action="store_true", help="Sauvegarde du modèle")
    parser.add_argument("-i", action="store_true", help="Modèle pour les images")

    return parser.parse_args()
