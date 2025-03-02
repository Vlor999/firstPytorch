#! /usr/bin/env python3

# import 
from sys import argv

def creationDicoAction()->dict:
    dicoAction = {
        "nom" : "",
        "-c" : [False, "Modèle pour les chiffres"],
        "-s" : [False, "Sauvegarde du modèle"],
        "-i" : [True, "Modèle pour les images"],
        "-help" : [False, "Information sur la ligne de commande"]
    }
    return dicoAction

def updateDico(argv, dicoAction):
    name = argv[0]
    dicoAction["nom"] = name
    argv = set(argv[1:])
    for arg in argv:
        if arg in dicoAction:
            dicoAction[arg][0] = True
        else :
            print("Argument : " + arg + " inconnue")
            exit(1)
    
def displayInfo(dicoAction):
    output = dicoAction["nom"] + " "
    for key in dicoAction:
        if key != "nom":
            output += f"[{key}] "
    output += "\n"
    for key in dicoAction:
        if key != "nom":
            output += f"{key} : {dicoAction[key][1]}\n"
    print(output)

def gereArgs(dicoAction):
    for cle in dicoAction:
        isActivated = dicoAction[cle][0]
        if isActivated and cle == "-help":
            displayInfo(dicoAction)

def main():
    dicoAction = creationDicoAction()
    updateDico(argv, dicoAction)
    gereArgs(dicoAction)

if __name__ == "__main__":
    main()    
