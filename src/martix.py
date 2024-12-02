import yaml
import json
from pathlib import Path


def extract_matrix(yaml_file: Path) -> None:
    # Charger le fichier YAML
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # Construire la matrice en extrayant les modèles et leurs attributs
    models = data.get("model", {})
    matrix = [
        {"model": model}  # Ajoute les détails de chaque modèle dans la matrice
        for model, details in models.items()
    ]

    # Afficher la matrice au format JSON
    print(json.dumps({"include": matrix}, indent=2))


if __name__ == "__main__":
    yaml_file = Path("params.yaml")
    extract_matrix(yaml_file)
