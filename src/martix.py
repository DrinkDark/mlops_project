import yaml
import json
from pathlib import Path


def extract_models(yaml_file: Path) -> None:
    # Charger le fichier YAML
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # Extraire les modèles sous 'model'
    models = list(data.get("model", {}).keys())

    # Générer la liste au format JSON
    matrix = {{"model": model} for model in models}

    # Afficher le JSON pour GitHub Actions
    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    yaml_file = Path("models.yaml")
    extract_models(yaml_file)
