import yaml
import json
from pathlib import Path


def extract_matrix(yaml_file: Path) -> None:
    """
    Check the number of model to use in the params.yaml
    and print for the github action matrix
    """
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    models = data.get("model", {})
    matrix = [
        {"model": model}  
        for model, details in models.items()
    ]
    print(json.dumps({"include": matrix}, indent=2))


if __name__ == "__main__":
    yaml_file = Path("params.yaml")
    extract_matrix(yaml_file)
