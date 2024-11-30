import json
import sys
from pathlib import Path
import shutil

def load_results(results_dir):
    results = {}
    for sub_dir in Path(results_dir).iterdir():
        if sub_dir.is_dir():
            json_files = list(sub_dir.glob("*.json"))
            if json_files:
                json_file = json_files[0]
                with open(json_file, 'r') as f:
                    results[sub_dir.name[3:]] = json.load(f)
    return results

def compare_results(results):
    best_model = None
    best_score = float('-inf')  # Modifier en fonction de la métrique utilisée
    for model, metrics in results.items():
        score = metrics.get("accuracy", 0)  # Remplacez "accuracy" par la métrique utilisée
        if score > best_score:
            best_score = score
            best_model = model
    return best_model, best_score

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_results.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    results = load_results(results_dir)

    best_model, best_score = compare_results(results)

    model_best_path=Path("model/best-model")
    ev_best_path=Path("evaluation/best-model")

    model_best_path.mkdir(parents=True, exist_ok=True)
    ev_best_path.mkdir(parents=True, exist_ok=True)


    shutil.copytree("model/model-{}".format(best_model),model_best_path)
    shutil.copytree("evaluation/ev-{}".format(best_model),ev_best_path)

    
    print(f"Le meilleur modèle est : {best_model} avec un score de : {best_score:.6f}")