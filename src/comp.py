import json
import sys
from pathlib import Path

def load_results(results_dir):
    results = {}
    for result_file in Path(results_dir).glob("*.json"):
        with open(result_file, 'r') as f:
            results[result_file.stem] = json.load(f)
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
    if len(sys.argv) < 2:
        print("Usage: python compare_results.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    results = load_results(results_dir)

    best_model, best_score = compare_results(results)
    print(f"Le meilleur modèle est : {best_model} avec un score de : {best_score}")
