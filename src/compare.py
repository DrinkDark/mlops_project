import json
import sys
from pathlib import Path
import shutil
import yaml

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

def normalize_results(results, optimization_directions):
    """
    Normalize the metrics across all models using 0-max normalization,
    considering whether each metric should be maximized or minimized.
    """
    # Collect all metrics keys and initialize min/max dictionaries
    all_metrics = {metric for model in results.values() for metric in model.keys()}
    min_values = {metric: float('inf') for metric in all_metrics}
    max_values = {metric: float('-inf') for metric in all_metrics}

    # Find min and max for each metric
    for model_metrics in results.values():
        for metric, value in model_metrics.items():
            if metric in min_values:
                min_values[metric] = min(min_values[metric], value)
                max_values[metric] = max(max_values[metric], value)

    # Normalize metrics for each model
    normalized_results = {}
    for model, model_metrics in results.items():
        normalized_results[model] = {}
        for metric in all_metrics:
            value = model_metrics.get(metric, 0)  # Default to 0 if missing
            min_val, max_val = min_values[metric], max_values[metric]

            if max_val > min_val:  # Avoid division by zero
                if optimization_directions.get(metric, "maximize") == "maximize":
                    normalized_value = (value-0) / (max_val-0)
                else:
                    normalized_value = ((1/value)-0) / ((1/min_val)-0)
            else:
                normalized_value = 0

            normalized_results[model][metric] = normalized_value

    return normalized_results

def load_metrics_config():
    config = yaml.safe_load(open("evaluation_metrics.yaml"))
    return config["evaluation_metrics"], config["optimization_directions"]

def compare_results(results, weights):
    """
    Compare models based on weighted and normalized metrics, considering directions.
    """
    best_model = None
    best_score = float('-inf')

    # Iterate through each model and calculate weighted scores
    for model, model_metrics in results.items():
        score = 0
        for metric, value in model_metrics.items():
            if metric in weights and weights[metric] is not None:
                score += weights[metric] * value

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
    weights, directions = load_metrics_config()
    normalize_results = normalize_results(results, directions)

    best_model, best_score = compare_results(normalize_results, weights)

    model_best_path=Path("model/best-model")
    ev_best_path=Path("evaluation/best-model")

    shutil.copytree("model/model-{}".format(best_model),model_best_path,dirs_exist_ok=True)
    shutil.copytree("evaluation/ev-{}".format(best_model),ev_best_path,dirs_exist_ok=True)

    
    print(f"The best model is : {best_model} with a score of : {best_score:.6f}")