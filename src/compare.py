import json
import sys
from pathlib import Path
import shutil
import yaml
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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

def normalize_results(results):
    """
    Normalize the metrics across all models using min-max normalization.
    """
    all_metrics = {metric for model in results.values() for metric in model.keys()}

    metric_values = {metric: [] for metric in all_metrics}
    for model_metrics in results.values():
        for metric in all_metrics:
            metric_values[metric].append(model_metrics.get(metric, 0))  # Default to 0 if missing

    normalized_results = {}
    scalers = {}
    for metric, values in metric_values.items():
        scaler = MinMaxScaler()
        values = np.array(values).reshape(-1, 1)
        normalized = scaler.fit_transform(values).flatten()
        scalers[metric] = scaler  # Save scalers if needed for later
        metric_values[metric] = normalized

    for i, (model, model_metrics) in enumerate(results.items()):
        normalized_results[model] = {metric: metric_values[metric][i] for metric in all_metrics}

    return normalized_results

def load_metrics_config():
    config = yaml.safe_load(open("evaluation_metrics.yaml"))
    return config["metrics"]

def compare_results(results):
    metrics = load_metrics_config()

    best_model = None
    best_score = float('-inf')

    # Iterate through each model and calculate weighted scores
    for model, metrics in results.items():
        score = 0
        for metric, value in metrics.items():
            if metric in metrics and metrics[metric] is not None:
                score += metrics[metric] * value
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
    normalize_results = normalize_results(results)

    best_model, best_score = compare_results(normalize_results)

    model_best_path=Path("model/best-model")
    ev_best_path=Path("evaluation/best-model")

    shutil.copytree("model/model-{}".format(best_model),model_best_path,dirs_exist_ok=True)
    shutil.copytree("evaluation/ev-{}".format(best_model),ev_best_path,dirs_exist_ok=True)

    
    print(f"The best model is : {best_model} with a score of : {best_score:.6f}")