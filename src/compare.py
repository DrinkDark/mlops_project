import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import yaml
import repository as rs
import numpy as np
import os
import report

def load_results(results_dir,repo:rs.RepositoryModel):
    results = {}
    for sub_dir in Path(results_dir).iterdir():
        if sub_dir.is_dir():
            repo.import_model("bento-model","model/model-"+sub_dir.name[3:])
           
            json_files = list(sub_dir.glob("*.json"))
            if json_files:
                json_file = json_files[0]
                with open(json_file, 'r') as f:
                    results[sub_dir.name[3:]] = json.load(f)
    return results

def udate_list_model():
    root_dir="model"
    list_model_path=root_dir+"/list_model.json"
    if os.path.exists(list_model_path):
        if os.path.getsize(list_model_path) != 0:
            with open(list_model_path, "r", encoding="utf-8") as file:
                model_list = json.load(file)
        else:
            model_list = []
    else:
        model_list = []
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        if os.path.isdir(subdir_path):
            name_model_path = os.path.join(subdir_path, "name_model.json")

            if os.path.exists(name_model_path):
                with open(name_model_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                name_model = data.get("name_model")
                if name_model and name_model not in model_list:
                    model_list.append(name_model)

    with open(list_model_path, "w", encoding="utf-8") as file:
        json.dump(model_list, file, ensure_ascii=False, indent=4)


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

     # Prepare data for heatmap
    models =[]
    for name in results.keys():
        json_files = list(Path("model/model-"+name).glob("*.json"))
        json_file = json_files[0]
        with open(json_file, 'r') as f:
                name_file = json.load(f)
        name_model = name_file.get("name_model")
        models.append(name_model)
    metrics = list(weights.keys())
    valid_metrics = [metric for metric in metrics if metric in weights and weights[metric] is not None]
    # Create a matrix to store weighted values
    weighted_matrix = []

    # Iterate through each model and calculate weighted scores
    for model, model_metrics in results.items():
        row = []
        score = 0
        for metric  in valid_metrics:
            if metric in model_metrics:
                weighted_value = weights[metric] * model_metrics[metric]
                row.append(weighted_value)
                score += weighted_value

        row.append(score)
        weighted_matrix.append(row)
        if score > best_score:
            best_score = score
            best_model = model

    # Add metric labels including 'Total'
    metrics_with_total = valid_metrics + ['Total']

    # Convert to numpy array for visualization
    weighted_matrix = np.array(weighted_matrix).T  # Transpose for metrics in rows

    # Plot heatmap manually
    fig_width = len(models) * 1.2  
    fig_height = len(metrics_with_total) * 1.2 + 8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cax = ax.matshow(weighted_matrix, cmap="coolwarm")

    # Add color bar
    fig.colorbar(cax, label='Weighted Value')

    # Set axis labels
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(metrics_with_total)))
    ax.set_xticklabels(models, rotation=45, ha="left")
    ax.set_yticklabels(metrics_with_total)


    # Annotate each cell with its value
    for i in range(len(metrics_with_total)):
        for j in range(len(models)):
            ax.text(j, i, f"{weighted_matrix[i, j]:.2f}", va='center', ha='center', color="black")

    plt.title('Weighted Metrics Heatmap')
    plt.xlabel('Models')
    plt.ylabel('Metrics')

    # Save plot as PNG
    plt.savefig("evaluation/weighted_metrics_heatmap.png")
    plt.close()

    return best_model, best_score

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_results.py <results_dir>")
        sys.exit(1)

    repo=rs.RepositoryModel()
    udate_list_model()
    with open("model/list_model.json", "r", encoding="utf-8") as file:
            model_list = json.load(file)

    results_dir = sys.argv[1]
    results = load_results(results_dir,repo=repo)
    weights, directions = load_metrics_config()
    normalize_results = normalize_results(results, directions)

    best_model, best_score = compare_results(normalize_results, weights)
    report.creat_report()

    model_best_path=Path("model/model-modelBest")
    ev_best_path=Path("evaluation/ev-modelBest")

    print(f"The best model is : {best_model} with a score of : {best_score:.6f}")
    if best_model !="modelBest":
        shutil.copytree("model/model-{}".format(best_model),model_best_path,dirs_exist_ok=True)
        shutil.copytree("evaluation/ev-{}".format(best_model),ev_best_path,dirs_exist_ok=True)

    
   