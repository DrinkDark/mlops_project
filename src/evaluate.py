import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import repository as rs
import os

def get_training_plot(model_history: dict) -> plt.Figure:
    """Plot the training and validation loss"""
    epochs = range(1, len(model_history["loss"]) + 1)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, model_history["loss"], label="Training loss")
    plt.plot(epochs, model_history["val_loss"], label="Validation loss")
    plt.xticks(epochs)
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    return fig


def get_pred_preview_plot(
    model: tf.keras.Model, ds_test: tf.data.Dataset, labels: List[str]
) -> plt.Figure:

    """Plot a preview of the predictions"""
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    for images, label_idxs in ds_test.take(1):
        preds = model.predict(images)
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            img = (images[i].numpy() * 255).astype("uint8")
            # Convert image to rgb if grayscale
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
                img = np.stack((img,) * 3, axis=-1)
            true_label = labels[int(label_idxs[i].numpy()[0])]
            pred_label = labels[np.argmax(preds[i])]
            # Add red border if the prediction is wrong else add green border
            img = np.pad(img, pad_width=((1, 1), (1, 1), (0, 0)))
            if true_label != pred_label:
                img[0, :, 0] = 255  # Top border
                img[-1, :, 0] = 255  # Bottom border
                img[:, 0, 0] = 255  # Left border
                img[:, -1, 0] = 255  # Right border
            else:
                img[0, :, 1] = 255
                img[-1, :, 1] = 255
                img[:, 0, 1] = 255
                img[:, -1, 1] = 255

            plt.imshow(img)
            plt.title(f"True: {true_label}\n" f"Pred: {pred_label}")
            plt.axis("off")

    return fig


def get_confusion_matrix_plot(conf_matrix) -> plt.Figure:
    """Plot a preview of the matrix"""
    labels = conf_matrix["labels"]
    cm = np.array(conf_matrix["matrix"])
    fig = plt.figure(figsize=(50, 50), tight_layout=True)
    plt.imshow(cm, cmap="Blues")
    # Plot cell values
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i, j]
            if value == 0:
                color = "lightgray"
            elif value > 0.5:
                color = "white"
            else:
                color = "black"
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")

    return fig

def   get_confusion_matrix_json(model: tf.keras.Model, ds_test: tf.data.Dataset, labels: List[str]) :
    """creat the confusion matrix json"""
   
    preds = model.predict(ds_test)

    conf_matrix = tf.math.confusion_matrix(
        labels=tf.concat([y for _, y in ds_test], axis=0),
        predictions=tf.argmax(preds, axis=1),
        num_classes=len(labels),
    )

    # Plot the confusion matrix
    conf_matrix = conf_matrix / tf.reduce_sum(conf_matrix, axis=1)
    cm = conf_matrix.numpy().tolist()
    cm_dict = {"labels": labels, "matrix": cm}
    return cm_dict


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 evaluate.py <model-folder> <prepared-dataset-folder>\n")
        exit(1)
    name_model = sys.argv[1]
    model_folder = Path("model") / Path(name_model)
    prepared_dataset_folder = Path(sys.argv[2])
    evaluation_folder = Path("evaluation") / Path(name_model)
    plots_folder = Path("plots")
   



    # Create folders
    (evaluation_folder / plots_folder).mkdir(parents=True, exist_ok=True)

    
    # Load files
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))
    labels = None
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)

    # load model to bentoml
    repr = rs.RepositoryModel(model_folder)
    if os.path.exists(os.path.join(model_folder, "name_model.json")):
        model = repr.import_load_model(name="bento-model")
        model_history = np.load(model_folder / "history.npy", allow_pickle=True).item()

        # Log metrics

        val_loss, val_acc = model.evaluate(ds_test)

        # Measure prediction time
        test_data = ds_test.take(1)

        predictions = model.predict(test_data)

        # Measure prediction time over 100 iterations
        times = []
        for predictions in range(100):
            start_time = time.time()
            predictions = model.predict(test_data)
            end_time = time.time()
            times.append(end_time - start_time)

        # Get batch size
        batch_size = 0
        for x_batch, y_batch in test_data:
            batch_size = x_batch.shape[0]

        # Calculate average prediction time in ms
        mean_prediction_time = np.mean(times) / batch_size * 1000

        conf_matrix = tf.math.confusion_matrix(
            labels=tf.concat([y for _, y in ds_test], axis=0),
            predictions=tf.argmax(model.predict(ds_test), axis=1),
            num_classes=len(labels)
        ).numpy()  # Convert to numpy for easier slicing

        # Initialize metric dictionaries

        metrics = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
        total_samples = np.sum(conf_matrix)
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        total_recall = 0
        total_fpr = 0
        num_classes = conf_matrix.shape[0]

        for i in range(num_classes):  # Iterate over each class
            TP = conf_matrix[i, i]
            FN = np.sum(conf_matrix[i, :]) - TP
            FP = np.sum(conf_matrix[:, i]) - TP
            TN = total_samples - (TP + FP + FN)
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

            # Accumulate TPR and FPR

            total_recall += TPR
            total_fpr += FPR

        # Calculate averages

        recall = total_recall / num_classes

        FPR = total_fpr / num_classes

        # Calculate F1 Score

        f1_score = TP / (TP + 0.5 * (FP + FN))

        # Overfitting Tendency

        training_loss = model_history['loss'][-1]

        validation_loss = model_history['val_loss'][-1]

        overfitting_tendency = training_loss - validation_loss

        # Complexity (Number of Parameters)

        complexity = model.count_params()

        print(f"Validation loss: {val_loss:.2f}")

        print(f"Validation accuracy: {val_acc * 100:.2f}%")

        print(f"Average prediction time: {mean_prediction_time:.6f} ms")

        print(f"Recall: {recall:.2f}")

        print(f"False Positive Rate: {FPR:.2f}")

        print(f"F1 Score: {f1_score:.2f}")

        print(f"Overfitting tendency: {overfitting_tendency:.2f}")

        print(f"Complexity: {complexity:.2f} parmeters")

        with open(evaluation_folder / "metrics.json", "w") as f:

            json.dump({"val_loss": val_loss, "val_acc": val_acc, "recall": recall, "fpr": FPR, "f1_score": f1_score,
                    "overfitting_tendency": overfitting_tendency, "complexity": complexity, "mean_predictions_time": mean_prediction_time}, f)





        # Save training history plot
        fig = get_training_plot(model_history)
        fig.savefig(evaluation_folder / plots_folder / "training_history.png")

        # Save predictions preview plot
        fig = get_pred_preview_plot(model, ds_test, labels)
        fig.savefig(evaluation_folder / plots_folder / "pred_preview.png")

        # Save confusion matrix plot
        cm =  get_confusion_matrix_json(model, ds_test, labels)
        fig = get_confusion_matrix_plot(cm)

        fig.savefig(evaluation_folder / plots_folder / "confusion_matrix.png")

        repr.update_model_metadata(
            tag="bento-model",
            metadata={
                "val_loss": val_loss,
                "val_acc": val_acc,
                "recall": recall,
                "fpr": FPR,
                "f1_score": f1_score,
                "overfitting_tendency": overfitting_tendency,
                "complexity": complexity,
                "mean_predictions_time": mean_prediction_time,
                "model_history": model_history,
                "confusion_matrix":cm
            }
        )

  
    else:
        model = repr.load_model("bento-model")
        model_bento = repr.get_model("bento-model")
        metadata=model_bento.info.metadata
        with open(evaluation_folder / "metrics.json", "w") as f:
            json.dump({"val_loss": metadata["val_loss"], "val_acc": metadata["val_acc"], "recall": metadata["recall"], "fpr": metadata["fpr"], "f1_score": metadata["f1_score"],
                 "overfitting_tendency": metadata["overfitting_tendency"], "complexity": metadata["complexity"], "mean_predictions_time": metadata["mean_predictions_time"]}, f)

        #add the name_model.json
        with open(model_folder / "name_model.json", "w") as f:
            json.dump({"name_model": str(model_bento.tag)}, f)
        # Save training history plot
        fig = get_training_plot(metadata["model_history"])
        fig.savefig(evaluation_folder / plots_folder / "training_history.png")

        # Save predictions preview plot
        fig = get_pred_preview_plot(model, ds_test, labels)
        fig.savefig(evaluation_folder / plots_folder / "pred_preview.png")

        # Save confusion matrix plot
        fig = get_confusion_matrix_plot(metadata["confusion_matrix"])
        fig.savefig(evaluation_folder / plots_folder / "confusion_matrix.png")
    
    print(
        f"\nEvaluation metrics and plot files saved at {evaluation_folder.absolute()}"
    )

if __name__ == "__main__":
    main()
