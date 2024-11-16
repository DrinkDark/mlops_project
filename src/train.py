import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
import yaml

from utils.seed import set_seed


def get_model(
    image_shape: Tuple[int, int, int],
    conv_size: int,
    dense_size: int,
    output_classes: int,
    model_v: int,
) -> tf.keras.Model:
    """Create a simple CNN model"""
    if(model_v==1):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    conv_size, (3, 3), activation="relu", input_shape=image_shape
                ),
                tf.keras.layers.MaxPooling2D((3, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(dense_size, activation="relu"),
                tf.keras.layers.Dense(output_classes),
            ]
        )
    elif(model_v==2):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    conv_size, (5, 5), activation="relu", input_shape=image_shape
                ),
                tf.keras.layers.MaxPooling2D((5, 5)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(dense_size, activation="relu"),
                tf.keras.layers.Dense(output_classes),
            ]
        )
    else:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    conv_size, (3, 3), activation="relu", input_shape=image_shape
                ),
                tf.keras.layers.MaxPooling2D((3, 3)),
                tf.keras.layers.Conv2D(
                    conv_size, (3, 3), activation="relu", input_shape=image_shape
                ),
                tf.keras.layers.MaxPooling2D((3, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(dense_size, activation="relu"),
                tf.keras.layers.Dense(output_classes),
            ]
        )
    return model


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 train.py <prepared-dataset-folder> <model-folder>\n")
        exit(1)

    # Load parameters
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]
    train_params = yaml.safe_load(open("params.yaml"))["train"]

    prepared_dataset_folder = Path(sys.argv[1])
    model_folder = Path("models") / Path(sys.argv[2])
    print(f"Training model: {sys.argv[1]}")
    model_v = int((sys.argv[1])[-1])

    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    image_shape = (*image_size, 1 if grayscale else 3)

    seed = train_params["seed"]
    lr = train_params["lr"]
    epochs = train_params["epochs"]
    conv_size = train_params["conv_size"]
    dense_size = train_params["dense_size"]
    output_classes = train_params["output_classes"]

    # Set seed for reproducibility
    set_seed(seed)

    # Load data
    ds_train = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))
    ds_val = tf.data.Dataset.load(str(prepared_dataset_folder / "val"))
    # Define the model

    model = get_model(image_shape, conv_size, dense_size, output_classes,model_v)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()


    # Train the model
    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_val,
    )

    # Save the model
    model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder / "model.keras"
    model.save(model_path)
    # Save the model history
    np.save(model_folder / "history.npy", model.history.history)

    print(f"\nModel saved at {model_folder.absolute()}")


if __name__ == "__main__":
    main()
