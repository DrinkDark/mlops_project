import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
import yaml

from utils.seed import set_seed
def get_model_from_config(image_shape: Tuple[int, int, int], config: dict) -> tf.keras.Model:
    """Create a CNN model based on YAML configuration."""
    inputs = tf.keras.Input(shape=image_shape)
    x=inputs

    model = tf.keras.models.Sequential()
    # Add convolutional layers
    for layer in config["layer"]:
        branches=[]
        for branche in layer["branche"]:
            for layer_type, params in branche.items():
                branch=x
                if layer_type == "conv_layers":
                    for param in params:
                        branch = tf.keras.layers.Conv2D(
                            filters=param["filters"],
                            kernel_size=tuple(param["kernel_size"]),
                            activation=param["activation"],
                        )(branch)
                elif layer_type == "max_pool":
                    for param in params:
                        branch = tf.keras.layers.MaxPooling2D(
                                pool_size=tuple(param["pool_size"])
                            )(branch)
                elif layer_type == "flatten":
                    branch = tf.keras.layers.Flatten()(branch)
                elif layer_type == "dense_layers":
                    branch = tf.keras.layers.Flatten()(branch)
                    for param in params:
                        branch = tf.keras.layers.Dense(
                            units=param["units"],
                            activation=param["activation"],
                        )(branch)
                elif layer_type == "output_classes":
                    branch = tf.keras.layers.Dense(
                        units=params,
                        activation=None,  # Use softmax during compilation
                    )(branch)
                    branches.append(branch)
            if len(branches)> 1:
                x = tf.keras.layers.Concatenate()(branches)
            else:
                x = branches[0]
        model = tf.keras.Model(inputs=inputs, outputs=x)
    return model




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

    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    train_params = params["train"]
    model_configs = params["model"]

    prepared_dataset_folder = Path(sys.argv[1])
    model_folder = Path("model") / Path(sys.argv[2])
    print(f"Training model: {sys.argv[2]}")
    model_v = (sys.argv[2])

    if model_v not in model_configs:
        print(f"Error: Model version '{model_v}' not defined in params.yaml.")
        exit(1)
    model_config=model_configs[model_v]

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

    #model = get_model(image_shape, conv_size, dense_size, output_classes,model_v)
    model = get_model_from_config(image_shape, model_config)

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
