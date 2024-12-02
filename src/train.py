import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
import yaml

from utils.seed import set_seed
def get_model_from_config(image_shape: Tuple[int, int, int], config: dict) -> tf.keras.Model:
    """Create a CNN model based on YAML configuration."""

    key_config,val_config = next(iter(config.items()))

    if key_config == "ResNet50":
        model=tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_shape=image_shape,
            classes=val_config,
        )

    else:
        inputs = tf.keras.Input(shape=image_shape)
        x = inputs
        model = tf.keras.models.Sequential()
        for layer in config["layer"]:
            branches=[]
            branch = x
            for branche in layer["branche"]:
                for branche_type in branche.items():
                    print(branche_type)
                    if branche_type[0] == "conv_layers":
                        branch = tf.keras.layers.Conv2D(
                            filters=branche_type[1]["filters"],
                            kernel_size=tuple(branche_type[1]["kernel_size"]),
                            activation=branche_type[1]["activation"],
                        )(branch)
                    elif branche_type[0] == "max_pool":
                        branch = tf.keras.layers.MaxPooling2D(
                                pool_size=tuple(branche_type[1]["pool_size"])
                            )(branch)
                    elif branche_type[0] == "flatten":
                        branch = tf.keras.layers.Flatten()(branch)
                    elif branche_type[0] == "dense_layers":
                        branch = tf.keras.layers.Dense(
                            units=branche_type[1]["units"],
                            activation=branche_type[1]["activation"],
                        )(branch)
                    elif branche_type[0] == "output_classes":
                        branch = tf.keras.layers.Dense(
                            units=branche_type[1],
                            activation=None,  # Use softmax during compilation
                        )(branch)
            branches.append(branch)
            if len(branches)> 1:
                x = tf.keras.layers.Concatenate()(branches)
            else:
                x = branches[0]
        model = tf.keras.Model(inputs=inputs, outputs=x)
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
    model_config=model_configs[model_v]["model"]


    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    image_shape = (*image_size, 1 if grayscale else 3)

    seed = train_params["seed"]
    lr = train_params["lr"]
    epochs = train_params["epochs"]

    #overwrite params
    
    if "params" in model_configs[model_v]:
        if "seed" in model_configs[model_v]["params"]:
            seed = model_configs[model_v]["params"]["seed"]
        elif "lr" in model_configs[model_v]["params"]:
            lr = model_configs[model_v]["params"]["lr"]
        elif "epochs" in model_configs[model_v]["params"]:
            epochs = model_configs[model_v]["params"]["epochs"]


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
