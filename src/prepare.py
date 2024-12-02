import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

from utils.seed import set_seed


def get_preview_plot(ds: tf.data.Dataset, labels: List[str],grayscale=False) -> plt.Figure:
    """Plot a preview of the prepared dataset"""
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    for images, label_idxs in ds.take(1):
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            image = (images[i].numpy() * 255).astype("uint8")
            if grayscale:
                plt.imshow(image,cmap="gray")
            else:
                plt.imshow(image)
            
            plt.title(labels[int(label_idxs[i].numpy()[0])])
            plt.axis("off")

    return fig


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 prepare.py <raw-dataset-folder> <prepared-dataset-folder>\n")
        exit(1)

    # Load parameters
    yaml_file = yaml.safe_load(open("params.yaml"))
    prepare_params = yaml_file["prepare"]
    train_params = yaml_file["train"]
    model_configs = yaml_file["model"]

    prepared_dataset_folder = Path(sys.argv[1])
    seed = prepare_params["seed"]
    split = prepare_params["split"]
    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    batch = train_params["batch"]
    
    model_v = (sys.argv[2])
    #overwrite
    if "params" in model_configs[model_v]:
        print(model_configs[model_v]["params"])
        if "batch" in model_configs[model_v]["params"]:
            batch = model_configs[model_v]["params"]["batch"]


    # Set seed for reproducibility
    set_seed(seed)

    # Read data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    class_names = [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle",
        "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
        "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon",
        "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
        "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
        "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree",
        "wolf", "woman", "worm"
    ]

    # Convert images to grayscale if specified
    if grayscale:
        x_train = tf.image.rgb_to_grayscale(x_train).numpy()
        x_test = tf.image.rgb_to_grayscale(x_test).numpy()

    # Resize
    x_train = tf.image.resize(x_train, image_size).numpy()
    x_test = tf.image.resize(x_test, image_size).numpy()

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Split train data into training and validation sets based on split parameter
    val_size = int(len(x_train) * split)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    # Create TensorFlow datasets
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    if not prepared_dataset_folder.exists():
        prepared_dataset_folder.mkdir(parents=True)

    # Save the preview plot
    preview_plot = get_preview_plot(ds_train, class_names,grayscale)
    preview_plot.savefig(prepared_dataset_folder / "preview.png")

        # Save the dataset labels and the datasets
    with open(prepared_dataset_folder / "labels.json", "w") as f:
        json.dump(class_names, f)
    ds_train.save(str(prepared_dataset_folder / "train"))
    ds_val.save(str(prepared_dataset_folder / "val"))
    ds_test.save(str(prepared_dataset_folder / "test"))
    print(ds_train)
    print(f"\nDataset saved at {prepared_dataset_folder.absolute()}")


if __name__ == "__main__":
    main()
