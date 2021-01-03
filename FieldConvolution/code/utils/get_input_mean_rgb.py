import config
import os
import json
import random
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_name_list():
    json_file_path = os.path.join(config.META_ROOT, "train_error_image.json")
    with open(json_file_path, "r") as file:
        name_list = json.load(file)
    return name_list


def trans_error_image_to_input_image(error_image):
    """Transform the relative path of error image to input image path.

    Args:
        error_image (str)

    Returns:
        input_image_name (str)
    """
    split_name = error_image.split("/")
    index = int(split_name[-1][:2])
    split_name[-1] = "{:02d}.png".format(index)
    split_name[-5] = "data_tf"
    input_image_name = "/".join(split_name)
    return input_image_name


def get_input_image_path(relpath):
    input_image_name = trans_error_image_to_input_image(relpath)
    return os.path.join(config.DATASET_ROOT, input_image_name)


def get_mean(name_list):
    # Transform to the predicted image path.
    pred_name_list = [get_input_image_path(name) for name in name_list]

    mean = np.zeros(3)
    for i in tqdm(range(10), position=1):
        random.shuffle(pred_name_list)
        for j in tqdm(range(10), position=0):
            image = Image.open(pred_name_list[j])
            image = np.array(image)
            # Remove the back ground.
            image = image[np.where(image[:, :, 3] != 0)]
            image = image[:, :3]
            mean += np.mean(image, axis=0)
    mean /= 100
    print("The mean pixel value of the predicted mesh image is: {}".format(mean/255.0))


def get_std(name_list):
    # Transform to the predicted image path.
    pred_name_list = [get_input_image_path(name) for name in name_list]

    std = np.zeros(3)
    for i in tqdm(range(10)):
        random.shuffle(pred_name_list)
        for j in tqdm(range(10)):
            image = Image.open(pred_name_list[j])
            image = np.array(image)
            image = image[np.where(image[:, :, 3] != 0)]
            image = image[:, :3]
            std += np.std(image, axis=0)
    std /= 1000
    print("The std pixel value of the predicted mesh image is: {}".format(std/255.0))


if __name__ == '__main__':

    get_mean(load_name_list())
    get_std(load_name_list())



