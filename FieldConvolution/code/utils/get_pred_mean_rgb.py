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


def trans_error_image_to_pred_image(error_image):
    split_name = error_image.split("/")
    split_name[-5] = "data_pred_img"
    pred_image_name = "/".join(split_name)
    return pred_image_name


def get_pred_image_path(relpath):
    pred_image_name = trans_error_image_to_pred_image(relpath)
    return os.path.join(config.DATASET_ROOT, pred_image_name)


def get_mean(name_list):
    # Transform to the predicted image path.
    pred_name_list = [get_pred_image_path(name) for name in name_list]

    mean = np.zeros(3)
    for i in tqdm(range(100)):
        random.shuffle(pred_name_list)
        # print(random_shuffle_list)
        for j in tqdm(range(10)):
            image = Image.open(pred_name_list[j])
            image = np.array(image)
            image = image[:, :, :3]
            mean += np.mean(image, axis=(0, 1))
    mean /= 1000
    print("The mean pixel value of the predicted mesh image is: {}".format(mean))


def get_std(name_list):
    # Transform to the predicted image path.
    pred_name_list = [get_pred_image_path(name) for name in name_list]

    std = np.zeros(3)
    for i in tqdm(range(100)):
        random.shuffle(pred_name_list)
        for j in tqdm(range(10)):
            image = Image.open(pred_name_list[j])
            image = np.array(image)
            image = image[:, :, :3]
            std += np.std(image, axis=(0, 1))
    std /= 1000
    print("The std pixel value of the predicted mesh image is: {}".format(std))


if __name__ == '__main__':

    get_std(load_name_list())


