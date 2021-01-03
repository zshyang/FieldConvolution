from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import os
import json
from PIL import Image
from skimage import io, transform
from torchvision.transforms import Normalize
from torch.utils.data.dataloader import default_collate


class ShapeNetUNet(Dataset):
    """Dataset wrapping ground truth images and target images for ShapeNet dataset.
    """
    def __init__(self, config, dataset, training):
        """
            config (module)
            dataset (easydict.EasyDict)
            training (bool)
        """
        self.config = config
        self.dataset = dataset
        self.training = training

        self.name_lists = {}

        # To load training name list or validation name list.
        if training:
            json_file_path = os.path.join(config.META_ROOT, dataset.train_fn)
            filename = os.path.splitext(dataset.train_fn)[0]
            if filename.startswith("train_"):
                filename = filename[len("train_"):]
            with open(json_file_path, "r") as file:
                name_list = json.load(file)
            # Transform the path for this project.
            error_name_list = [self.get_error_image_path(name) for name in name_list]
            self.name_lists.update({filename: error_name_list})  # error_image
            # Transform to the input image path.
            input_name_list = [self.get_input_image_path(name) for name in name_list]
            self.name_lists.update({"input_image": input_name_list})  # input_image
            # Transform to the predicted image path.
            pred_name_list = [self.get_pred_image_path(name) for name in name_list]
            self.name_lists.update({"pred_image": pred_name_list})  # pred_image
        else:
            json_file_path = os.path.join(config.META_ROOT, dataset.test_fn)
            filename = os.path.splitext(dataset.test_fn)[0]
            if filename.startswith("test_"):
                filename = filename[len("test_"):]
            with open(json_file_path, "r") as file:
                name_list = json.load(file)
            # Transform the path for this project.
            error_name_list = [self.get_error_image_path(name) for name in name_list]
            self.name_lists.update({filename: error_name_list})  # error_image
            # Transform to the input image path.
            input_name_list = [self.get_input_image_path(name) for name in name_list]
            self.name_lists.update({"input_image": input_name_list})  # input_image
            # Transform to the predicted image path.
            pred_name_list = [self.get_pred_image_path(name) for name in name_list]
            self.name_lists.update({"pred_image": pred_name_list})  # pred_image

    def get_error_image_path(self, relpath):
        return os.path.join(self.config.DATASET_ROOT, relpath)

    @staticmethod
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

    @staticmethod
    def trans_error_image_to_pred_image(error_image):
        """Transform the relative path of error image to predicted image path.

        Args:
            error_image (str)

        Returns:
            pred_image_name (str)
        """
        split_name = error_image.split("/")
        split_name[-5] = "data_pred_img"
        pred_image_name = "/".join(split_name)
        return pred_image_name

    def get_input_image_path(self, relpath):
        input_image_name = self.trans_error_image_to_input_image(relpath)
        return os.path.join(self.config.DATASET_ROOT, input_image_name)

    def get_pred_image_path(self, relpath):
        pred_image_name = self.trans_error_image_to_pred_image(relpath)
        return os.path.join(self.config.DATASET_ROOT, pred_image_name)

    def __len__(self):
        return 32

    def preprocess_error_image(self, image_name: str) -> torch.Tensor:
        """Preprocess the error image.

        :param image_name
        :return: img
        """
        img = io.imread(image_name)  # numpy.ndarray, 137, 137, 4
        # Convert image to float data type and scale it between 0 and 1.
        img = np.array(img, dtype=np.float32) / 255.0
        # Convert RGB to error scale.
        img = img[:, :, 0] * (1.0 - img[:, :, 1])
        img = np.expand_dims(img, axis=-1)
        # Resize the image.
        img = transform.resize(
            img, (self.dataset.resize, self.dataset.resize),
            mode='constant', anti_aliasing=False
        )  # to match behavior of old versions
        # Convert to a tensor.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        return img

    def preprocess_render_image(self, image_name: str) -> torch.Tensor:
        """Preprocess ShapeNet input RGB image.
        Refer from the link below.
        https://github.com/nywang16/Pixel2Mesh/blob/7c5a7a142d5f64f15250f9e98ee496980f69e005/p2m/fetcher.py#L66

        :param image_name
        :return: img
        """
        img = io.imread(image_name)
        # Remove the alpha channel.
        img[np.where(img[:, :, 3] == 0)] = 255
        img = transform.resize(img, (self.dataset.resize, self.dataset.resize))  # to match behavior of old versions
        img = img[:, :, :3].astype(np.float32)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        return img

    def __getitem__(self, index):

        error_image_name = self.name_lists["error_image"][index]
        input_image_name = self.name_lists["input_image"][index]
        pred_image_name = self.name_lists["pred_image"][index]

        split_name = error_image_name.split("/")
        if split_name[-5] == "data_gt_error":
            loss_name = "gt"
        elif split_name[-5] == "data_pred_error":
            loss_name = "pred"
        else:
            raise ValueError("No such folder.")

        error_image = self.preprocess_error_image(error_image_name)
        input_image = self.preprocess_render_image(input_image_name)
        pred_image = self.preprocess_render_image(pred_image_name)

        return {
            "error_image": error_image,
            "input_image": input_image,
            "pred_image": pred_image,
            "loss_name": loss_name,
        }

    def collate(self, batch):
        """Collate batch together for training.

        :param batch
        :return:
        """
        batch = default_collate(batch)
        return batch
