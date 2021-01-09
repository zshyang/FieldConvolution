"""This dataloader is based on pointnet2.py. And instead of using random sampling on the surface, this one will use
furthers point sampling on the surface.
----Zhangsihao Yang, Jan 8, 2021
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict


META_ROOT = os.path.join("../data/", "meta")
SDF_ROOT = os.path.join("../data/", "sdf")


def farthest_point_sample(point, npoint):
    """Farthest point sampling for numpy implementation.

    Input:
        xyz: point cloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled point cloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def get_sdf_path(name: [str]) -> str:
    """Get the path to the signed distance field file.

    Args:
        name: The list of string information.
            name[0]: The stage name.
            name[1]: The index name.

    Returns:
        The path to the sdf file.
    """
    assert len(name) == 2, "The length of the input name is not correct!"
    return os.path.join(SDF_ROOT, name[0], "{}.npz".format(name[1]))


def clean_name_list(name_list: list) -> list:
    """Clean redundant name from the name list.

    Args:
        name_list: The name list.

    Returns:
        return_name_list: The cleaned name list.
    """
    return_name_list = []
    for name in name_list:
        return_name = [name[0], name[1]]
        if return_name not in return_name_list:
            return_name_list.append(return_name)
    return return_name_list


class PointNetPlusPlus(Dataset):
    """Dataset for mesh classification for PointNet++.
    """
    def __init__(self, config, dataset: EasyDict, training: bool):
        """Initialize the class.

        Args:
            config (module)
            dataset (easydict.EasyDict)
            training (bool)
        """
        self.config = config
        self.dataset = dataset
        self.training = training

        self.name_lists = {}

        # Create the label dictionary.
        label_dict = {}
        for i, label in enumerate(dataset.label):
            label_dict[label] = i
        self.label_dict = label_dict

        # To load training name list or validation name list.
        if training:
            json_file_path = os.path.join(META_ROOT, dataset.train_fn)
            with open(json_file_path, "r") as file:
                name_list = json.load(file)

            # sdf file list.
            sdf_name_list = [get_sdf_path(name) for name in name_list]
            self.name_lists.update({"sdf": sdf_name_list})

        else:
            json_file_path = os.path.join(META_ROOT, dataset.test_fn)
            with open(json_file_path, "r") as file:
                name_list = json.load(file)

            # sdf file list.
            sdf_name_list = [get_sdf_path(name) for name in name_list]
            self.name_lists.update({"sdf": sdf_name_list})

    def __len__(self):
        return len(self.name_lists["sdf"])

    def __getitem__(self, index: int) -> dict:
        """The function to get an item in the dataset.

        Args:
            index:

        Returns:
            The dictionary to be returned.

        """
        # Load the signed distance field.
        sdf_file_name = self.name_lists["sdf"][index]
        sdf = np.load(sdf_file_name)
        sdf_pos = sdf["pos"]
        sdf_neg = sdf["neg"]
        point_sdf = np.concatenate((sdf_pos, sdf_neg), axis=0)

        point = point_sdf[:, :3]
        sdf = point_sdf[:, 3:]

        # Pick the points that close enough to the surface.
        surface_point = point[(np.abs(sdf) < 1e-3)[:, 0]]

        # Use FPS to pick 2500 points from the surface.
        picked_point = farthest_point_sample(surface_point, 2500)

        # label
        label = sdf_file_name.split("/")[-2]
        label = self.label_dict[label]

        return {
            "label": label, "point": picked_point,
        }

    @staticmethod
    def collate(batch: [dict]) -> dict:
        """Collate batch together for training.

        Args:
            batch: A list of dict. In each dictionary, There are
                "label": The label of this item.
                "point": The point of this item.

        Returns:
            The dictionary of collated batch.
                "point": Tensor with shape (B, N, 3).
                "label": tensor with shape (B) contains label.
        """
        # Point.
        point = torch.stack([torch.from_numpy(item["point"]) for item in batch])
        point = point.permute(0, 2, 1)

        # label
        label = torch.cat([torch.tensor(item["label"]).view(1) for item in batch], dim=0)

        return {
            "label": label,
            "xyz": point,
        }


def visualize_point(points: np.ndarray, scene):
    """Visualize the signed distance field.
    Args:
        points: The locations. (N, 3)
        scene: The scene to render the point cloud.
    Returns:
        scene: The scene to render the point cloud.
    """
    import pyrender

    cloud = pyrender.Mesh.from_points(points)

    scene.add(cloud)

    return scene


def test():
    """Test __getitem__ function.
    """
    print("In test, ")

    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"

    dt = PointNetPlusPlus(config=config, dataset=dataset, training=True)
    print(len(dt))

    for key in dt[3]:
        value = dt[3][key]
        if key != "label":
            print(key, dt[3][key].shape)
        else:
            print(key, value)


def test_1():
    """Test the collate function.
    """
    print("In test 1, ")
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"
    options = EasyDict()

    dt = PointNetPlusPlus(config=config, dataset=dataset, training=False)
    from torch.utils.data import DataLoader
    train_data_loader = DataLoader(
        dt,
        batch_size=4,
        num_workers=10,
        pin_memory=True,
        shuffle=True,
        collate_fn=dt.collate,
    )
    for batch in train_data_loader:
        for item in batch:
            print(item, batch[item])
        break


def test_6():
    """Go over the data loader.
    """
    print("In test 6, ")
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"
    options = EasyDict()

    dt = PointNetPlusPlus(config=config, dataset=dataset, training=True)
    from torch.utils.data import DataLoader
    import random
    random.seed(4124036635)
    np.random.seed(4124036635)
    torch.manual_seed(4124036635)
    train_data_loader = DataLoader(
        dt,
        batch_size=128,
        num_workers=10,
        pin_memory=True,
        shuffle=True,
        collate_fn=dt.collate,
    )

    for i, batch in enumerate(train_data_loader):
        print(i)
        for item in batch:
            print(item, batch[item].shape, batch[item].max())


def test_9():
    """Test the sampling is good or not.
    """
    import pyrender
    print("In test 9, ")

    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"

    dt = PointNetPlusPlus(config=config, dataset=dataset, training=True)
    print(len(dt))

    point = dt[3]["point"]
    scene = pyrender.Scene()
    visualize_point(point, scene)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


if __name__ == '__main__':
    test_9()
