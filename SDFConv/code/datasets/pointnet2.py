import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict
import trimesh
from scipy.spatial.transform import Rotation
from utils.fps import farthest_point_sample


META_ROOT = os.path.join("../data/", "meta")
left_mesh_name = "LHippo_60k.obj"
right_mesh_name = "RHippo_60k.obj"
MESH_ROOT = "/home/exx/georgey/dataset/hippocampus/obj/"


class PointNetPlusPlus(Dataset):
    """Dataset for mesh classification for PointNet++.
    """
    def __init__(self, config, dataset: EasyDict, training: str):
        """Initialize the class.

        Args:
            config (module)
            dataset (easydict.EasyDict)
            training (str): the stage of the training.
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

        # load meta list
        json_file_path = os.path.join(META_ROOT, self.dataset.meta_fn)
        with open(json_file_path, "r") as file:
            meta_list = json.load(file)

        # update the name list
        name_list = []
        for label in self.dataset.label:

            # the stage to be loaded
            if training == "train":
                identity_list = meta_list[label]["train"]
            elif training == "val":
                identity_list = meta_list[label]["val"]
            elif training == "test":
                identity_list = meta_list[label]["test"]
            else:
                raise ValueError("This stage {} is not known!".format(training))

            # concatenate stage label with identity
            stage_identity_list = [[label, identity] for identity in identity_list]
            name_list.extend(stage_identity_list)
        self.name_lists["name_list"] = name_list

    def __len__(self):
        return len(self.name_lists["name_list"])

    def __getitem__(self, index: int) -> dict:
        """The function to get an item in the dataset.

        Args:
            index: the index of the data set.

        Returns:
            The dictionary to be returned.

        """

        # find the name of the mesh.
        stage_identity = self.name_lists["name_list"][index]
        left_mesh_name_ = os.path.join(MESH_ROOT, stage_identity[0], stage_identity[1], left_mesh_name)
        right_mesh_name_ = os.path.join(MESH_ROOT, stage_identity[0], stage_identity[1], right_mesh_name)

        # load the mesh
        dict_args = {"process": False}
        left_mesh = trimesh.load(left_mesh_name_, **dict_args)
        right_mesh = trimesh.load(right_mesh_name_, **dict_args)

        # concatenate the vertices
        left_vertices = np.array(left_mesh.vertices, dtype=np.float32)
        right_vertices = np.array(right_mesh.vertices, dtype=np.float32)
        vertices = np.concatenate((left_vertices, right_vertices), axis=0)

        # furthest point sampling the vertices
        fps_vertices = farthest_point_sample(vertices, npoint=2500)

        # center, and scale the mesh
        centered_vertices = fps_vertices - np.expand_dims(np.mean(fps_vertices, axis=0), 0)  # center
        if self.dataset.scalar is None:
            dist = np.max(np.sqrt(np.sum(centered_vertices ** 2, axis=1)), 0)
            scaled_vertices = centered_vertices / dist  # scale
        else:
            scaled_vertices = centered_vertices / self.dataset.scalar  # scale

        # randomly rotate, and jitter the mesh
        if self.dataset.data_augmentation:
            augmented_vertices = np.array(scaled_vertices, dtype=np.float32)
            alpha = np.random.uniform(0, np.pi * 2)
            beta = np.random.uniform(0, np.pi * 2)
            gamma = np.random.uniform(0, np.pi * 2)
            r = Rotation.from_rotvec([alpha, beta, gamma])
            rotation_matrix = r.as_matrix()
            augmented_vertices = augmented_vertices @ rotation_matrix  # random rotation
            augmented_vertices += np.random.normal(0, 0.02, size=scaled_vertices.shape)  # random jitter
        else:
            augmented_vertices = scaled_vertices

        # label
        label = stage_identity[0]
        label = self.label_dict[label]

        return {
            "label": label, "point": augmented_vertices,
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
        point = point.float()

        # label
        label = torch.cat([torch.tensor(item["label"]).view(1) for item in batch], dim=0)

        return {
            "label": label,
            "xyz": point,
        }


def visualize_point(points: np.ndarray, scene, colors: np.ndarray = None):
    """Visualize the signed distance field.

    Args:
        points: The locations. (N, 3)
        scene: The scene to render the point cloud.
        colors: The color of the vertices

    Returns:
        scene: The scene to render the point cloud.
    """
    import pyrender

    if colors is None:
        cloud = pyrender.Mesh.from_points(points)
    else:
        cloud = pyrender.Mesh.from_points(points, colors=colors)

    scene.add(cloud)

    return scene


def test():
    """Test __getitem__ function.
    """
    print("In test, ")

    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.meta_fn = "10_fold/000.json"
    dataset.scalar = None
    dataset.data_augmentation = True

    print("For training data set: ")
    train_dt = PointNetPlusPlus(config=config, dataset=dataset, training="train")
    print("The length of the train data set is {}".format(len(train_dt)))
    for key in train_dt[3]:
        value = train_dt[3][key]
        if key != "label":
            print(key, train_dt[3][key].shape)
        else:
            print(key, value)

    print("For validation data set: ")
    val_dt = PointNetPlusPlus(config=config, dataset=dataset, training="val")
    print("The length of the validation data set is {}".format(len(val_dt)))

    print("For test data set: ")
    test_dt = PointNetPlusPlus(config=config, dataset=dataset, training="test")
    print("The length of the test data set is {}".format(len(test_dt)))


def test_1():
    """Test the collate function.
    """
    from torch.utils.data import DataLoader

    print("In test 1, ")
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.meta_fn = "10_fold/000.json"
    dataset.scalar = None
    dataset.data_augmentation = True

    print("For training data set: ")
    train_dt = PointNetPlusPlus(config=config, dataset=dataset, training="train")
    train_data_loader = DataLoader(
        train_dt,
        batch_size=4,
        num_workers=10,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_dt.collate,
    )
    for batch in train_data_loader:
        for item in batch:
            print(item, batch[item])
        break

    print("For validation data set: ")
    val_dt = PointNetPlusPlus(config=config, dataset=dataset, training="val")
    data_loader = DataLoader(
        val_dt,
        batch_size=4,
        num_workers=10,
        pin_memory=True,
        shuffle=True,
        collate_fn=val_dt.collate,
    )
    for batch in data_loader:
        for item in batch:
            print(item, batch[item])
        break

    print("For test data set: ")
    test_dt = PointNetPlusPlus(config=config, dataset=dataset, training="test")
    data_loader = DataLoader(
        test_dt,
        batch_size=4,
        num_workers=10,
        pin_memory=True,
        shuffle=True,
        collate_fn=test_dt.collate,
    )
    for batch in data_loader:
        for item in batch:
            print(item, batch[item])
        break


def test_2():
    """test with having a scalar added.
    """
    print("In test 2, ")

    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.meta_fn = "10_fold/000.json"
    dataset.scalar = 38.2
    dataset.data_augmentation = True

    print("For training data set: ")
    train_dt = PointNetPlusPlus(config=config, dataset=dataset, training="train")
    print("The length of the train data set is {}".format(len(train_dt)))
    data_sand = train_dt[3]
    for key in data_sand:
        value = data_sand[key]
        if key != "label":
            print(key, data_sand[key].shape)
        else:
            print(key, value)


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
    dataset.meta_fn = "10_fold/000.json"
    dataset.scalar = None
    dataset.data_augmentation = True

    dt = PointNetPlusPlus(config=config, dataset=dataset, training="train")
    print(len(dt))

    point = dt[3]["point"]
    scene = pyrender.Scene()
    visualize_point(point, scene)
    dict_args = {"use_raymond_lighting": True, "point_size": 2, "show_world_axis": True}
    viewer = pyrender.Viewer(scene, **dict_args)


def ref():
    # a temporary visualization code.
    import pyrender
    scene = pyrender.Scene()

    fps_colors = np.zeros(fps_vertices.shape)
    fps_colors[:, 2] = 1
    # visualize_point(fps_vertices, scene, fps_colors)

    # visualize_point(centered_vertices, scene)

    scaled_colors = np.zeros(scaled_vertices.shape)
    scaled_colors[:, 0] = 1
    visualize_point(scaled_vertices, scene, scaled_colors)

    augmented_colors = np.zeros(scaled_vertices.shape)
    augmented_colors[:, 1] = 1
    visualize_point(augmented_vertices, scene, colors=augmented_colors)

    dict_args = {"use_raymond_lighting": True, "point_size": 2, "show_world_axis": True}
    viewer = pyrender.Viewer(scene, **dict_args)


if __name__ == '__main__':
    test_2()
