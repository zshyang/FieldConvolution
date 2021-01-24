import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict
import trimesh
from utils.fps import farthest_point_sample


META_ROOT = os.path.join("../data/", "meta")
left_mesh_name = "LHippo_60k.obj"
right_mesh_name = "RHippo_60k.obj"
MESH_ROOT = "/home/exx/georgey/dataset/hippocampus/obj/"


# def get_sdf_path(name: [str]) -> str:
#     """Get the path to the signed distance field file.
#
#     Args:
#         name: The list of string information.
#             name[0]: The stage name.
#             name[1]: The index name.
#
#     Returns:
#         The path to the sdf file.
#     """
#     assert len(name) == 2, "The length of the input name is not correct!"
#     return os.path.join(SDF_ROOT, name[0], "{}.npz".format(name[1]))


# def clean_name_list(name_list: list) -> list:
#     """Clean redundant name from the name list.
#
#     Args:
#         name_list: The name list.
#
#     Returns:
#         return_name_list: The cleaned name list.
#     """
#     return_name_list = []
#     for name in name_list:
#         return_name = [name[0], name[1]]
#         if return_name not in return_name_list:
#             return_name_list.append(return_name)
#     return return_name_list


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

        # load the scalar information

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

        # center scale, and randomly rotate the mesh
        center_vertices = fps_vertices - np.expand_dims(np.mean(fps_vertices, axis=0), 0)  # center

        # a temporary visualization code.
        import pyrender
        scene = pyrender.Scene()
        fps_colors = np.zeros(fps_vertices.shape)
        fps_colors[:, 2] = 1
        visualize_point(fps_vertices, scene, fps_colors)
        visualize_point(center_vertices, scene)
        dict_args = {"use_raymond_lighting": True, "point_size": 2}
        viewer = pyrender.Viewer(scene, **dict_args)

        print(center_vertices.shape)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        # Load the signed distance field.
        sdf_file_name = self.name_lists["sdf"][index]
        sdf = np.load(sdf_file_name)
        sdf_pos = sdf["pos"]
        sdf_neg = sdf["neg"]
        point_sdf = np.concatenate((sdf_pos, sdf_neg), axis=0)

        point = point_sdf[:, :3]
        sdf = point_sdf[:, 3:]

        # Pick the first 2500 points that are closest to the surface.
        # Sort the signed distance according to the absolute value.
        sort_index = np.argsort(np.abs(sdf), axis=0)
        picked_point = point[sort_index[:2500, 0]]  # 2500 is hard coded here.

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
    dataset.scale = None

    print("For training data set: ")
    train_dt = PointNetPlusPlus(config=config, dataset=dataset, training="train")
    print(len(train_dt))
    for key in train_dt[3]:
        value = train_dt[3][key]
        if key != "label":
            print(key, train_dt[3][key].shape)
        else:
            print(key, value)

    print("For validation data set: ")
    val_dt = PointNetPlusPlus(config=config, dataset=dataset, training="val")

    print("For test data set: ")
    test_dt = PointNetPlusPlus(config=config, dataset=dataset, training="test")
    print(len(train_dt))




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
    test()
