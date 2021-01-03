import json
import os

import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes
from torch.utils.data import Dataset


class ShapeCad(Dataset):
    """Dataset wrapping ground truth images and target images for ShapeNet dataset.
    """
    def __init__(self, config, dataset, training):
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
            json_file_path = os.path.join(config.META_ROOT, dataset.train_fn)
            with open(json_file_path, "r") as file:
                name_list = json.load(file)
            # mesh file list.
            mesh_name_list = [self.get_surf_path(name) for name in name_list]
            self.name_lists.update({"mesh": mesh_name_list})

            # lrf file list.
            lrf_name_list = [self.get_lrf_path(name) for name in name_list]
            self.name_lists.update({"lrf": lrf_name_list})

        else:
            json_file_path = os.path.join(config.META_ROOT, dataset.test_fn)
            with open(json_file_path, "r") as file:
                name_list = json.load(file)
            # mesh file list.
            mesh_name_list = [self.get_surf_path(name) for name in name_list]
            self.name_lists.update({"mesh": mesh_name_list})

            # lrf file list.
            lrf_name_list = [self.get_lrf_path(name) for name in name_list]
            self.name_lists.update({"lrf": lrf_name_list})

    def get_surf_path(self, name: str) -> str:
        """Get the path of the surface file.

        Args:
            name (str): The stage index brain name of the file.

        Returns:
            The path to the surface obj file.
        """
        return os.path.join(self.config.SURF_ROOT, name + ".obj")

    def get_lrf_path(self, name: str) -> str:
        """Get the path of the local reference frame file.

        Args:
            name (str): The stage index brain name of the file.

        Returns:
            The path to the lrf file.
        """
        return os.path.join(self.config.LRF_ROOT, name + ".json")

    def __len__(self):
        return len(self.name_lists["mesh"])

    def __getitem__(self, index):

        # mesh
        mesh_name = self.name_lists["mesh"][index]
        mesh = trimesh.load(mesh_name, force="mesh")

        # lrf size of N X 3 X 3.
        lrf_name = self.name_lists["lrf"][index]
        with open(lrf_name, "r") as file:
            lrf = json.load(file)
        lrf = np.array(lrf, dtype=np.float32)

        # label
        label = mesh_name.split("/")[-3]
        label = self.label_dict[label]

        return {
            "mesh": mesh,
            "lrf": lrf,
            "label": label,
        }

    @staticmethod
    def collate(batch):
        """Collate batch together for training.

        Args:
            batch

        Returns:
            The dictionary of collated batch.
        """
        # mesh
        vert_list = [
            torch.tensor(item["mesh"].vertices, dtype=torch.float32) for item in batch
        ]
        face_list = [torch.tensor(item["mesh"].faces) for item in batch]
        mesh = Meshes(verts=vert_list, faces=face_list)

        # number of vertices list.
        vert_num_list = [int(item["mesh"].vertices.shape[0]) for item in batch]

        # lrf
        lrf = torch.cat([torch.from_numpy(item["lrf"]) for item in batch], dim=0)

        # label
        label = torch.cat([torch.tensor(item["label"]).view(1) for item in batch], dim=0)

        return {
            "mesh": mesh,
            "vert_num_list": vert_num_list,
            "lrf": lrf,
            "label": label,
        }

























