import json
import os

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from easydict import EasyDict


META_ROOT = os.path.join("../data/", "meta")
SDF_ROOT = os.path.join("../data/", "mesh")


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
    return os.path.join(SDF_ROOT, name[0], "{}.npy".format(name[1]))


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
            name_list = self.expand_list(name_list)
            print(len(name_list))
            name_list = clean_name_list(name_list)
            print(len(name_list))
            # print()

            # sdf file list.
            sdf_name_list = []

            # mesh file list.
            mesh_name_list = [self.get_surf_path(name) for name in name_list]
            self.name_lists.update({"mesh": mesh_name_list})

            # lrf file list.
            lrf_name_list = [self.get_lrf_path(name) for name in name_list]
            self.name_lists.update({"lrf": lrf_name_list})

            # Geodesic distance file list.
            dist_name_list = [self.get_dist_path(name) for name in name_list]
            self.name_lists.update({"dist": dist_name_list})

            # fps index on mesh list.
            fps_index_list = [self.get_fps_index_path(name) for name in name_list]
            self.name_lists.update({"fps_index": fps_index_list})

        else:
            json_file_path = os.path.join(META_ROOT, dataset.test_fn)
            with open(json_file_path, "r") as file:
                name_list = json.load(file)
            name_list = self.expand_list(name_list)

            # mesh file list.
            mesh_name_list = [self.get_surf_path(name) for name in name_list]
            self.name_lists.update({"mesh": mesh_name_list})

            # lrf file list.
            lrf_name_list = [self.get_lrf_path(name) for name in name_list]
            self.name_lists.update({"lrf": lrf_name_list})

            # Geodesic distance file list.
            dist_name_list = [self.get_dist_path(name) for name in name_list]
            self.name_lists.update({"dist": dist_name_list})

            # fps index on mesh list.
            fps_index_list = [self.get_fps_index_path(name) for name in name_list]
            self.name_lists.update({"fps_index": fps_index_list})


    def get_fps_index_path(self, name: list) -> str:
        return os.path.join(DIST_ROOT, name[0], name[1], "fps", "{:05d}.npy".format(name[2]))

    def expand_list(self, name_list: list):
        expanded_name_list = []
        for name in name_list:
            for i in range(30):
                name_id = name + [i]
                expanded_name_list.append(name_id)
        return expanded_name_list

    def get_dist_path(self, name: list) -> str:
        """Get the path of the geodesic distance file.

        Args:
            name: The stage index of the brain file.

        Returns:
            The path to the geodesic distance file.
        """
        return os.path.join(DIST_ROOT, name[0], name[1], "partial", "{:05d}.npy".format(name[2]))

    def get_surf_path(self, name: list) -> str:
        """Get the path of the surface file.

        Args:
            name : The stage index of the brain file.

        Returns:
            The path to the surface obj file.
        """

        return os.path.join(SURF_ROOT, name[0], name[1] + ".obj")

    def get_lrf_path(self, name: list) -> str:
        """Get the path of the local reference frame file.

        Args:
            name (list): The stage index brain name of the file.

        Returns:
            The path to the lrf file.
        """
        return os.path.join(LRF_ROOT, name[0], name[1], "{:05d}.npy".format(name[2]))

    def __len__(self):
        return len(self.name_lists["mesh"])

    def __getitem__(self, index):
        # mesh
        mesh_name = self.name_lists["mesh"][index]
        dict_args = {"process": False}
        mesh = trimesh.load(mesh_name, **dict_args)
        # print(mesh_name)

        # lrf size of N X 3 X 3.
        lrf_name = self.name_lists["lrf"][index]
        lrf = np.load(lrf_name)
        lrf_mask = np.logical_not(np.logical_and(lrf > 0.0, lrf < 1.0))
        lrf[lrf_mask] = 1.0

        # Geodesic distance. (N, N)
        dist_name = self.name_lists["dist"][index]
        # print(dist_name)
        dist = np.load(dist_name)
        dist = process_inf_dist(dist)

        # label
        label = mesh_name.split("/")[-2]
        label = self.label_dict[label]

        # Load fps index.
        fps_index_fn = self.name_lists["fps_index"][index]
        fps_index = np.load(fps_index_fn)

        # normal. (N, 3)
        normal = np.array(mesh.vertex_normals, dtype=np.float32)
        normal = normal[fps_index]
        normal_mask = np.logical_and(normal > 0.0, normal < 1.0)
        normal_mask = np.logical_not(normal_mask)
        normal[normal_mask] = 1.0

        # Vertices. (N, 3)
        verts = np.array(mesh.vertices, dtype=np.float32)
        verts = verts[fps_index]

        return {
            "dist": dist,
            "verts": verts,
            "lrf": lrf,
            "label": label,
            "normal": normal,
        }

    @staticmethod
    def collate(batch: list) -> dict:
        """Collate batch together for training.

        Args:
            batch: A list of dict.

        Returns:
            The dictionary of collated batch.
                "padded_verts": tensor of vertices of shape (N, max(V_n), 3).
                "padded_faces": tensor of faces of shape (N, max(F_n), 3).
                "vert_num": tensor with shape (N) contains V_n.
                "face_num": tensor with shape (N) contains F_n.
                "lrf": tensor of local reference frame of shape (N, max(V_n), 3, 3).
                "label": tensor with shape (N) contains label,
        """

        # # mesh
        # vert_list = [
        #     torch.tensor(item["mesh"].vertices, dtype=torch.float32) for item in batch
        # ]
        # face_list = [torch.tensor(item["mesh"].faces) for item in batch]
        # mesh = Meshes(verts=vert_list, faces=face_list)
        #
        # # Padded vertices and faces.
        # padded_verts = mesh.verts_padded()  # padded with 0
        # padded_faces = mesh.faces_padded()  # padded with -1
        #
        # # number of vertices list.
        # vert_num = torch.stack(
        #     [torch.tensor(int(item["mesh"].vertices.shape[0])) for item in batch]
        # )
        # face_num = torch.stack(
        #     [torch.tensor(int(item["mesh"].faces.shape[0])) for item in batch]
        # )
        # # Get the padded vertices.
        # valid_list = [np.ones((int(item["mesh"].vertices.shape[0]), 1)) for item in batch]
        # valid = list_to_padded([torch.from_numpy(item) for item in valid_list], pad_value=0.0)

        # lrf
        lrf = torch.stack([torch.from_numpy(item["lrf"]).view(-1, 9) for item in batch])

        # dist
        dist = torch.stack([torch.from_numpy(item["dist"]) for item in batch])
        # dist = list_to_padded([torch.from_numpy(item["dist"]) for item in batch], pad_value=0.0)

        # label
        label = torch.cat([torch.tensor(item["label"]).view(1) for item in batch], dim=0)

        # normal
        normal = torch.stack([torch.from_numpy(item["normal"]) for item in batch])

        # verts
        verts = torch.stack([torch.from_numpy(item["verts"]) for item in batch])

        return {
            "dist_map": dist,
            "padded_verts": verts,
            "lrf": lrf,
            "label": label,
            "normal": normal,
        }


def test():
    """Test the get item function.
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
        if key != "mesh" and key != "label":
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

    dt = ShapeCad(config=config, dataset=dataset, training=False)
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
            print(item, batch[item].shape)
        break


def test_2():
    """Test the distance map is correct.
    """
    print("In test 2, ")
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"
    options = EasyDict()

    dt = ShapeCad(config=config, dataset=dataset, training=True)

    for i in range(len(dt)):
        for key in dt[i]:
            if key is "dist":
                # print(np.sum(dt[i][key]))
                # if np.isnan(np.sum(dt[i][key])):
                #     print(i)
                print(i, np.sum(dt[i][key]))


def test_3():
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"
    options = EasyDict()

    dt = ShapeCad(config=config, dataset=dataset, training=True)

    for i in range(len(dt)):
        print(dt.name_lists["mesh"][i])
        for key in dt[i]:
            if key is "dist":
                # print(np.sum(dt[i][key]))
                if np.isnan(np.sum(dt[i][key])):
                    print(i)


def test_4():
    """Fix the issue with inf in distance map.
    """
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"
    options = EasyDict()

    dt = ShapeCad(config=config, dataset=dataset, training=True)

    dist = dt[3913]["dist"]
    print("The initial distance map is: \n", dist)
    dist_mask = dist > 10**6
    dist[dist_mask] = 0.0
    dist_true_max = dist.max()
    print("The max distance except inf is: \t\t", dist_true_max)
    dist[dist_mask] = dist_true_max
    print("The distance map after the precessing: \t", dist.max())


def process_inf_dist(dist: np.ndarray):
    """Replace inf in distance map.

    Args:
        dist: The initial distance map.

    Returns:
        dist: The processed distance map.
    """
    num_invalid = np.logical_not(np.logical_and(dist > 0, dist < 100))
    dist_mask = np.logical_or(num_invalid, (np.isnan(dist)))
    dist[dist_mask] = 0.0
    dist_true_max = dist.max()
    dist[dist_mask] = dist_true_max
    return dist


def test_5():
    """Test the function process_inf_dist.
    """
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"
    options = EasyDict()

    dt = ShapeCad(config=config, dataset=dataset, training=True)

    dist = dt[3913]["dist"]
    print("The initial distance map is: \n", dist)
    print("The maximum distance map before the precessing: \t", dist.max())
    dist = process_inf_dist(dist)
    print("The maximum distance map after the precessing: \t\t", dist.max())


def test_6():
    """Test what is wrong with the data loader.
    """
    print("In test 6, ")
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"
    options = EasyDict()

    dt = ShapeCad(config=config, dataset=dataset, training=True)
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
    # print(iter(train_data_loader)[58])
    # print(train_data_loader[138])
    for i, batch in enumerate(train_data_loader):
        print(i)
        for item in batch:
            print(item, batch[item].shape, batch[item].max())


def test_7():
    """Test what is wrong with the data loader.
    """
    print("In test 7, ")
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"
    options = EasyDict()

    dt = ShapeCad(config=config, dataset=dataset, training=True)
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
    for i, batch in enumerate(dt):
        print(i)
        for item in batch:
            if item != "label":
                print(item, batch[item].shape)


def test_8():
    """Test what is wrong with the data loader.
    """
    print("In test 8, ")
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"
    options = EasyDict()

    dt = ShapeCad(config=config, dataset=dataset, training=True)

    for i in range(0, len(dt)):
        batch = dt[i]
        # print(i)
        for item in batch:
            if item == "dist":
                print(i, item, batch[item].shape, batch[item].max())


if __name__ == '__main__':
    test()
