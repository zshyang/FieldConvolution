import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict


ROOT = os.path.join("../data/", "yh")


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


class HipoDataLoader(Dataset):
    def __init__(self, root, npoints=2048, train=True):
        self.npoints = npoints
        self.root = root
        if train:
            pos = os.path.join(self.root, 'train/pos')
            neg = os.path.join(self.root, 'train/neg')
            self.pos_shapes_dir = [os.path.join(pos, f) for f in listdir(pos) if isfile(join(pos, f))]
            self.neg_shapes_dir = [os.path.join(neg, f) for f in listdir(neg) if isfile(join(neg, f))]
        else:
            pos = os.path.join(self.root, 'test/pos')
            neg = os.path.join(self.root, 'test/neg')
            self.pos_shapes_dir = [os.path.join(pos, f) for f in listdir(pos) if isfile(join(pos, f))]
            self.neg_shapes_dir = [os.path.join(neg, f) for f in listdir(neg) if isfile(join(neg, f))]
        self.datapath = np.concatenate((self.pos_shapes_dir, self.neg_shapes_dir), axis=0)
        label1 = np.zeros(len(self.pos_shapes_dir))
        label2 = np.ones(len(self.neg_shapes_dir))
        self.label = np.concatenate((label1, label2), axis=0)
    def __len__(self):
        return len(self.datapath)
    def _get_item(self, index):
        fn = self.datapath[index]
        cls = self.label[index]
        point_set = np.loadtxt(fn).astype(np.float32)
        point_set = point_set[:, 0:3]
        point_set = farthest_point_sample(point_set, self.npoints)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return point_set, cls
    def __getitem__(self, index):
        return self._get_item(index)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
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


class PointNetPlusPlus(Dataset):
    """Dataset for mesh classification for PointNet++ for data processed by Yonghui Fan.
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
        self.npoints = 2048
        if training:
            pos = os.path.join(ROOT, 'train/pos')
            neg = os.path.join(ROOT, 'train/neg')

            self.pos_shapes_dir = [
                os.path.join(pos, f) for f in os.listdir(pos) if os.path.isfile(os.path.join(pos, f))
            ]
            self.neg_shapes_dir = [
                os.path.join(neg, f) for f in os.listdir(neg) if os.path.isfile(os.path.join(neg, f))
            ]
        else:
            pos = os.path.join(ROOT, 'test/pos')
            neg = os.path.join(ROOT, 'test/neg')

            self.pos_shapes_dir = [
                os.path.join(pos, f) for f in os.listdir(pos) if os.path.isfile(os.path.join(pos, f))
            ]
            self.neg_shapes_dir = [
                os.path.join(neg, f) for f in os.listdir(neg) if os.path.isfile(os.path.join(neg, f))
            ]

        self.datapath = np.concatenate((self.pos_shapes_dir, self.neg_shapes_dir), axis=0)
        label1 = np.zeros(len(self.pos_shapes_dir))
        label2 = np.ones(len(self.neg_shapes_dir))
        self.label = np.concatenate((label1, label2), axis=0)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        fn = self.datapath[index]
        cls = self.label[index]
        point_set = np.loadtxt(fn).astype(np.float32)
        point_set = point_set[:, 0:3]
        point_set = farthest_point_sample(point_set, self.npoints)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return {
            "point": point_set, "label": cls,
        }

    def __getitem__(self, index):
        return self._get_item(index)

    @staticmethod
    def collate(batch: [dict]) -> dict:
        """Collate batch together

        Args:
            batch:

        Returns:

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

    print(dt[3])


def test_1():
    """Test the collate function.
    """
    print("In test 1, ")
    config = None
    dataset = EasyDict()
    dataset.label = ["AD_pos", "NL_neg"]
    dataset.test_fn = "AD_pos_NL_neg_test.json"
    dataset.train_fn = "AD_pos_NL_neg_train.json"

    dt = PointNetPlusPlus(config=config, dataset=dataset, training=False)
    from torch.utils.data import DataLoader
    train_data_loader = DataLoader(
        dt,
        batch_size=4,
        num_workers=10,
        pin_memory=True,
        shuffle=True,
        collate_fn=dt.collate
    )
    for batch in train_data_loader:
        print(batch)
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
