import os
import json
import trimesh
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.neighbors import (
    NeighborhoodComponentsAnalysis, KNeighborsClassifier
)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


META_ROOT = "../data/meta"
fold_folder = "10_fold"
train_fn = "AD_pos_NL_neg_train.json"
MESH_ROOT = "/home/exx/georgey/dataset/hippocampus/obj/"


def get_mesh_path(name, side="both"):
    mesh_file_name = os.path.join(MESH_ROOT, name[0], name[1] + ".obj")
    return mesh_file_name


def get_left_names(name_list):
    left_names = []
    for name in name_list:
        if name[1].endswith("LHippo_60k"):
            left_names.append(name)
    return left_names


def get_right_names(name_list):
    right_names = []
    for name in name_list:
        if name[1].endswith("RHippo_60k"):
            right_names.append(name)
    return right_names


def merge_left_right_names(left_names, right_names):
    left_right_names = []
    for left_name in left_names:
        identity = left_name[0] + left_name[1][:-11]
        for right_name in right_names:
            right_identity = right_name[0] + right_name[1][:-11]
            if identity == right_identity:
                # print(left_name, right_name)
                left_right_names.append([left_name, right_name])
    # print(len(left_right_names))


def load_train_json():
    json_file_path = os.path.join(META_ROOT, train_fn)
    with open(json_file_path, "r") as file:
        name_list = json.load(file)

    # Reorganize the name list given the left right and both.
    left_names = get_left_names(name_list)  # 300
    right_names = get_right_names(name_list)  # 311
    lr_names = merge_left_right_names(left_names, right_names)

    mesh_name_list = [get_mesh_path(name) for name in name_list]

    dict_args = {"process": False}
    mesh = trimesh.load(mesh_name_list[0], **dict_args)

    print(mesh.volume)


def compute_volume():
    with open(os.path.join(META_ROOT, fold_folder, "000.json"), "r") as file:
        split_list = json.load(file)
    for stage in split_list:
        for tvt in split_list[stage]:
            for identity in tqdm(split_list[stage][tvt]):
                left_mesh_name = os.path.join(
                    MESH_ROOT, stage, identity, "LHippo_60k.obj"
                )
                right_mesh_name = os.path.join(
                    MESH_ROOT, stage, identity, "RHippo_60k.obj"
                )
                dict_args = {"process": False}
                left_mesh = trimesh.load(left_mesh_name, **dict_args)
                right_mesh = trimesh.load(right_mesh_name, **dict_args)

                volume_path = os.path.join(
                    "../data", "volume", stage, identity, "volume.npy"
                )
                os.makedirs(os.path.dirname(volume_path), exist_ok=True)
                volumes = np.array([left_mesh.volume, right_mesh.volume], dtype=np.float)

                np.save(volume_path, volumes)


def plot_volume():
    with open(os.path.join(META_ROOT, fold_folder, "000.json"), "r") as file:
        split_list = json.load(file)
    volume_dict = dict()
    stage_list = list()
    for stage in split_list:
        stage_list.append(stage)
        volume_dict[stage] = dict()
        for tvt in split_list[stage]:
            volume_list = []
            for identity in split_list[stage][tvt]:

                volume_path = os.path.join(
                    "../data", "volume", stage, identity, "volume.npy"
                )
                volumes = np.load(volume_path)

                volume_list.append(volumes)
            volume_array = np.stack(volume_list, axis=0)
            volume_dict[stage][tvt] = volume_array

    f, ax = plt.subplots(figsize=(7, 7))
    desired_stage = ["AD_pos", "NL_neg"]
    for stage in stage_list:
        if stage in desired_stage:
            volume = volume_dict[stage]["train"]
            ax.scatter(volume[:, 0], volume[:, 1])
    plt.legend(desired_stage)
    plt.show()


def volume_knn_classification(fold: int, desired_stage: [str]):
    """Use 1.6.7. Neighborhood Components Analysis in sklearn to do classification.
    https://scikit-learn.org/stable/modules/neighbors.html

    Args:
        fold: The fold number to train and test.
        desired_stage: The stages that we want to classify.
    """
    with open(os.path.join(META_ROOT, fold_folder, "{:03d}.json".format(fold)), "r") as file:
        split_list = json.load(file)
    volume_dict = dict()
    stage_list = list()
    for stage in split_list:
        stage_list.append(stage)
        volume_dict[stage] = dict()
        for tvt in split_list[stage]:
            volume_list = []
            for identity in split_list[stage][tvt]:

                volume_path = os.path.join(
                    "../data", "volume", stage, identity, "volume.npy"
                )
                volumes = np.load(volume_path)

                volume_list.append(volumes)
            volume_array = np.stack(volume_list, axis=0)
            volume_dict[stage][tvt] = volume_array

    volume_train, volume_test, y_train, y_test = [], [], [], []
    for i, stage in enumerate(volume_dict):
        if stage in desired_stage:
            volume = np.concatenate([volume_dict[stage]["train"], volume_dict[stage]["val"]], axis=0)
            volume_train.append(volume)
            y_train.extend([i for _ in range(volume.shape[0])])
            volume = volume_dict[stage]["test"]
            volume_test.append(volume)
            y_test.extend([i for _ in range(volume.shape[0])])

    volume_train = np.concatenate(volume_train, axis=0)
    volume_test = np.concatenate(volume_test, axis=0)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(volume_train, y_train)

    return nca_pipe.score(volume_test, y_test)


def get_knn_acc():
    desired_stage = ["AD_pos", "NL_neg"]
    print("For {}, the 10 fold accuracy is: ".format(desired_stage))
    acc = []
    for i in range(10):
        acc.append(volume_knn_classification(i, desired_stage))
    acc = np.array(acc)
    print("Mean is {}".format(acc.mean()), "Std is {}".format(acc.std()))

    desired_stage = ["MCI_pos", "MCI_neg"]
    print("For {}, the 10 fold accuracy is: ".format(desired_stage))
    acc = []
    for i in range(10):
        acc.append(volume_knn_classification(i, desired_stage))
    acc = np.array(acc)
    print("Mean is {}".format(acc.mean()), "Std is {}".format(acc.std()))

    desired_stage = ["NL_pos", "NL_neg"]
    print("For {}, the 10 fold accuracy is: ".format(desired_stage))
    acc = []
    for i in range(10):
        acc.append(volume_knn_classification(i, desired_stage))
    acc = np.array(acc)
    print("Mean is {}".format(acc.mean()), "Std is {}".format(acc.std()))


if __name__ == '__main__':

    # load_train_json()
    # compute_volume()
    # plot_volume()
    get_knn_acc()
