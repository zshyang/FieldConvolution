import os
import json
import numpy as np
from glob import glob


np.random.seed(4124036635)

meta_folder = "../data/meta"
fold_folder = "10_fold"
dataset_path = os.path.join("/home/exx/georgey/dataset")
dataset = "hippocampus"
sub_dataset = "obj"
left_mesh_name = "LHippo_60k.obj"
right_mesh_name = "RHippo_60k.obj"
whole_mesh_name = ""


def split_train_val_test_list(meta_list: list, train: float, val: float):
    """Split the list with train, validation, and test.

    Args:
        meta_list: The list of meta information.
        train: The percentage of train data.
        val: The percentage of validation data.

    Returns:
        The dictionary of train, validation, and test list.
    """
    meta_np = np.array(meta_list)
    np.random.shuffle(meta_np)
    split_point_0 = int(train * len(meta_list))
    split_point_1 = int((train + val) * len(meta_list))
    return {
        "train": meta_np[:split_point_0].tolist(),
        "val": meta_np[split_point_0:split_point_1].tolist(),
        "test": meta_np[split_point_1:].tolist(),
    }


def split_train_val_test(meta_dict: {str: list}, train: float, val: float, test: float, number_fold: int):
    """Split the meta dictionary into train, validation, and test.

    Args:
        meta_dict: The meta information dictionary.
        train: The percentage of train data.
        val: The percentage of validation data.
        test: The percentage of test data.
        number_fold: The number of fold.

    Returns:

    """
    fold_split = []
    for _ in range(number_fold):
        fold_split_dict = {}
        for key in meta_dict:
            fold_split_dict[key] = split_train_val_test_list(meta_dict[key], train, val)
        fold_split.append(fold_split_dict)
    return fold_split


def generate_10_fold_meta():
    # make the folder to save the 10 fold meta information.
    os.makedirs(os.path.join(meta_folder, "10_fold"), exist_ok=True)

    # save the stages and identities inside each stage.
    stage_identity_dict = dict()
    stage_identities = glob(
        os.path.join(dataset_path, dataset, sub_dataset, "*", "*")
    )
    for stage_identity in stage_identities:
        stage = stage_identity.split("/")[-2]
        identity = stage_identity.split("/")[-1]
        if stage in stage_identity_dict:
            stage_identity_dict[stage].append(identity)
        else:
            stage_identity_dict[stage] = [identity]

    # split the dictionary.
    split_fold = split_train_val_test(stage_identity_dict, 0.64, 0.16, 0.20, 10)

    # save the split fold.
    for i in range(10):
        with open(os.path.join(meta_folder, fold_folder, "{:03d}.json".format(i)), "w") as file:
            json.dump(split_fold[i], file)


if __name__ == '__main__':
    generate_10_fold_meta()
