import os
import json
import trimesh


META_ROOT = "../data/meta"
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


if __name__ == '__main__':

    load_train_json()
    compute_volume()
