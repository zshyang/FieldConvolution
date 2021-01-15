import os
from glob import glob


dataset_path = os.path.join("/home/exx/georgey/dataset")
dataset = "hippocampus"
sub_dataset = "obj"


def reorganize_mesh_files():
    # Get the name of the mesh files.
    # /home/exx/georgey/dataset/hippocampus/obj/AD_pos/002_S_0729_I291876_LHippo_60k.obj
    mesh_file_template = os.path.join(dataset_path, dataset, sub_dataset, "*", "*.obj")
    mesh_file_names = glob(mesh_file_template)

    for mesh_file_name in mesh_file_names:
        if os.path.exists(mesh_file_name):

            stage = mesh_file_name.split("/")[-2]
            identity = mesh_file_name.split("/")[-1][:-15]
            specific_file_name = mesh_file_name.split("/")[-1][-14:]

            target_obj_file_name = os.path.join(
                dataset_path, dataset, sub_dataset, stage, identity, specific_file_name
            )

            print(target_obj_file_name)

        break


if __name__ == '__main__':
    reorganize_mesh_files()
