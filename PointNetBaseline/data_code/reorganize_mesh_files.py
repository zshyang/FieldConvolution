import os
import shutil
import trimesh
from glob import glob
from tqdm import tqdm
from parse import parse
import numpy as np


dataset_path = os.path.join("/home/exx/georgey/dataset")
dataset = "hippocampus"
sub_dataset = "obj"
left_mesh_name = "LHippo_60k.obj"
right_mesh_name = "RHippo_60k.obj"
whole_mesh_name = ""


def m2obj(m_file, obj_file):
    """Convert m file to obj file.

    Args:
        m_file:
        obj_file:
    """
    verts = []
    faces = []
    with open(m_file, "r") as file:
        vert_idx = 1
        vert_map = {}
        for line in file:
            if line[0] == "V":
                vert = parse("Vertex {} {} {} {} {}", line)
                verts.append([float(vert[1]), float(vert[2]), float(vert[3])])
                vert_map.update({int(vert[0]): vert_idx})
                vert_idx = vert_idx + 1
            if line[0] == "F":
                face = parse("Face {} {:d} {:d} {:d} {}", line)
                # faces.append([face[1], face[2], face[3]])
                faces.append([vert_map[face[1]], vert_map[face[2]], vert_map[face[3]]])
    with open(obj_file, "w") as file:
        for vert in verts:
            file.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))
        for face in faces:
            file.write("f {} {} {}\n".format(face[0], face[1], face[2]))


def reorganize_mesh_files():
    # Get the name of the mesh files.
    # /home/exx/georgey/dataset/hippocampus/obj/AD_pos/002_S_0729_I291876_LHippo_60k.obj
    mesh_file_template = os.path.join(dataset_path, dataset, sub_dataset, "*", "*.obj")
    mesh_file_names = glob(mesh_file_template)

    for mesh_file_name in tqdm(mesh_file_names):
        if os.path.exists(mesh_file_name):

            stage = mesh_file_name.split("/")[-2]
            identity = mesh_file_name.split("/")[-1][:-15]
            specific_file_name = mesh_file_name.split("/")[-1][-14:]

            target_obj_file_name = os.path.join(
                dataset_path, dataset, sub_dataset, stage, identity, specific_file_name
            )
            target_obj_folder = os.path.join(
                dataset_path, dataset, sub_dataset, stage, identity
            )

            os.makedirs(target_obj_folder, exist_ok=True)

            shutil.move(src=mesh_file_name, dst=target_obj_file_name)


def convert_all_m_file_obj_file():
    """Convert all the m file to obj format.
    """

    # Get all the names of the m files.
    all_m_file_names = glob(os.path.join(dataset_path, "Hippocampus", "surface", "*", "*60k.m"))

    for m_file_name in tqdm(all_m_file_names):

        # Get stage, identity, and specific file name.
        stage = m_file_name.split("/")[-2]
        identity = m_file_name.split("/")[-1][:-13]
        specific_file_name = m_file_name.split("/")[-1][-12:-2]

        # Get the obj file name and folder name to save the obj file.
        target_obj_file_name = os.path.join(
            dataset_path, dataset, sub_dataset, stage, identity, specific_file_name + ".obj"
        )
        target_obj_folder = os.path.join(
            dataset_path, dataset, sub_dataset, stage, identity
        )
        os.makedirs(target_obj_folder, exist_ok=True)

        # Generate the obj file if not exist.
        if os.path.exists(target_obj_file_name):
            continue
        else:
            m2obj(m_file_name, target_obj_file_name)


def check_all_obj_file_good():
    """Check all the obj file could be load by trimesh.
    """

    obj_file_names = glob(
        os.path.join(
            dataset_path, dataset, sub_dataset, "*", "*", "*.obj"
        )
    )
    for obj_file_name in tqdm(obj_file_names):
        dict_args = {"process": False}
        _ = trimesh.load(obj_file_name, **dict_args)


def merge_all_left_right():
    all_folders = glob(
        os.path.join(
            dataset_path, dataset, sub_dataset, "*", "*"
        )
    )
    for mesh_folder in all_folders:
        dict_args = {"process": False}
        left_mesh = trimesh.load(
            os.path.join(mesh_folder, left_mesh_name), **dict_args
        )
        right_mesh = trimesh.load(
            os.path.join(mesh_folder, right_mesh_name), **dict_args
        )
        left_vertices = np.array(left_mesh.vertices, dtype=np.float)
        left_faces = np.array(left_mesh.faces, dtype=np.int)
        right_vertices = np.array(right_mesh.vertices, dtype=np.float)
        right_faces = np.array(right_mesh.faces, dtype=np.int)

        vertices = np.concatenate((left_vertices, right_vertices), axis=0)
        faces = np.concatenate(
            (left_faces, right_faces + left_vertices.shape[0]), axis=0
        )

        whole_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False
        )
        whole_mesh.show()

        
if __name__ == '__main__':
    check = False
    reorganize_mesh_files()
    convert_all_m_file_obj_file()
    if check:
        check_all_obj_file_good()
    merge_all_left_right()
