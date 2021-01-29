import os
import shutil
import trimesh
import numpy as np

from glob import glob
from tqdm import tqdm


left_mesh_name = "LHippo_60k.obj"
right_mesh_name = "RHippo_60k.obj"


def move_meta():
    meta_folder_path = os.path.join("..", "..", "PointNetBaseline", "data", "meta")
    meta_folder_path = os.path.abspath(meta_folder_path)

    target_folder_path = os.path.join("../data/meta")
    target_folder_path = os.path.abspath(target_folder_path)

    shutil.copytree(meta_folder_path, target_folder_path)


def load_merge_left_right_mesh(mesh_root, stage, identity):
    # get the name of the mesh
    left_mesh_name_ = os.path.join(
        mesh_root, stage, identity, left_mesh_name
    )
    right_mesh_name_ = os.path.join(
        mesh_root, stage, identity, right_mesh_name
    )

    # load the mesh
    dict_args = {"process": False}
    left_mesh = trimesh.load(left_mesh_name_, **dict_args)
    right_mesh = trimesh.load(right_mesh_name_, **dict_args)

    # concatenate the vertices
    left_vertices = np.array(left_mesh.vertices, dtype=np.float32)
    right_vertices = np.array(right_mesh.vertices, dtype=np.float32)
    vertices = np.concatenate((left_vertices, right_vertices), axis=0)

    # concatenate the faces
    left_faces = np.array(left_mesh.faces, dtype=np.int)
    right_faces = np.array(right_mesh.faces, dtype=np.int)
    faces = np.concatenate([left_faces, right_faces + left_vertices.shape[0]], axis=0)

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False
    )

    return mesh


def generate_meta_size():
    # get the name to save the size meta
    meta_folder = os.path.join("../data/meta")
    meta_folder = os.path.abspath(meta_folder)
    meta_fn = os.path.join(meta_folder, "size.json")  # fn is filename.

    # gather the list of identity
    stage_identity_list = glob("/home/exx/georgey/dataset/hippocampus/obj/*/*")
    mesh_root = os.path.join("/home/exx/georgey/dataset/hippocampus/obj")

    # go over the mesh files
    for stage_identity in tqdm(stage_identity_list):
        print(stage_identity)
        # load and merge the mesh
        stage = stage_identity.split("/")[-2]
        identity = stage_identity.split("/")[-1]
        mesh = load_merge_left_right_mesh(mesh_root, stage, identity)
        mesh.show()
        # find the name of the mesh.


        # furthest point sampling the vertices
        fps_vertices = farthest_point_sample(vertices, npoint=2500)

        # center, and scale the mesh
        centered_vertices = fps_vertices - np.expand_dims(np.mean(fps_vertices, axis=0), 0)  # center
        if self.dataset.scalar is None:
            dist = np.max(np.sqrt(np.sum(centered_vertices ** 2, axis=1)), 0)
            scaled_vertices = centered_vertices / dist  # scale
        else:
            scaled_vertices = centered_vertices / self.dataset.scalar  # scale
        # compute the bounding box radius
        # save the radius into the


def main():
    moved = True
    if not moved:
        move_meta()

    # generate the size meta.
    generate_meta_size()


if __name__ == '__main__':
    main()
