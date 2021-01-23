from mesh_to_sdf import mesh_to_voxels

import trimesh
import numpy as np
import skimage.measure
import pyrender

mesh = trimesh.load('/home/exx/georgey/dataset/hippocampus/obj/AD_pos/100_S_5106_I369381/LHippo_60k.obj')

# voxels = mesh_to_voxels(mesh, 64, pad=True)
# print(voxels.shape)
#
# vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=0)
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# mesh.show()
from mesh_to_sdf import sample_sdf_near_surface
points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
print(points.min(), points.max())
colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
