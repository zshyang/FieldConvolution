import pyrender
import trimesh


mesh = trimesh.load('/home/exx/georgey/dataset/hippocampus/obj/AD_pos/100_S_5106_I369381/LHippo_60k.obj')

# create the scene.
scene = pyrender.Scene()

cloud = pyrender.Mesh.from_points(mesh.vertices)

scene.add(cloud)

dict_args = {"use_raymond_lighting": True, "point_size": 2}
viewer = pyrender.Viewer(scene, **dict_args)
