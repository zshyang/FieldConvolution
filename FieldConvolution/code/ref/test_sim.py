import trimesh

mesh = trimesh.load_mesh("lh.obj")
import open3d

open3d_mesh = open3d.geometry.TriangleMesh(
    vertices=open3d.utility.Vector3dVector(mesh.vertices),
    triangles=open3d.utility.Vector3iVector(mesh.faces)
)

simple = open3d_mesh.simplify_quadric_decimation(10000)

simple = trimesh.Trimesh(vertices=simple.vertices, faces=simple.triangles)
print(simple)
# simple.show()
simple.export("test.obj")
# print(mesh.as_open3d)
# trimesh.base.Trimesh.simplify_quadratic_decimation
# mesh.simplify_quadratic_decimation(10000)
print(mesh)
# mesh.show()
