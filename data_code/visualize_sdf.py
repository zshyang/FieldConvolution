import numpy as np
import pyrender


def load_sdf(file_name: str) -> (np.ndarray, np.ndarray):
    """Load the signed distance file.

    Args:
        file_name: The name of the file.

    Returns:
        points: The locations. (N, 3)
        sdf: The distance. (N, 1)
    """

    data = np.load(file_name, mmap_mode="r")

    point_sfd = np.concatenate((data["pos"], data["neg"]), axis=0)

    points = point_sfd[:, :3]
    sdf = point_sfd[:, [-1]]

    return points, sdf


def visualize_sdf(points: np.ndarray, sdf: np.ndarray, scene: pyrender.scene.Scene) -> pyrender.scene.Scene:
    """Visualize the signed distance field.

    Args:
        points: The locations. (N, 3)
        sdf: The distance. (N, 1)
        scene: The scene to render the point cloud.

    Returns:
        scene: The scene to render the point cloud.
    """

    colors = np.zeros(points.shape)
    colors[sdf[:, 0] < 0, 2] = 1
    colors[sdf[:, 0] > 0, 0] = 1

    cloud = pyrender.Mesh.from_points(points, colors=colors)

    scene.add(cloud)

    return scene


def main():
    points, sdf = load_sdf("../data/141_S_1255_I297902_RHippo_60k.npz")
    scene = pyrender.Scene()
    visualize_sdf(points, sdf, scene)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


if __name__ == '__main__':
    main()
