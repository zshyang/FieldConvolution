import numpy as np


def farthest_point_sample(point, npoint):
    """Farthest point sampling for numpy implementation.

    Input:
        xyz: point cloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled point cloud index, [npoint, D]
    """
    number_points, point_cloud_dimension = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((number_points,)) * 1e10
    farthest = np.random.randint(0, number_points)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
