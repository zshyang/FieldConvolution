import numpy as np
import torch
from data_code.visualize_sdf import load_sdf


def numpy_tensor(numpy_array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to a tensor.

    Args:
        numpy_array: The numpy array.

    Returns:
        torch_tensor: The torch tensor.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_tensor = torch.from_numpy(numpy_array).to(device)

    return torch_tensor


def get_sdf_threshold(sdf: np.ndarray, target_number: int):
    """Get the threshold of signed distance given a target number.

    Args:
        sdf: The
        target_number:

    Returns:
        threshold:
    """

    # A random initial value. It does not matter.
    number_fit = 1000000

    if target_number > sdf.shape[0]:
        raise ValueError("The number of target points should be lower than the points!")

    threshold_high = 0.5
    threshold_low = 0.0

    threshold = (threshold_high + threshold_low) / 2.0

    while number_fit != target_number:
        threshold = (threshold_high + threshold_low) / 2.0
        number_fit = np.sum(np.logical_and(sdf > -threshold, sdf < threshold))
        if number_fit > target_number:
            threshold_high = threshold
        else:
            threshold_low = threshold

    return threshold


def make_batch(points: [np.ndarray], sdfs: [np.ndarray], number_sample: int) -> {str: torch.Tensor}:
    """Make two list of points and sdfs on to GPU.

    Args:
        points: The list of the points.
        sdfs: The list of the signed distance fields.
        number_sample: The number points to be sampled on the surface.

    Returns:
        A dictionary of the input batch.
    """

    numbers = [point.shape[0] for point in points]
    min_number = min(numbers)

    batch_size = len(points)

    sampled_points = []
    sampled_sdfs = []
    thresholds = []
    for i in range(batch_size):
        index = np.random.randint(low=0, high=numbers[i], size=min_number)

        sampled_points.append(points[i][index])
        sampled_sdfs.append(sdfs[i][index])

        threshold = get_sdf_threshold(sampled_sdfs[i], number_sample)
        thresholds.append(threshold)

    # Load on to gpu.
    batch_points = numpy_tensor(np.stack(sampled_points))
    batch_sdfs = numpy_tensor(np.stack(sampled_sdfs))
    batch_thresholds = numpy_tensor(np.stack(thresholds))
    print(batch_thresholds.shape)

    return {
        "points": batch_points, "sdfs": batch_sdfs, "thresholds": batch_thresholds,
    }


def nearest_point_sample(xyz: torch.Tensor, number_sample: int, sdf: torch.Tensor) -> torch.Tensor:
    pass



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def forward(batch: {str: torch.Tensor}):

    # Get the convolution center index. (B, C). C is the number of convolution center.

    # Group the points around the convolution center given the distance to the center.

    # Get the weights given the convolution center.

    # Convolution input feature with the convolution weights.

    #
    pass


def test():

    # Load the sdf.
    point_0, sdf_0 = load_sdf("../../../data/141_S_1255_I297902_RHippo_60k.npz")
    point_1, sdf_1 = load_sdf("../../../data/1a6f615e8b1b5ae4dbbc9440457e303e.npz")

    # Make the sdf a batch.
    batch = make_batch([point_0, point_1], [sdf_0, sdf_1], number_sample=16 ** 3)

    # Forward the batch.
    forward(batch)


if __name__ == '__main__':
    test()
