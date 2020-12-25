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
        sdf: The signed distance field. (N, 1)
        target_number: The target number to sample in the signed distance field.

    Returns:
        threshold: The threshold.
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

        if threshold_high - threshold_low <= 1e-10:
            break

        number_fit = np.sum(np.logical_and(sdf > -threshold, sdf < threshold))
        if number_fit > target_number:
            threshold_high = threshold
        else:
            threshold_low = threshold

    return threshold


def nearest_index(sdf: np.ndarray, number_sample: int) -> np.ndarray:
    """Return the index of the nearest points around the surface of the mesh.

    Args:
        sdf: The signed distance field. (N, 1)
        number_sample: The number of sample.

    Returns:
        sort_index: The index. (Ns, 1)
    """

    sort_index = np.argsort(np.abs(sdf), axis=0)
    sort_index = sort_index[:number_sample, :]

    return sort_index


def make_batch(points: [np.ndarray], sdfs: [np.ndarray], number_sample: int) -> {str: torch.Tensor}:
    """Make two list of points and sdfs on to GPU.

    Args:
        points: The list of the points.
        sdfs: The list of the signed distance fields.
        number_sample: The number points to be sampled on the surface.

    Returns:
        A dictionary of the input batch.
            "points": The batch points. (B, N, 3)
            "sdfs": The batch signed distance fields. (B, N, 1)
            "index": The batch index. (B, Ns, 1)
    """

    numbers = [point.shape[0] for point in points]
    min_number = min(numbers)

    batch_size = len(points)

    sampled_points = []
    sampled_sdfs = []
    sampled_index = []
    for i in range(batch_size):
        index = np.random.randint(low=0, high=numbers[i], size=min_number)

        sampled_points.append(points[i][index])
        sampled_sdfs.append(sdfs[i][index])

        index = nearest_index(sampled_sdfs[i], number_sample)
        sampled_index.append(index)

    # Load on to gpu.
    batch_points = numpy_tensor(np.stack(sampled_points))
    batch_sdfs = numpy_tensor(np.stack(sampled_sdfs))
    batch_index = numpy_tensor(np.stack(sampled_index))

    return {
        "points": batch_points, "sdfs": batch_sdfs, "index": batch_index,
    }


def nearest_point_sample(sdf: torch.Tensor, thresholds: torch.Tensor, number_sample: int) -> torch.Tensor:

    batch_size, number_points, _ = sdf.shape
    device = sdf.device

    # Get the mask first.
    thresholds = torch.unsqueeze(thresholds, -2)  # (B, 1, 1)
    mask = torch.abs(sdf) > thresholds  # (B, N, 1)

    group_index = torch.arange(number_points, dtype=torch.long).to(device).view(1, number_points, 1).repeat([batch_size, 1, 1])
    group_index[mask] = number_points
    group_index = group_index.sort(dim=-2)[0][:, :number_sample, :]
    print(group_index.shape)


    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]


    return mask


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
    index = torch.squeeze(batch["index"], -1)

    # Group the points around the convolution center given the distance to the center.


    # Get the weights given the convolution center.

    # Convolution input feature with the convolution weights.

    #
    return True


def test():

    # Load the sdf.
    point_0, sdf_0 = load_sdf("../../../data/141_S_1255_I297902_RHippo_60k.npz")
    point_1, sdf_1 = load_sdf("../../../data/1a6f615e8b1b5ae4dbbc9440457e303e.npz")

    # Make the sdf a batch.
    batch = make_batch([point_0, point_1], [sdf_0, sdf_1], number_sample=16 ** 3)

    # Forward the batch.
    forward(batch, number_sample=16 ** 3)


if __name__ == '__main__':
    test()
