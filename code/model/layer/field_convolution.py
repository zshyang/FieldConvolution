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


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Group the points given the index.

    Args:
        points: The input points data. (B, N, C)
        idx: The sample index data. (B, S)

    Returns:
        new_points: The indexed points data. (B, S, C)
    """

    device = points.device
    batch_size = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]

    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def abs_distance(src, dst):
    """Calculate absolute distance between each two points.
    dist = max(abs(xn - xm), abs(yn - ym), abs(zn - zm))

    Args:
        src: Source points, (B, N, C)
        dst: Target points, (B, M, C)

    Returns:
        dist: Per-point abs distance, (B, N, M)
    """

    assert len(src) == 3, "The source points should be a 3D tensor!"
    assert len(dst) == 3, "The target points should be a 3D tensor!"
    assert src.shape[0] == dst.shape[0], "The batch size should be same!"
    assert src.shape[-1] == 3, "The source points should be a point cloud!"
    assert dst.shape[-1] == 3, "The target points should be a point cloud!"

    batch_size, number_n, _ = src.shape
    _, number_m, _ = dst.shape

    dist = torch.max(
        torch.abs(
            src.view(batch_size, number_n, 1, 3) - dst.view(batch_size, 1, number_m, 3)
        ), dim=-1
    )

    return dist


def query_cube_point(
        edge: float, neighbor_sample_number: int, points: torch.Tensor, center_points: torch.Tensor
) -> torch.Tensor:
    """Query the index of the point cloud within the cube.

    Args:
        edge: The edge length of the cube.
        neighbor_sample_number: The number to be sampled within the cube.
        points: The large point cloud.
        center_points: The convolution center points.

    Returns:
        group_index: The index to be selected for the large point cloud.
    """

    # Conditions.
    assert len(points.shape) == 3, "The dimension of points should be 3!"
    assert len(center_points.shape) == 3, "The dimension of center points should be 3!"
    assert points.shape[-1] == 3, "The point should be in 3D space!"
    assert center_points.shape[-1] == 3, "The point should be in 3D space!"

    # Set up.
    device = points.device
    batch_size, number_points, number_channel = points.shape
    _, number_sample, _ = center_points.shape

    group_index = torch.arange(
        number_points, dtype=torch.long
    ).to(device).view(1, 1, number_points).repeat([batch_size, number_sample, 1])

    abs_distances = abs_distance(center_points, points)

    group_index[abs_distances > (edge / 2.0)] = number_points

    group_index = group_index.sort(dim=-1)[0][:, :, :neighbor_sample_number]

    # Make the points with index outside of the cube with the index within the cube.
    group_first = group_index[:, :, 0].view(batch_size, number_sample, 1).repeat(1, 1, neighbor_sample_number)
    mask = group_index == number_points
    group_index[mask] = group_first[mask]

    return group_index


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def group(index: torch.Tensor, xyz: torch.Tensor, neighbor_sample=32):

    new_xyz = index_points(xyz, index)  # (B, Ns, 3)
    print(xyz.shape, new_xyz.shape)
    torch.cuda.empty_cache()

    index = query_cube_point(
        edge=0.03, neighbor_sample_number=neighbor_sample,
        points=xyz, center_points=new_xyz
    )
    torch.cuda.empty_cache()

    grouped_xyz = index_points(xyz, index)
    print(grouped_xyz.shape)




    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points
    pass


def forward(batch: {str: torch.Tensor}):

    # Get the convolution center index. (B, C). C is the number of convolution center.
    index = torch.squeeze(batch["index"], -1)

    # Group the points around the convolution center given the distance to the center.
    group(index, batch["points"])


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
    forward(batch)


if __name__ == '__main__':
    test()
