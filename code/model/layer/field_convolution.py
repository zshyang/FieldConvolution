import numpy as np
import torch
import torch.nn as nn
from data_code.visualize_sdf import load_sdf
from torch.nn.parameter import Parameter
import math


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
    min_number = int(min(numbers) / 30)

    batch_size = len(points)

    sampled_points = []
    sampled_sdfs = []
    for i in range(batch_size):
        index = np.random.randint(low=0, high=numbers[i], size=min_number)

        sampled_points.append(points[i][index])
        sampled_sdfs.append(sdfs[i][index])

    # Load on to gpu.
    batch_points = numpy_tensor(np.stack(sampled_points))
    batch_sdfs = numpy_tensor(np.stack(sampled_sdfs))

    return {
        "points": batch_points, "sdfs": batch_sdfs,
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

    assert len(src.shape) == 3, "The source points should be a 3D tensor!"
    assert len(dst.shape) == 3, "The target points should be a 3D tensor!"
    assert src.shape[0] == dst.shape[0], "The batch size should be same!"
    assert src.shape[-1] == 3, "The source points should be a point cloud!"
    assert dst.shape[-1] == 3, "The target points should be a point cloud!"

    batch_size, number_n, _ = src.shape
    _, number_m, _ = dst.shape

    dist = torch.max(
        torch.abs(
            src.view(batch_size, number_n, 1, 3) - dst.view(batch_size, 1, number_m, 3)
        ), dim=-1
    )[0]

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


def group(
        index: torch.Tensor, xyz: torch.Tensor, feature: torch.Tensor, edge_length: float, neighbor_sample=32
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """Group the point cloud and feature given the center index and the length of the edge.

    Args:
        index: The index of the center points. (B, Nc). c stand for center.
        xyz: The point cloud. (B, N, 3)
        feature: The feature associated with each point. (B, N, F)
        edge_length: The length of the edge of the cube.
        neighbor_sample: The number of point to be sampled within each center. Noted with Nn.

    Returns:
        grouped_xyz_norm: The normalized grouped point cloud. (B, Nc, Nn, 3)
        grouped_feature: The grouped feature. (B, Bc, Nn, F)
        new_xyz: The new position of the point cloud. (B, Nc, 3)
    """

    assert len(index.shape) == 2, "The index dimension is should be 2 not {}!".format(len(index.shape))
    assert len(xyz.shape) == 3, "The input point cloud dimension should be 3 not {}!".format(len(xyz.shape))
    assert len(feature.shape) == 3, "The input feature dimension should be 3 not {}!".format(len(feature.shape))
    assert index.shape[0] == xyz.shape[0] == feature.shape[0], "The batch size should be same!"
    assert xyz.shape[2] == 3, "The point cloud should be in 3D space!"

    batch_size, _, number_channel = xyz.shape
    _, number_center = index.shape

    new_xyz = index_points(xyz, index)  # (B, Nc, 3)
    torch.cuda.empty_cache()

    index = query_cube_point(
        edge=edge_length, neighbor_sample_number=neighbor_sample,
        points=xyz, center_points=new_xyz
    )  # (B, Nc, Nn)
    torch.cuda.empty_cache()

    grouped_xyz = index_points(xyz, index)  # (B, Nc, Nn, 3)
    grouped_feature = index_points(feature, index)  # (B, Nc, Nn, F)
    torch.cuda.empty_cache()

    grouped_xyz_norm = grouped_xyz - new_xyz.view(batch_size, number_center, 1, number_channel)
    torch.cuda.empty_cache()

    return grouped_xyz_norm, grouped_feature, new_xyz


class FieldConv(nn.Module):
    """Field convolution layer.
    """
    def __init__(
        self, edge_length: float, filter_sample_number: int, center_number: int, in_channels: int, out_channels: int
    ):
        """The initialization function.

        Args:
            edge_length: The length of the cube. Equivalent to the length of the filter.
            filter_sample_number: The number of point to be sampled within the filter.
                Equivalent to the filter size. Noted as Nn.
            center_number: The number of convolution centers. Equivalent to the stride of the filter. Noted as Nc.
            in_channels (int): Number of channels in the input image. Noted as I.
            out_channels (int): Number of channels produced by the convolution. Noted as O.
        """

        super(FieldConv, self).__init__()

        self.edge_length = edge_length
        self.filter_sample_number = filter_sample_number
        self.center_number = center_number
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight_net = WeightNet(3, out_channels * in_channels)

        self.bias = Parameter(torch.empty(1, 1, out_channels))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, inputs: {str: torch.Tensor}):
        """The forward function.

        Args:
            inputs: The dictionary of the inputs.
               points: The input point clouds. (B, N, 3)
               sdfs: The signed distance fields. (B, N, 1)

        Returns:

        """

        points = inputs["points"]
        sdfs = inputs["sdfs"]

        assert len(points.shape) == 3, "The input point cloud should be batched!"
        assert len(sdfs.shape) == 3, "The input signed distance field should be batched!"
        assert points.shape[2] == 3, "The input point cloud should be in 3D space!"
        assert sdfs.shape[2] == 1, "The signed distance field should have 1 at last dimension!"

        # Get the convolution center index. (B, C). C is the number of convolution center.
        _, indices = torch.sort(torch.abs(sdfs), 1)
        indices = indices[:, :self.center_number, :]
        indices = torch.squeeze(indices, -1)

        # Group the points around the convolution center given the distance to the center.
        grouped_xyz_norm, grouped_feature, new_xyz = group(
            indices, points, sdfs, self.edge_length, self.filter_sample_number
        )  # (B, Nc, Nn, 3), (B, Bc, Nn, F), (B, Nc, 3)

        # Get the weights given the convolution center.
        weight = self.weight_net(grouped_xyz_norm)  # (B, Nc, Nn, I * O)
        weight_shape = list(weight.shape)
        weight_shape[-1] = self.out_channels
        weight_shape.append(self.in_channels)
        weight = weight.view(weight_shape)

        # Convolution input feature with the convolution weight and bias.
        feature = torch.unsqueeze(grouped_feature, -1)  # (B, Nc, Nn, I, 1)
        print(weight.shape, feature.shape)
        feature = torch.einsum("abcij,abcjk->abcik", weight, feature)  # (B, Nc, Nn, O, 1)
        feature = torch.squeeze(feature, -1)  # (B, Nc, Nn, O)
        feature = torch.max(feature, 2)[0] + self.bias

        return feature


class WeightNet(nn.Module):
    """The network used to compute weight.
    """
    def __init__(self, dim_in: int, dim_out: int):
        """The initialization function.
        """
        super(WeightNet, self).__init__()

        assert dim_in == 3, "The input should be a position of a point!"

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hid = dim_in * 2  # To keep the network's size.
        self.weight_net = nn.Sequential(
            nn.Linear(in_features=self.dim_in, out_features=self.dim_hid),
            nn.ReLU(True),
            nn.BatchNorm1d(self.dim_hid),
            nn.Linear(in_features=self.dim_hid, out_features=self.dim_hid),
            nn.ReLU(True),
            nn.BatchNorm1d(self.dim_hid),
            nn.Linear(in_features=self.dim_hid, out_features=self.dim_out),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """The forward function.

        Args:
            feature: The input feature. (B, N, K, F)

        Returns:
            The feature after the weight net.
        """

        assert len(feature.shape) == 4, "The dimension of the feature should be 4 not {}!".format(len(feature.shape))
        assert feature.shape[3] == self.dim_in, "The input feature should have same channel size as setting!"

        view_shape = list(feature.shape)
        view_shape[-1] = self.dim_out

        return self.weight_net(feature.view(-1, 3)).view(view_shape)


def test():

    # Load the sdf.
    point_0, sdf_0 = load_sdf("../../../data/141_S_1255_I297902_RHippo_60k.npz")
    point_1, sdf_1 = load_sdf("../../../data/1a6f615e8b1b5ae4dbbc9440457e303e.npz")

    # Make the sdf a batch.
    batch = make_batch([point_0, point_1], [sdf_0, sdf_1], number_sample=16 ** 3)

    # Forward the batch.
    field_conv = FieldConv(
        edge_length=0.03, filter_sample_number=64, center_number=16**3, in_channels=1, out_channels=2
    ).cuda()

    feature = field_conv(batch)
    print(feature.shape)


if __name__ == '__main__':
    test()
