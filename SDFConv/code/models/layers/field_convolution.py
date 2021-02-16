import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_weight(weight, xyz):
    print(weight.shape, xyz.shape)
    indexx = 3
    indexy = 0
    weight_sample = weight[:, :, :, indexx, indexy]
    xyz_np = xyz.view(-1, 3).cpu().detach().numpy()
    print(weight_sample.shape)
    weight_py = weight_sample.view(-1).cpu().detach().numpy()
    print(type(xyz_np))
    print(xyz_np.shape)
    print(xyz_np.min(0))
    print(xyz_np.max(0))
    wmin = weight_py.min()
    wmax = weight_py.max()
    wmid = wmin / 2 + wmax / 2

    sample_number = 10000
    weight_py = weight_py[:10000]
    xyz_np = xyz_np[:10000, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = xyz_np[weight_py < wmid, 0]
    y = xyz_np[weight_py < wmid, 1]
    z = xyz_np[weight_py < wmid, 2]
    # z = z * 0

    ax.scatter(x, y, z, c='r', marker='o')

    x = xyz_np[weight_py >= wmid, 0]
    y = xyz_np[weight_py >= wmid, 1]
    z = xyz_np[weight_py >= wmid, 2]
    # z = z * 0

    ax.scatter(x, y, z, c='g', marker='x')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    print(xyz_np.dafsdf())


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
        self, edge_length: float, filter_sample_number: int, center_number: int, in_channels: int, out_channels: int,
        feature_is_sdf: bool
    ):
        """The initialization function.

        Args:
            edge_length: The length of the cube. Equivalent to the length of the filter.
            filter_sample_number: The number of point to be sampled within the filter.
                Equivalent to the filter size. Noted as Nn.
            center_number: The number of convolution centers. Equivalent to the stride of the filter. Noted as Nc.
            in_channels (int): Number of channels in the input image. Noted as I.
            out_channels (int): Number of channels produced by the convolution. Noted as O.
            feature_is_sdf: The indicator of whether the input contains sdf as its feature.
        """

        super(FieldConv, self).__init__()

        self.edge_length = edge_length
        self.filter_sample_number = filter_sample_number
        self.center_number = center_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_is_sdf = feature_is_sdf

        self.weight_net = WeightNet(
            3,
            out_channels * in_channels
        )  # the in channel is hard coded with plus 3

        self.bias = Parameter(torch.empty(1, 1, out_channels))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, in_feature: torch.Tensor):
        """The forward function.

        Args:
            in_feature: The input feature with the concatenation of the following two tensors.
               points: The input point clouds. (B, N, 3)
               feature: The signed distance fields. (B, N, Fin)

        Returns:
            out_feature: The output feature with the concatenation of the following two tensors.
               points: The input point clouds. (B, Nc, 3)
               feature: The signed distance fields. (B, Nc, Fout)
        """

        points = in_feature[:, :, :3]
        feature = in_feature[:, :, 3:]

        assert len(points.shape) == 3, "The input point cloud should be batched!"
        assert len(feature.shape) == 3, "The input signed distance field should be batched!"
        assert points.shape[2] == 3, "The input point cloud should be in 3D space!"

        # Get the convolution center index. (B, C). C is the number of convolution center.
        if self.feature_is_sdf:
            _, indices = torch.sort(torch.abs(feature), 1)
            indices = indices[:, :self.center_number, :]
            indices = torch.squeeze(indices, -1)
        else:
            _, indices = torch.sort(torch.norm(feature, dim=-1, keepdim=True), 1)
            indices = indices[:, :self.center_number, :]
            indices = torch.squeeze(indices, -1)

        # Group the points around the convolution center given the distance to the center.
        grouped_xyz_norm, grouped_feature, new_xyz = group(
            indices, points, feature, self.edge_length, self.filter_sample_number
        )  # (B, Nc, Nn, 3), (B, Bc, Nn, Fin), (B, Nc, 3)

        # Get the weights given the convolution center.
        weight = self.weight_net(grouped_xyz_norm)  # (B, Nc, Nn, I * O)
        weight_shape = list(weight.shape)
        weight_shape[-1] = self.out_channels
        weight_shape.append(self.in_channels)
        weight = weight.view(weight_shape)

        plot = False
        if plot:
            plot_weight(weight, grouped_xyz_norm)

        # Convolution input feature with the convolution weight and bias.
        feature = torch.unsqueeze(
            grouped_feature,
            -1
        )  # (B, Nc, Nn, I, 1)
        feature = torch.einsum("abcij,abcjk->abcik", weight, feature)  # (B, Nc, Nn, O, 1)
        feature = torch.squeeze(feature, -1)  # (B, Nc, Nn, O)
        feature = torch.max(feature, 2)[0] + self.bias

        # Concatenate the new location and new feature.
        out_feature = torch.cat([new_xyz, feature], -1)

        return out_feature


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
            nn.BatchNorm1d(self.dim_hid),
            nn.ReLU(True),
            nn.Linear(in_features=self.dim_hid, out_features=self.dim_hid),
            nn.BatchNorm1d(self.dim_hid),
            nn.ReLU(True),
            nn.Linear(in_features=self.dim_hid, out_features=self.dim_out),
            # nn.BatchNorm1d(self.dim_out),
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
    from FieldConvolution.data_code.visualize_sdf import load_sdf

    # Load the sdf.
    point_0, sdf_0 = load_sdf("../../../data/141_S_1255_I297902_RHippo_60k.npz")
    point_1, sdf_1 = load_sdf("../../../data/1a6f615e8b1b5ae4dbbc9440457e303e.npz")

    # Make the sdf a batch.
    batch = make_batch([point_0, point_1], [sdf_0, sdf_1], number_sample=16 ** 3)

    # Forward the batch.
    field_conv = FieldConv(
        edge_length=0.03, filter_sample_number=64, center_number=16**3, in_channels=1, out_channels=2,
        feature_is_sdf=False
    ).cuda()

    feature = field_conv(torch.cat([batch["points"], batch["sdfs"]], -1))
    print(feature.shape)


if __name__ == '__main__':
    test()
