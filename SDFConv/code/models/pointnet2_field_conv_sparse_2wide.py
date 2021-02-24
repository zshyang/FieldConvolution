"""Add field convolution to the first layer of pointnet2.
----ZhangsihaoYang.Jan.10.2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_util import PointNetSetAbstraction
from easydict import EasyDict
from models.layers.field_convolution import FieldConv
import sparseconvnet as scn


block_reps = 1
m = 16
residual_blocks = False


class Net(nn.Module):
    """The PointNet++ classification model.
    """
    def __init__(self, options: EasyDict):
        """The initialization function.

        Args:
            options: The options to define the model.
        """
        super(Net, self).__init__()

        self.in_channel = 16  # Hard coded here.

        self.sparseModel = scn.SparseVggNet(
            3, 1,
            [
                ['C', 16], ['C', 16], 'MP',
                ['C', 32], ['C', 32], 'MP',
                ['C', 48], ['C', 48], 'MP',
                ['C', 64], ['C', 64], 'MP',
                ['C', 96], ['C', 96], "MP",
                ['C', 128], ['C', 128], "MP"
            ]
        ).add(
            scn.Convolution(3, 128, 128, 3, 2, False)
        ).add(
            scn.BatchNormReLU(128)
        ).add(scn.SparseToDense(3, 128))
        # print(self.sparseModel)
        # self.spatial_size = self.sparseModel.input_spatial_size(
        #     torch.LongTensor([512, 512, 512])
        # )
        # self.inputLayer = scn.InputLayer(2, self.spatial_size, 2)
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 2)

        self.input_layer = scn.InputLayer(3, 255, mode=4)
        # self.linear = nn.Linear(m, 20)

    def forward(self, batch: dict) -> dict:
        """The forward function.

        Args:
            batch: The input batch.
                "xyz_sdf": The input point cloud concatenated with signed distance field.

        Returns:
            The output batch.
        """
        xyz_sdf = batch["xyz_sdf"]

        batch_size, num_points, _ = xyz_sdf.shape

        # batch_index = torch.arange(batch_size)
        device = xyz_sdf.device

        view_shape = list(xyz_sdf.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)

        repeat_shape = list(xyz_sdf.shape)
        repeat_shape[0] = 1
        repeat_shape[-1] = 1
        batch_index = torch.arange(batch_size, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

        locations = xyz_sdf[:, :, :3]
        locations = locations - locations.min()
        locations = locations * 200
        locations = locations.long()

        locations = torch.cat([locations.view(-1, 3), batch_index.view(-1, 1)], dim=-1)
        locations = locations.long()

        features = xyz_sdf[:, :, 3:]
        features = features.view(-1, 1)

        # print(locations.shape, features.shape)

        # field_feature = self.field_conv(xyz_sdf)
        # field_feature = field_feature.permute(0, 2, 1)
        #
        # l1_xyz, l1_points = self.sa1(field_feature[:, :3, :], field_feature[:, 3:, :])
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #
        # x = l3_points.view(batch_size, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)

        x = self.input_layer([locations, features])
        # print(x)
        x = self.sparseModel(x)
        # print(x.shape)
        x = x.view(-1, 128)
        x = self.linear1(x)
        x = self.linear2(x)
        # print(x.shape)

        return {"pred_label": x, }


def test():
    """The test function just for the network shape checking.
    """

    torch.manual_seed(0)
    dim_b = 32
    dim_n = 1600
    batch = {
        "dist_map": torch.randn(dim_b, dim_n, dim_n),
        "padded_verts": torch.randn(dim_b, dim_n, 3),
        "lrf": torch.randn(dim_b, dim_n, 3, 3),
        "label": torch.randn(dim_b),
        "normal": torch.randn(dim_b, dim_n, 3),
        "xyz": torch.randn(dim_b, 3, dim_n),
        "xyz_sdf": torch.randn(dim_b, dim_n, 4)
    }

    options = EasyDict()
    options.model = EasyDict()

    # options.model.base_dim = 4
    # options.model.base_radius = 0.05
    options.model.out_channel = 2

    gkcnet = Net(options)

    out = gkcnet(batch)

    print(out)


if __name__ == '__main__':
    test()
