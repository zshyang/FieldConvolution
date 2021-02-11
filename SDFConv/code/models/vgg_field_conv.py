"""Add field convolution to the first layer of pointnet2.
----ZhangsihaoYang.Jan.10.2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_util import PointNetSetAbstraction
from easydict import EasyDict
from models.layers.field_convolution import FieldConv
from models.layers.sdf_pooling import FieldPooling


class Net(nn.Module):
    """The PointNet++ classification model.
    """
    def __init__(self, options: EasyDict):
        """The initialization function.

        Args:
            options: The options to define the model.
        """
        super(Net, self).__init__()

        self.base_channel = 64  # Hard coded here.

        image_size = 224.0
        cube_size = 2.0

        # the vgg network
        self.sdf_conv_1 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=(224 * 224), in_channels=1,
            out_channels=self.base_channel,
            feature_is_sdf=True,
        )
        self.sdf_conv_2 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=(224 * 224), in_channels=self.base_channel,
            out_channels=self.base_channel,
            feature_is_sdf=False,
        )
        self.sdf_max_pooling_1 =
        self.sdf_conv_3 =





        self.num_class = options.model.out_channel

        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=self.in_channel+3, mlp=[64, 64, 128], group_all=False
        )  # The in channels are increased by 3.
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, self.num_class)

    def forward(self, batch: dict) -> dict:
        """The forward function.

        Args:
            batch: The input batch.
                "xyz_sdf": The input point cloud concatenated with signed distance field.

        Returns:
            The output batch.
        """
        xyz_sdf = batch["xyz_sdf"]

        batch_size, _, _ = xyz_sdf.shape

        field_feature = self.field_conv(xyz_sdf)
        field_feature = field_feature.permute(0, 2, 1)

        l1_xyz, l1_points = self.sa1(field_feature[:, :3, :], field_feature[:, 3:, :])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(batch_size, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return {"pred_label": x, }


def test():
    """The test function just for the network shape checking.
    """

    torch.manual_seed(0)
    dim_b = 4
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
