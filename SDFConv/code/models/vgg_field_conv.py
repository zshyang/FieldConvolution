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
from models.layers.sdf_relu import FieldReLU


class Net(nn.Module):
    """The PointNet++ classification model.
    """
    def __init__(self, options: EasyDict):
        """The initialization function.

        Args:
            options: The options to define the model.
        """
        super(Net, self).__init__()

        self.base_channel = 16  # Hard coded here.

        image_size = 56.0
        cube_size = 2.0

        # the vgg network
        self.relu = FieldReLU()
        self.sdf_conv_1 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=1,
            out_channels=self.base_channel,
            feature_is_sdf=True,
        )
        self.sdf_conv_2 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=self.base_channel,
            out_channels=self.base_channel,
            feature_is_sdf=False,
        )

        image_size = image_size / 2.0
        self.sdf_max_pooling_1 = FieldPooling(center_number=int(image_size * image_size))
        self.sdf_conv_3 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=self.base_channel,
            out_channels=(self.base_channel * 2),
            feature_is_sdf=False,
        )
        self.sdf_conv_4 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=(self.base_channel * 2),
            out_channels=(self.base_channel * 2),
            feature_is_sdf=False,
        )

        image_size = image_size / 2.0
        self.sdf_max_pooling_2 = FieldPooling(center_number=int(image_size * image_size))
        self.sdf_conv_5 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=(self.base_channel * 2),
            out_channels=(self.base_channel * 4),
            feature_is_sdf=False,
        )
        self.sdf_conv_6 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=(self.base_channel * 4),
            out_channels=(self.base_channel * 4),
            feature_is_sdf=False,
        )
        self.sdf_conv_7 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=(self.base_channel * 4),
            out_channels=(self.base_channel * 4),
            feature_is_sdf=False,
        )

        image_size = image_size / 2.0
        self.sdf_max_pooling_3 = FieldPooling(center_number=int(image_size * image_size))
        self.sdf_conv_8 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=(self.base_channel * 4),
            out_channels=(self.base_channel * 8),
            feature_is_sdf=False,
        )
        self.sdf_conv_9 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=(self.base_channel * 8),
            out_channels=(self.base_channel * 8),
            feature_is_sdf=False,
        )
        self.sdf_conv_10 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=(self.base_channel * 8),
            out_channels=(self.base_channel * 8),
            feature_is_sdf=False,
        )

        image_size = image_size / 2.0
        self.sdf_max_pooling_4 = FieldPooling(center_number=int(image_size * image_size))
        self.sdf_conv_11 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=(self.base_channel * 8),
            out_channels=(self.base_channel * 8),
            feature_is_sdf=False,
        )
        self.sdf_conv_12 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=(self.base_channel * 8),
            out_channels=(self.base_channel * 8),
            feature_is_sdf=False,
        )
        self.sdf_conv_13 = FieldConv(
            edge_length=(cube_size / image_size * 3.0),
            filter_sample_number=(3 * 3), center_number=int(image_size * image_size),
            in_channels=(self.base_channel * 8),
            out_channels=(self.base_channel * 8),
            feature_is_sdf=False,
        )

        image_size = image_size / 2.0
        self.sdf_max_pooling_5 = FieldPooling(center_number=int(image_size * image_size))
        self.sdf_conv_14 = FieldConv(
            edge_length=float(cube_size),
            filter_sample_number=int(image_size), center_number=int(1),
            in_channels=(self.base_channel * 8),
            out_channels=(self.base_channel * 16),
            feature_is_sdf=False,
        )

        self.num_class = options.model.out_channel

        self.fc1 = nn.Linear(self.base_channel * 16 + 3, 512)
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

        out = self.sdf_conv_1(xyz_sdf)
        out = self.relu(out)

        out = self.sdf_conv_2(out)
        out = self.relu(out)

        out = self.sdf_max_pooling_1(out)
        out = self.sdf_conv_3(out)
        out = self.relu(out)

        out = self.sdf_conv_4(out)
        out = self.relu(out)

        out = self.sdf_max_pooling_2(out)
        out = self.sdf_conv_5(out)
        out = self.relu(out)

        out = self.sdf_conv_6(out)
        out = self.relu(out)

        out = self.sdf_conv_7(out)
        out = self.relu(out)

        out = self.sdf_max_pooling_3(out)
        out = self.sdf_conv_8(out)
        out = self.relu(out)

        out = self.sdf_conv_9(out)
        out = self.relu(out)

        out = self.sdf_conv_10(out)
        out = self.relu(out)

        out = self.sdf_max_pooling_4(out)
        out = self.sdf_conv_11(out)
        out = self.relu(out)

        out = self.sdf_conv_12(out)
        out = self.relu(out)

        out = self.sdf_conv_13(out)
        out = self.relu(out)

        out = self.sdf_max_pooling_5(out)

        out = self.sdf_conv_14(out)
        out = self.relu(out)

        x = out.view(batch_size, self.base_channel * 16 + 3)
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
