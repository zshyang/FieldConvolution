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

        # The first field convolution layer.
        self.field_conv = FieldConv(
            edge_length=0.03, filter_sample_number=64, center_number=16 ** 3, in_channels=1,
            out_channels=self.in_channel,
            feature_is_sdf=True,
        )

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

        # self.sparseModel = scn.Sequential().add(
        #     scn.SubmanifoldConvolution(3, 1, m, 3, False)).add(
        #     scn.UNet(3, block_reps, [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], residual_blocks)).add(
        #     scn.BatchNormReLU(m)).add(
        #     scn.OutputLayer(3))
        self.sparseModel = scn.SparseVggNet(3, 1, [
            ['C', 16], ['C', 16], 'MP',
            ['C', 32], ['C', 32], 'MP',
            ['C', 48], ['C', 48], 'MP',
            ['C', 64], ['C', 64], 'MP',
            ['C', 96], ['C', 96]]
        ).add(
            scn.Convolution(3, 96, 128, 3, 2, False)
        ).add(
            scn.BatchNormReLU(128)
        ).add(scn.SparseToDense(3, 128))
        # print(self.sparseModel)
        # self.spatial_size = self.sparseModel.input_spatial_size(
        #     torch.LongTensor([512, 512, 512])
        # )
        # self.inputLayer = scn.InputLayer(2, self.spatial_size, 2)
        self.linear1 = nn.Linear(128 * 3 * 3 * 3, 128)
        self.linear2 = nn.Linear(128, 2)

        self.input_layer = scn.InputLayer(3, 127, mode=4)
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
        locations = locations * 128
        locations = locations - locations.min()
        locations = locations.long()

        locations = torch.cat([locations.view(-1, 3), batch_index.view(-1, 1)], dim=-1)
        locations = locations.long()

        features = xyz_sdf[:, :, 3:]
        features = features.view(-1, 1)

        print(locations.shape, features.shape)

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
        x = x.view(-1, 128 * 3 * 3 * 3)
        x = self.linear1(x)
        x = self.linear2(x)
        # print(x.shape)

        return {"pred_label": x, }


def test():
    """The test function just for the network shape checking.
    """
    import torch
    import sparseconvnet as scn

    # Use the GPU if there is one, otherwise CPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = scn.Sequential().add(
        scn.SparseVggNet(2, 1,
                         [['C', 8], ['C', 8], ['MP', 3, 2],
                          ['C', 16], ['C', 16], ['MP', 3, 2],
                          ['C', 24], ['C', 24], ['MP', 3, 2]])
    ).add(
        scn.SubmanifoldConvolution(2, 24, 32, 3, False)
    ).add(
        scn.BatchNormReLU(32)
    ).add(
        scn.SparseToDense(2, 32)
    ).to(device)

    # output will be 10x10
    inputSpatialSize = model.input_spatial_size(torch.LongTensor([10, 10]))
    input_layer = scn.InputLayer(2, inputSpatialSize)

    msgs = [[" X   X  XXX  X    X    XX     X       X   XX   XXX   X    XXX   ",
             " X   X  X    X    X   X  X    X       X  X  X  X  X  X    X  X  ",
             " XXXXX  XX   X    X   X  X    X   X   X  X  X  XXX   X    X   X ",
             " X   X  X    X    X   X  X     X X X X   X  X  X  X  X    X  X  ",
             " X   X  XXX  XXX  XXX  XX       X   X     XX   X  X  XXX  XXX   "],

            [" XXX              XXXXX      x   x     x  xxxxx  xxx ",
             " X  X  X   XXX       X       x   x x   x  x     x  x ",
             " XXX                X        x   xxxx  x  xxxx   xxx ",
             " X     X   XXX       X       x     x   x      x    x ",
             " X     X          XXXX   x   x     x   x  xxxx     x ", ]]

    # Create Nx3 and Nx1 vectors to encode the messages above:
    locations = []
    features = []
    for batchIdx, msg in enumerate(msgs):
        for y, line in enumerate(msg):
            for x, c in enumerate(line):
                if c == 'X':
                    locations.append([y, x, batchIdx])
                    features.append([1])
    locations = torch.LongTensor(locations)
    features = torch.FloatTensor(features).to(device)

    input = input_layer([locations, features])
    print('Input SparseConvNetTensor:', input)
    print(locations.shape)
    print(features.shape)
    output = model(input)

    # Output is 2x32x10x10: our minibatch has 2 samples, the network has 32 output
    # feature planes, and 10x10 is the spatial size of the output.
    # print('Output SparseConvNetTensor:', output)


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
