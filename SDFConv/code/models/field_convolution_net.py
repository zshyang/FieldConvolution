"""The model that utilize field convolution.
----Zhangsihao.Yang.Jan.02.2021
"""
import torch
import torch.nn as nn
from easydict import EasyDict
from models.layers.field_convolution import FieldConv
# A random change


class Net(nn.Module):
    """The network.
    """
    def __init__(self, options:  EasyDict):
        """The initialization function.

        Args:
            options: The options.
        """
        super(Net, self).__init__()

        self.conv_1 = FieldConv(
            edge_length=0.01, filter_sample_number=32, center_number=4096, in_channels=1, out_channels=32,
            feature_is_sdf=True,
        )



    def forward(self, batch: dict) -> dict:
        """The forward function.

        Args:
            batch: The input batch.

        Returns:

        """

        out = self.conv_1(batch["points_sdf"])
        print(out.shape)


def test():
    """The test function.
    """

    torch.manual_seed(4124036635)

    batch_size = 4
    number_points = 10000
    batch = {
        "points_sdf": torch.randn(batch_size, number_points, 4),
    }

    options = EasyDict()

    net = Net(options)

    net(batch)


# import torch
# import torch.nn as nn
# from easydict import EasyDict
# from pytorch3d.ops import GraphConv
# from pytorch3d.structures import Meshes
# from pytorch3d.structures import padded_to_list, list_to_packed
# from pytorch3d.structures.utils import packed_to_list
# from torch.nn import functional as F
#
# from models.layers.graph_kernel_conv import GraphKernelConv
#
#
# class ResidualModule(nn.Module):
#     def __init__(
#         self, dim_in: int, dim_out: int, k: int, layer_type: str, r: float
#     ):
#         super(ResidualModule, self).__init__()
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         self.filter_1 = {
#             "feature_dim": self.dim_in,
#             "feature_input": ["fea", "xyz", "n1", "n2", "geo"],
#             "k": k,
#             "layer_type": layer_type,
#             "out_dim": dim_out,
#             "perception_field": r,
#             "weight_input": ["fea", "xyz", "n1", "n2", "geo"],
#         }
#         self.filter_2 = {
#             "feature_dim": self.dim_out,
#             "feature_input": ["fea", "xyz", "n1", "n2", "geo"],
#             "k": k,
#             "layer_type": layer_type,
#             "out_dim": dim_out,
#             "perception_field": r,
#             "weight_input": ["fea", "xyz", "n1", "n2", "geo"],
#         }
#         self.gkc_1 = GraphKernelConv(filter=self.filter_1)
#         self.gkc_2 = GraphKernelConv(filter=self.filter_2)
#         self.bn_1 = nn.BatchNorm1d(dim_out)
#         self.bn_2 = nn.BatchNorm1d(dim_out)
#         if dim_in != dim_out:
#             self.lin = nn.Linear(in_features=dim_in, out_features=dim_out)
#
#     def forward(self, inputs: dict):
#         """The forward function.
#
#         Args:
#             inputs: The dictionary contains all the input.
#                 "dist_map": The geodesic distance map. (B, N, N)
#                 "feature": The input feature. (B, N, F)
#                 "lrf": The local reference frame. (B, N, 3, 3)
#                 "normal": The normal direction. (B, N, 3)
#                 "position": The position. (B, N, 3)
#         Returns:
#             conv2: The output feature. (B, N, O)
#                 O: stands for the output dimension.
#         """
#         # Save the input feature.
#         fea_in = inputs["fea"]
#
#         # The first convolution layer.
#         conv_1 = self.gkc_1(inputs)
#         conv_1 = F.relu(conv_1)
#         conv_1 = self.bn_1(
#             conv_1.view(-1, conv_1.shape[-1])
#         ).view(conv_1.shape)
#
#         # The second convolution layer.
#         inputs["fea"] = conv_1
#         conv_2 = self.gkc_2(inputs)
#         conv_2 = F.relu(conv_2)
#         conv_2 = self.bn_2(
#             conv_2.view(-1, conv_2.shape[-1])
#         ).view(conv_2.shape)
#
#         # addition part
#         if self.dim_in == self.dim_out:
#             conv_2 = conv_2 + fea_in
#         else:
#             conv_2 = conv_2 + self.lin(fea_in)
#
#         return conv_2
#
#
# class Net(nn.Module):
#     """The network with continous convolution layer.
#     """
#     def __init__(self, options: EasyDict):
#         """Initialization.
#
#         Args:
#             options: The option.
#                 model:
#                     base_dim: The base dimension.
#                     base_radius: The base radius.
#
#         """
#         super(Net, self).__init__()
#         self.base_dim = options.model.base_dim
#         self.base_radius = options.model.base_radius
#         self.filter_1 = {
#             "feature_input": ["xyz", "n1", "n2", "geo"],
#             "k": 25,
#             "layer_type": "0",
#             "out_dim": self.base_dim,
#             "perception_field": self.base_radius,
#             "weight_input": ["xyz", "n1", "n2", "geo"],
#         }
#         self.gkc_1 = GraphKernelConv(filter=self.filter_1)
#         self.bn_1 = nn.BatchNorm1d(self.filter_1["out_dim"])
#
#         # Hard code layer type.
#         lt = "0"
#         self.res_1 = ResidualModule(
#             dim_in=self.filter_1["out_dim"], dim_out=self.base_dim,
#             k=5, layer_type=lt, r=self.base_radius*2,
#         )
#
#         self.res_2_1 = ResidualModule(
#             dim_in=self.base_dim, dim_out=self.base_dim*2,
#             k=5, layer_type=lt, r=self.base_radius*4
#         )
#         self.res_2_2 = ResidualModule(
#             dim_in=self.base_dim*2, dim_out=self.base_dim*2,
#             k=5, layer_type=lt, r=self.base_radius*4
#         )
#
#         self.res_3 = ResidualModule(
#             dim_in=self.base_dim*2, dim_out=self.base_dim * 4,
#             k=5, layer_type=lt, r=self.base_radius * 8
#         )
#
#         self.res_4_1 = ResidualModule(
#             dim_in=self.base_dim*4, dim_out=self.base_dim*8,
#             k=5, layer_type=lt, r=self.base_radius*16
#         )
#         self.res_4_2 = ResidualModule(
#             dim_in=self.base_dim*8, dim_out=self.base_dim*8,
#             k=5, layer_type=lt, r=self.base_radius*16
#         )
#
#         self.n_classes = options.model.out_channel
#
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=self.base_dim*8, out_features=self.base_dim*4),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(in_features=self.base_dim*4, out_features=self.n_classes),
#         )
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         """Initialize the classifier weights.
#         """
#         for m in self.classifier:
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, batch: dict):
#         """The forward function.
#
#         Args:
#             batch: The batch from data loader.
#                 "padded_verts": tensor of vertices of shape (N, max(V_n), 3).
#                 "padded_faces": tensor of faces of shape (N, max(F_n), 3).
#                 "vert_num": tensor with shape (N) contains V_n.
#                 "face_num": tensor with shape (N) contains F_n.
#                 "lrf": tensor of local reference frame of shape (N, max(V_n), 3, 3).
#                 "label": tensor with shape (N) contains label,
#
#         Returns:
#             The dictionary of prediction.
#         """
#         inputs_1 = {
#             "dist_map": batch["dist_map"],
#             "fea": None,
#             "lrf": batch["lrf"],
#             "normal": batch["normal"],
#             "position": batch["padded_verts"],
#         }
#         conv_1 = self.gkc_1(inputs_1)
#         conv_1 = F.relu(conv_1)
#         conv_1 = self.bn_1(conv_1.view(-1, conv_1.shape[-1])).view(conv_1.shape)
#
#         # Block 2.
#         inputs_1["fea"] = conv_1
#         conv_2 = self.res_1(inputs_1)
#
#         # Block 3.
#         inputs_1["fea"] = conv_2  # (B, N, BD X 1)
#         conv_3_1 = self.res_2_1(inputs_1)
#         inputs_1["fea"] = conv_3_1
#         conv_3_2 = self.res_2_2(inputs_1)
#
#         # Block 4.
#         inputs_1["fea"] = conv_3_2
#         conv_4 = self.res_3(inputs_1)
#
#         # Block 5.
#         inputs_1["fea"] = conv_4
#         conv_5_1 = self.res_4_1(inputs_1)
#         inputs_1["fea"] = conv_5_1
#         conv_5_2 = self.res_4_2(inputs_1)  # (B, V, BD X 8)
#
#         # Classifier.
#         logits = self.classifier(conv_5_2)  # (B, V, C)
#         logits = torch.mean(logits, dim=-2)
#
#         return {"pred_label": logits}
#
#
# def test_0():
#     torch.manual_seed(0)
#     dim_b = 4
#     dim_n = 16
#     batch = {
#         "dist_map": torch.randn(dim_b, dim_n, dim_n),
#         "padded_verts": torch.randn(dim_b, dim_n, 3),
#         "lrf": torch.randn(dim_b, dim_n, 3, 3),
#         "label": torch.randn(dim_b),
#         "normal": torch.randn(dim_b, dim_n, 3),
#     }
#
#     options = EasyDict()
#     options.model = EasyDict()
#
#     options.model.base_dim = 4
#     options.model.base_radius = 0.05
#     options.model.out_channel = 2
#
#     gkcnet = Net(options)
#
#     out = gkcnet(batch)
#     print(out)


if __name__ == '__main__':
    test()


