"""
----ZhangsihaoYang.Feb.14.2021
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from models.pointnet_util import farthest_point_sample
from models.pointnet_util import index_points
import torch.nn.functional as F


class FieldReLU(nn.Module):
    """Field convolution layer.
    """
    def __init__(
        self
    ):
        """The initialization function.

        Args:
            center_number: The number of convolution centers. Equivalent to the stride of the filter. Noted as Nc.
        """

        super(FieldReLU, self).__init__()

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

        feature = F.relu(feature)

        # Concatenate the new location and new feature.
        out_feature = torch.cat([points, feature], -1)

        return out_feature
