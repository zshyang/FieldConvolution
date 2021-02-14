"""In this version, I just use SDF to pick the feature.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from models.pointnet_util import farthest_point_sample
from models.pointnet_util import index_points


class FieldBatchNormalization(nn.Module):
    """Field convolution layer.
    """
    def __init__(
        self, in_channel: int
    ):
        """The initialization function.

        Args:
            in_channel: The number of input channels.
        """

        super(FieldBatchNormalization, self).__init__()

        self.bn = nn.BatchNorm1d(in_channel)

        self.in_channel = in_channel

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
        feature = feature.permute(0, 2, 1)

        assert len(points.shape) == 3, "The input point cloud should be batched!"
        assert len(feature.shape) == 3, "The input signed distance field should be batched!"
        assert points.shape[2] == 3, "The input point cloud should be in 3D space!"

        bn_feature = self.bn(feature)
        bn_feature = bn_feature.permute(0, 2, 1)

        # Concatenate the new location and new feature.
        out_feature = torch.cat([points, bn_feature], -1)

        return out_feature
