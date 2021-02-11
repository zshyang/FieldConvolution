"""In this version, I just use SDF to pick the feature.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from models.pointnet_util import farthest_point_sample
from models.pointnet_util import index_points


class FieldPooling(nn.Module):
    """Field convolution layer.
    """
    def __init__(
        self, center_number: int
    ):
        """The initialization function.

        Args:
            center_number: The number of convolution centers. Equivalent to the stride of the filter. Noted as Nc.
        """

        super(FieldPooling, self).__init__()

        self.center_number = center_number

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

        index = farthest_point_sample(points, self.center_number)
        fps_points = index_points(points, index)
        fps_feature = index_points(feature, index)

        # Concatenate the new location and new feature.
        out_feature = torch.cat([fps_points, fps_feature], -1)

        return out_feature
