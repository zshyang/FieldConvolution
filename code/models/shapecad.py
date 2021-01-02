"""Vanilla model is
LRFGConv(3, 32) -> GConv(32, 128) -> MaxPool -> FC(128, 64, 1)
----Zhangsihao.Yang.Nov.11.2020
"""
import torch
import torch.nn as nn
from easydict import EasyDict
# from models.layers.unet_layers import DoubleConv, Down, Up, OutConv
from models.layers.lrfgconv import LRFGraphConv
from pytorch3d.ops import GraphConv
from torch.nn import functional as F
from pytorch3d.structures.utils import packed_to_list


class Net(nn.Module):
    """The vanilla version the network.
    """
    def __init__(self, options: EasyDict):
        """Initializetion.

        Args:
            options: The option.
        """
        super(Net, self).__init__()
        self.n_channels = options.model.in_channel
        self.n_classes = options.model.out_channel

        self.conv1 = LRFGraphConv(input_dim=self.n_channels, output_dim=32)
        self.conv2 = GraphConv(input_dim=32, output_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=self.n_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the classifier weights.
        """
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch: dict):
        """The forward function.

        Args:
            batch: The batch from data loader.

        Returns:
        """
        # The first LRF conv layer.
        out1 = F.relu(
            self.conv1(
                verts=batch["mesh"].verts_packed(),
                edges=batch["mesh"].edges_packed(),
                lrf=batch["lrf"]
            )
        )  # (V, 32)
        # The second graph conv layer.
        out2 = F.relu(self.conv2(verts=out1, edges=batch["mesh"].edges_packed()))  # (V, 128)

        # Do the max-pooling on each batch.
        max_feature = []
        out2_list = packed_to_list(x=out2, split_size=batch["vert_num_list"])
        for out2_item in out2_list:
            out2_max, _ = torch.max(out2_item, dim=0, keepdim=True)
            max_feature.append(out2_max)
        max_feature = torch.cat(max_feature, dim=0)

        # The classifier layers.
        logits = self.classifier(max_feature)

        return {"pred_label": logits}






