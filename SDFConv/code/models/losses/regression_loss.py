import torch.nn as nn
from torch import Tensor
import torch


class MSELoss(nn.MSELoss):
    def __init__(self, options):
        super().__init__(reduction="none")

    def forward(self, out_batch: dict, input_batch: dict) -> tuple:
        loss = super(MSELoss, self).forward(input=out_batch["pred_error_image"], target=input_batch["error_image"])
        return loss.mean(), {
            "loss": torch.mean(loss, (1, 2, 3)),
        }

