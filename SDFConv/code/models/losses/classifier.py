import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, options):
        """The initialization function.

        Args:
             options: The options.
        """
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none").cuda()

    def forward(self, outputs: dict, input_batch: dict):
        """Forward function of loss function.

        Args:
            outputs: The output of batch.
                "pred_label": The predicted label with size B X C.
            input_batch: The input batch.
                "label": The label with size N.

        Returns:
            loss: The loss.
            And a dictionary of loss summary.
        """
        # The ground truth label.
        labels = input_batch["label"]
        # The batch loss.
        loss = self.cross_entropy(outputs["pred_label"], labels)  # (B)
        # The batch prediction index.
        _, predicted = torch.max(outputs["pred_label"].data, 1)  # (B)
        # Batch size.
        total = labels.shape[0]
        # Batch correct prediction.
        correct = (predicted == labels).float()  # (B)
        # The total number of correct.
        correct_ = torch.sum(correct)
        # Put the total on GPU.
        total = torch.ones_like(correct, dtype=torch.float32) * total
        # The mean loss.
        loss_ = torch.mean(loss)
        return loss_, {
            "loss": loss,
            "acc": correct_ / total,
            "correct": correct,
        }



