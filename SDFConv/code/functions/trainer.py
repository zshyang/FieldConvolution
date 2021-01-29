"""Remember that epoch starts with 1. Step also starts at 1.
----ZhangsihaoYang.Nov.12.2020
"""
import torch
from torch.utils.data import DataLoader

from functions.base import CheckpointRunner
from functions.evaluator import Evaluator
from functions.helper import pick_model, pick_loss
from functions.helper import pick_vis_func
from utils.average_meter import AverageMeter
from utils.tensor import recursive_detach
from pytorch3d.structures import Meshes


class Trainer(CheckpointRunner):
    """Trainer.
    """
    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        """The initialization function for Trainer.

        :param shared_model: The model.
        :param kwargs: The input dictionary.
        """
        # Visualization.
        self.visualizer = pick_vis_func(self.options)

        # Build the model.
        if shared_model is not None:  # Use shared model.
            self.model = shared_model
        else:
            self.model = pick_model(self.options)
            if len(self.gpus) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()
            else:
                self.model = self.model.cuda()

        # Setup optimizer for the model.
        if self.options.optim.name == "adam":
            self.optimizer = torch.optim.Adam(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd
            )
        elif self.options.optim.name == "sgd":
            self.optimizer = torch.optim.SGD(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                momentum=self.options.optim.sgd_momentum,
                weight_decay=self.options.optim.wd
            )
        else:
            raise NotImplementedError("Your optimizer is not found")
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor
        )

        # Create loss function.
        self.criterion = pick_loss(self.options)

        # Create AverageMeters for losses
        self.losses = AverageMeter()
        self.acc = AverageMeter()

        # Evaluators
        self.evaluators = [Evaluator(self.options, self.logger, self.summary_writer, shared_model=self.model)]

    def models_dict(self) -> dict:
        return {"model": self.model}

    def optimizers_dict(self) -> dict:
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }

    def train_step(self, input_batch: dict):
        """Train one step. This function does not need to be changed in future.

        Args:
            input_batch:

        Returns:

        """
        # Enable grad.
        self.model.train()

        # predict with model
        out = self.model(input_batch)

        # Compute loss.
        loss, loss_summary = self.criterion(out, input_batch)

        # Update AverageMeter.
        self.losses.update(loss.detach().cpu().item())
        self.acc.update(loss_summary["acc"].cpu().numpy())

        # Do back propagation.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments to be used for visualization
        return recursive_detach(out), recursive_detach(loss_summary)

    def train(self):
        # Run training for num_epochs epochs.
        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            self.epoch_count += 1

            # Create a new data loader for every epoch
            train_data_loader = DataLoader(
                self.dataset,
                batch_size=self.options.train.batch_size * self.options.num_gpus,
                num_workers=self.options.num_workers,
                pin_memory=self.options.pin_memory,
                shuffle=self.options.train.shuffle,
                collate_fn=self.dataset_collate_fn
            )

            # Reset AverageMeter
            self.losses.reset()

            # Iterate over all batches in an epoch
            for step, batch in enumerate(train_data_loader):

                # Send input to GPU
                batch = {
                    k: v.cuda() if isinstance(v, (torch.Tensor, Meshes)) else v for k, v in batch.items()
                }

                # Run training step.
                out = self.train_step(batch)

                self.step_count += 1

                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.train.summary_steps == 0:
                    self.train_summaries(batch, *out)

                # Kaichun web page visualizer every kc_steps.
                if self.step_count % self.options.train.kc_steps == 0:
                    # self.logger.info("Visualizing ...")
                    self.visualizer(
                        batch, out, training=True, epoch=self.epoch_count,
                        step=self.step_count, options=self.options, logger=self.logger
                    )

                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.options.train.checkpoint_steps == 0:
                    self.dump_checkpoint()

            # Save checkpoint after each epoch.
            self.dump_checkpoint()

            # Run validation every test_epochs.
            if self.epoch_count % self.options.train.test_epochs == 0:
                self.test()

            # lr scheduler step
            self.lr_scheduler.step()

    def train_summaries(self, input_batch, out_summary, loss_summary):

        # Save each loss items in Tensorboard.
        for k, v in loss_summary.items():
            self.summary_writer.add_scalar(k, v.mean(), self.step_count)

        # Save info to log.
        self.logger.info(
            "Epoch {:03d}, Step {:06d}/{:06d}, Time elapsed {}, Loss {:12.9f} (average {:12.9f}) ACC {:.6f} ".format(
                self.epoch_count, self.step_count,
                self.options.train.num_epochs * len(self.dataset) // (
                    self.options.train.batch_size * self.options.num_gpus
                ), self.time_elapsed,
                self.losses.val, self.losses.avg,
                self.acc.val,
            )
        )

    def test(self):
        """Do evaluation.
        """
        for evaluator in self.evaluators:
            evaluator.evaluate()
