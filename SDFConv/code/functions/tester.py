from logging import Logger

import torch
from torch.utils.data import DataLoader

from functions.base import CheckpointRunner
from functions.helper import pick_vis_func, pick_model, pick_loss
from utils.average_meter import AverageMeter
from pytorch3d.structures import Meshes


class Tester(CheckpointRunner):
    """The tester.
    """
    def __init__(self, options, logger: Logger, writer, shared_model=None):
        super().__init__(options, logger, writer, training="test", shared_model=shared_model)

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
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

        # Create loss function.
        self.criterion = pick_loss(self.options)

        # Evaluate step count, useful in summary
        self.evaluate_step_count = 0
        self.total_step_count = 0

    def models_dict(self):
        return {'model': self.model}

    def evaluate_loss(self, out_batch, input_batch):
        loss, loss_summary = self.criterion(out_batch, input_batch)
        self.loss_meter.update(loss.cpu().numpy())
        self.acc_meter.update(loss_summary["acc"].cpu().numpy())
        return loss_summary

    def evaluate_step(self, input_batch):
        self.model.eval()

        # Run inference
        with torch.no_grad():
            out = self.model(input_batch)

            loss_summary = self.evaluate_loss(out, input_batch)

        return out, loss_summary

    # noinspection PyAttributeOutsideInit
    def evaluate(self):
        self.logger.info("Running test ...")
        # clear evaluate_step_count, but keep total step count uncleared.
        self.evaluate_step_count = 0
        # Test data loader.
        test_data_loader = DataLoader(
            self.dataset,
            batch_size=self.options.test.batch_size * self.options.num_gpus,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory,
            shuffle=self.options.test.shuffle,
            collate_fn=self.dataset_collate_fn
        )
        # Set up average meter.
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()

        # Iterate over all batches in an epoch
        for step, batch in enumerate(test_data_loader):

            # add later to log at step 0
            self.evaluate_step_count += 1
            self.total_step_count += 1

            # Send input to GPU
            batch = {k: v.cuda() if isinstance(v, (torch.Tensor, Meshes)) else v for k, v in batch.items()}
            # Run evaluation step
            out = self.evaluate_step(batch)
            # Tensorboard logging every summary_steps steps
            if self.evaluate_step_count % self.options.test.summary_steps == 0:
                self.evaluate_summaries(batch, out)

            # Kaichun web page visualizer every kc_steps.
            if self.evaluate_step_count % self.options.test.kc_steps == 0:
                self.logger.info("Visualizing ...")
                self.visualizer(
                    batch, out, training=False, epoch=self.epoch_count,
                    step=self.evaluate_step_count, options=self.options, logger=self.logger
                )

        scalar = self.loss_meter.avg
        key = "loss"
        self.logger.info("Test [{:06d}] {}: {:.6f}".format(self.total_step_count, key, scalar))
        self.summary_writer.add_scalar("eval_" + key, scalar, self.total_step_count + 1)
        scalar = self.acc_meter.avg
        key = "acc"
        self.logger.info("Test [{:06d}] {}: {:.6f}".format(self.total_step_count, key, scalar))
        self.summary_writer.add_scalar("eval_" + key, scalar, self.total_step_count + 1)

    def evaluate_summaries(self, input_batch, out_summary):
        self.logger.info(
            "Test Step {:06d} / {:06d} ({:06d}): {:.6f} ACC {:.6f}".format(
                self.evaluate_step_count,
                len(self.dataset) // (self.options.num_gpus * self.options.test.batch_size),
                self.total_step_count,
                self.loss_meter.val,
                self.acc_meter.val,
            )
        )





