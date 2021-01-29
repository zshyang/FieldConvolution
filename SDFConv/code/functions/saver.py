"""This is a template for checkpoint save and load. And you no longer to change anything inside this file.
----ZY.Oct.2020
"""
import os
import parse
from logging import Logger
from easydict import EasyDict

import torch
import torch.nn


class CheckpointSaver(object):
    """Class that handles saving and loading checkpoints during training."""
    def __init__(self, logger: Logger, options: EasyDict, training: str):
        """Initialization function.
        :param logger
        :param options
        """
        self.logger = logger
        self.options = options

        # Checkpoint is given in the options.
        if options.checkpoint_file is not None:
            local_ckpt_file = os.path.join(options.checkpoint_dir, options.checkpoint_file)
            # Do not exist.
            if not os.path.exists(local_ckpt_file):
                raise ValueError("Checkpoint file [{}] does not exist!".format(options.checkpoint_file))
            # Only accept valid checkpoint file for training.
            if training == "train" and not self.check_end_epoch(options.checkpoint_file):
                raise ValueError(
                    "{} is not at the end of epoch for training".format(
                        options.checkpoint_file
                    )
                )
            self.save_dir = os.path.dirname(os.path.abspath(local_ckpt_file))
            self.checkpoint_file = os.path.abspath(local_ckpt_file)
        else:  # Checkpoint is not given in the options.
            # checkpoint_dir is not provided.
            if options.checkpoint_dir is None:
                raise ValueError("Checkpoint directory must be not None in case file is not provided!")
            self.save_dir = os.path.abspath(options.checkpoint_dir)
            if options.latest_checkpoint or options.overwrite:
                self.checkpoint_file = self.get_latest_checkpoint()
                if training == "train" and self.checkpoint_file is not None:
                    raise ValueError(
                        "It is not recommended to train without know which epoch and step you start with!"
                    )
            else:  # Not load latest ckpt.
                self.checkpoint_file = os.path.abspath(
                    os.path.join(self.save_dir, "{:06d}_{:06d}.pt".format(options.load_step, options.load_epoch))
                )
                # Do not exist.
                if not os.path.exists(self.checkpoint_file):
                    raise ValueError("Checkpoint file [{}] does not exist!".format(self.checkpoint_file))
                # Not end of epoch and training.
                if training == "train" and not self.check_end_epoch(self.checkpoint_file):
                    raise ValueError("The step should be the end of one epoch")

    def check_end_epoch(self, fn: str) -> bool:
        """Check if the checkpoint file is the end of one epoch.

        :param fn:
        :return:
        """
        num_step_each_epoch = self.options.dataset.len_train // (self.options.train.batch_size * self.options.num_gpus)
        format_string = "{:06d}_{:06d}.pt"
        parsed = parse.parse(format_string, fn.split("/")[-1])
        ckpt_step = parsed[0]
        ckpt_epoch = parsed[1]
        return (ckpt_epoch * num_step_each_epoch) == ckpt_step

    def get_latest_checkpoint(self):
        """This will automatically find the checkpoint with latest modified time.
        And return the path to the latest checkpoint.
        """
        checkpoint_list = []
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename.endswith('.pt'):
                    file_path = os.path.abspath(os.path.join(dirpath, filename))
                    modified_time = os.path.getmtime(file_path)
                    checkpoint_list.append((file_path, modified_time))
        checkpoint_list = sorted(checkpoint_list, key=lambda x: x[1])
        if len(checkpoint_list) == 0:  # No checkpoint is found.
            return None
        else:  # Return the last modified checkpoint.
            return checkpoint_list[-1][0]

    def load_checkpoint(self):
        """Load checkpoint.
        """
        if self.checkpoint_file is None:
            self.logger.info("Checkpoint file not found, skipping...")
            return None
        else:
            self.logger.info("Loading checkpoint file: {}".format(self.checkpoint_file))
            try:
                return torch.load(self.checkpoint_file)
            except UnicodeDecodeError:
                # Compatible with old encoding methods.
                return torch.load(self.checkpoint_file, encoding="bytes")

    def save_checkpoint(self, obj: dict, name: str):
        """Save checkpoint.

        :param obj: The dictionary contains the network training information.
        :param name: The name for saving.
        """
        # checkpoint_file is updated at this step.
        self.checkpoint_file = os.path.join(self.save_dir, "{}.pt".format(name))
        self.logger.info("Dumping to checkpoint file: {}".format(self.checkpoint_file))
        torch.save(obj, self.checkpoint_file)
