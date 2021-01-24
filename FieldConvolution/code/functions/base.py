"""This is a template for future use. Basically, you do not need to modify anything in this file.
----ZY.2020.Oct
"""
import os
import torch
from logging import Logger
from tensorboardX import SummaryWriter
from functions.helper import pick_dataset
import config
import time
from functions.saver import CheckpointSaver
from datetime import timedelta


class CheckpointRunner(object):

    def __init__(self, options, logger: Logger, summary_writer: SummaryWriter, training: str, shared_model=None):
        """Do the initialization.

        Args:
            options:
            logger:
            summary_writer:
            training: the stage of the training.
            shared_model: In the case of evaluation or prediction, we do not need create a new model.
                A shared model could be used to do evaluation.
        """

        self.options = options
        self.logger = logger
        self.summary_writer = summary_writer
        self.training = training

        # GPUs
        if not torch.cuda.is_available() and self.options.num_gpus > 0:
            raise ValueError("CUDA not found yet number of GPUs is set to be greater than 0")
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            logger.info("CUDA visible devices is activated here, number of GPU setting is not working")
            self.gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            self.options.num_gpus = len(self.gpus)
            enumerate_gpus = list(range(self.options.num_gpus))
            logger.info(
                "CUDA is asking for " + str(self.gpus) +
                ", PyTorch to doing a mapping, changing it to " + str(enumerate_gpus)
            )
        else:
            self.gpus = list(range(self.options.num_gpus))
            logger.info("Using GPUs: " + str(self.gpus))

        # Initialize dataset.
        dataset = options.dataset
        self.dataset = self.load_dataset(dataset, training)
        if training == "train":  # Update the number of training samples in options.
            self.options.dataset.len_train = len(self.dataset)
        elif training == "val":  # Update the number of validation samples in options.
            self.options.dataset.len_val = len(self.dataset)
        elif training == "test":
            self.options.dataset.len_test = len(self.dataset)
        else:
            raise ValueError("This stage {} is not known!".format(training))
        self.dataset_collate_fn = self.dataset.collate

        # By default, epoch_count = step_count = 0.
        self.epoch_count = 0
        self.step_count = 0
        self.time_start = time.time()

        # Override this function to define your model, optimizers etc.
        # In case you want to use a model that is defined in a trainer or other place in the code,
        # shared_model should help. in this case, checkpoint is not used
        self.logger.info("Running model initialization...")
        self.init_fn(shared_model=shared_model)  # This function would also create the model.

        # In the case, no shared model is provided.
        if shared_model is None:
            # checkpoint is loaded if any
            self.saver = CheckpointSaver(
                self.logger, self.options, training
            )
            self.init_with_checkpoint()

    def load_dataset(self, dataset, training: str):
        """Load the dataset.

        Args:
            dataset
            training (str)
        """
        # Logging.
        self.logger.info("Loading datasets: {}".format(dataset.name[0]))
        # Load dataset.
        return pick_dataset(config, dataset, training)

    def init_fn(self, shared_model=None, **kwargs):
        raise NotImplementedError('You need to provide an _init_fn method')

    def models_dict(self):
        """Pack models in a dict - necessary for checkpoint save and load.
        """
        return None

    def optimizers_dict(self):
        """Pack optimizers in a dict - necessary for checkpoint save and load.
        NOTE: optimizers and models cannot have conflicting names.
        """
        return None

    def init_with_checkpoint(self):
        """Initialize the model with checkpoint.
        :return:
        """
        # Load checkpoint.
        checkpoint = self.saver.load_checkpoint()
        # No checkpoint to be loaded.
        if checkpoint is None:
            self.logger.info("Checkpoint not loaded")
        else:  # One checkpoint is found.
            # Load model.
            for model_name, model in self.models_dict().items():
                if model_name in checkpoint:
                    if isinstance(model, torch.nn.DataParallel):  # Multiple GPUs.
                        model.module.load_state_dict(checkpoint[model_name], strict=False)
                    else:  # Single GPU.
                        model.load_state_dict(checkpoint[model_name], strict=False)
            # Load optimizer and learning rate scheduler.
            if self.optimizers_dict() is not None:
                for optimizer_name, optimizer in self.optimizers_dict().items():
                    if optimizer_name in checkpoint:
                        optimizer.load_state_dict(checkpoint[optimizer_name])
            else:
                self.logger.warning("Optimizers not found in the runner, skipping...")
            # Load epoch.
            if "epoch" in checkpoint:
                self.epoch_count = checkpoint["epoch"]
            # Load step.
            if "total_step_count" in checkpoint:
                self.step_count = checkpoint["total_step_count"]

    def dump_checkpoint(self):
        """Save the checkpoint.
        """
        # Epoch and step information.
        checkpoint = {
            "epoch": self.epoch_count,
            "total_step_count": self.step_count
        }
        # Get model weights.
        for model_name, model in self.models_dict().items():
            if isinstance(model, torch.nn.DataParallel):
                checkpoint[model_name] = model.module.state_dict()
            else:
                checkpoint[model_name] = model.state_dict()
            for k, v in list(checkpoint[model_name].items()):
                if isinstance(v, torch.Tensor) and v.is_sparse:  # Remove sparse tensor.
                    checkpoint[model_name].pop(k)
        # Get optimizer information.
        if self.optimizers_dict() is not None:
            for optimizer_name, optimizer in self.optimizers_dict().items():
                checkpoint[optimizer_name] = optimizer.state_dict()
        self.saver.save_checkpoint(checkpoint, "{:06d}_{:06d}".format(self.step_count, self.epoch_count))

    @property
    def time_elapsed(self):
        return timedelta(seconds=time.time() - self.time_start)
