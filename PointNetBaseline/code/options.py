""" Create a global variable options.
And everything about options will be defined in this file.
New option items could be defined in the yaml file.

functions: (These functions could be shared across projects)
    _update_dict()
    _update_options()
    update_options()
"""
import os
import pprint
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter

from logger import create_logger

# Global parameters.
options = EasyDict()
options.checkpoint_dir = None
options.checkpoint_file = None
options.log_level = "info"
options.log_dir = None
options.name = "3d_detail_unet"
options.num_gpus = 0
options.num_workers = 0
options.overwrite = True
options.save_dir = None
options.seed = 4124036635
options.summary_dir = None
options.version = None
options.pin_memory = True
options.latest_checkpoint = False
options.load_epoch = None
options.load_step = None

# Dataset
options.dataset = EasyDict()
options.dataset.name = None
options.dataset.resize = 0
options.dataset.train_fn = []

# Model
options.model = EasyDict()
options.model.bilinear_up = False
options.model.in_channel = 2
options.model.name = []
options.model.out_channel = 2

# Loss
options.loss = EasyDict()
options.loss.name = None

# Training.
options.train = EasyDict()
options.train.num_epochs = 5
options.train.batch_size = 2
options.train.num_workers = 6
options.train.shuffle = True

# Test.
options.test = EasyDict()
options.test.batch_size = 2

# Optimization.
options.optim = EasyDict()
options.optim.name = "adam"
options.optim.adam_beta1 = 0.9
options.optim.sgd_momentum = 0.9
options.optim.lr = 1.0E-1
options.optim.wd = 1.0E-6
options.optim.lr_step = [30, 45]
options.optim.lr_factor = 0.1

# Visualization.
options.vis = EasyDict()


def _update_dict(full_key, val, d):
    """Update dictionary given a value. The dictionary is a global variable.
    Only the key within d will be updated.

    Args:
        full_key (str): The position of the key.
        val (dict): The values used to update the dictionary.
        d (EasyDict): The dictionary to be updated.
    """
    for vk, vv in val.items():
        # The key of value is not in d.
        # if vk not in d:
        #     # Exit.
        #     raise ValueError("{}.{} does not exist in options".format(full_key, vk))
        # else:  # The key of val is in d.
        if isinstance(vv, list):  # The value of the key is list.
            d[vk] = np.array(vv)  # Store it as a numpy array.
        elif isinstance(vv, dict):  # The value of the key is dictionary.
            _update_dict(full_key + "." + vk, vv, d[vk])  # Call the function again.
        else:  # At the leaf of the dictionary.
            d[vk] = vv


def _update_options(options_file):
    with open(options_file) as file:
        options_dict = yaml.safe_load(file)  # dict
        if "based_on" in options_dict:
            # Go through all the defined based on yaml files.
            for base_options_file in options_dict["based_on"]:
                options_file_dir = os.path.dirname(options_file)
                base_options_file_path = os.path.join(options_file_dir, base_options_file)
                # The options is updated implicitly at this step.
                _update_options(base_options_file_path)
            # Remove "based_on" key in the options_dict.
            options_dict.pop("based_on")

        # Update options given options_dict get from yaml files.
        _update_dict("", options_dict, options)


def update_options(options_file):
    _update_options(options_file)


def slugify(filename):
    """Slugify the filename and create a prefix.

    Args:
        filename (str): The path to the yaml file.
    """
    filename = os.path.relpath(filename, "")  # The relative path from current folder to the yaml file.
    # Remove "experiments" in filename.
    if filename.startswith("experiments/"):
        filename = filename[len("experiments/"):]
    return os.path.splitext(filename)[0].lower().replace("/", "_").replace(".", "_")


def overwrite_folder(folder_path, overwrite):
    """Remove folder if exists.

    Args:
        folder_path (str)
        overwrite (bool)
    """
    if os.path.exists(folder_path):
        if overwrite:
            shutil.rmtree(folder_path)


def reset_options(input_options, args, phase="train"):
    """Reset all the options and create needed folder to save training process.

    Args:
        input_options (easydict.EasyDict)
        args (argparse.Namespace)
        phase (str)
    Returns:
        logger (logging.RootLogger)
        writer (tensorboardX.writer.SummaryWriter)
    """
    # Overwrite arguments in options.
    if hasattr(args, "batch_size") and args.batch_size:
        input_options.train.batch_size = args.batch_size
        input_options.test.batch_size = args.batch_size
    if hasattr(args, "checkpoint") and args.checkpoint:
        options.checkpoint = args.checkpoint
    if hasattr(args, "name") and args.name:
        options.name = args.name
    if hasattr(args, "num_epochs") and args.num_epochs:
        options.train.num_epochs = args.num_epochs
    if hasattr(args, "version") and args.version:
        options.version = args.version

    # Set up the model version is not provided. Please provide it when training.
    if options.version is None:
        prefix = ""
        if args.options:
            prefix = slugify(args.options) + "_"
        options.version = prefix + datetime.now().strftime("%m%d%H%M%S")  # ignore %Y

    # Create folders.
    name_version = options.name + "_" + options.version
    # Log folder.
    options.log_dir = os.path.join(options.save_dir, name_version, options.log_dir)
    print("=> creating {}".format(options.log_dir))
    if phase == "train":
        overwrite_folder(options.log_dir, options.overwrite)
    os.makedirs(options.log_dir, exist_ok=True)

    # Checkpoint folder.
    options.checkpoint_dir = os.path.join(options.save_dir, name_version, options.checkpoint_dir)
    print("=> creating {}".format(options.checkpoint_dir))
    if phase == "train":
        overwrite_folder(options.checkpoint_dir, options.overwrite)
    os.makedirs(options.checkpoint_dir, exist_ok=True)

    # Summary folder.
    options.summary_dir = os.path.join(options.save_dir, name_version, options.summary_dir)
    print("=> creating {}".format(options.summary_dir))
    if phase == "train":
        overwrite_folder(options.summary_dir, options.overwrite)
    os.makedirs(options.summary_dir, exist_ok=True)

    # Visualization folder.
    options.vis.dir = os.path.join(options.save_dir, name_version, options.vis.dir)
    print("=> creating {}".format(options.vis.dir))
    if phase == "train":
        overwrite_folder(options.vis.dir, options.overwrite)
    os.makedirs(options.vis.dir, exist_ok=True)

    logger = create_logger(options, phase=phase)
    options_text = pprint.pformat(vars(options))  # str
    logger.info(options_text)

    writer = SummaryWriter(options.summary_dir)

    # Control randomness
    if options.seed < 0:
        options.seed = random.randint(1, 10000)
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)

    return logger, writer


if __name__ == "__main__":
    update_options("experiments/default.yml")

