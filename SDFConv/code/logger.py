import logging
import os
from easydict import EasyDict
import numpy as np
from glob import glob


def create_logger(cfg, phase="train"):

    log_file = "{}_{}.log".format(cfg.version, phase)
    final_log_file = os.path.join(cfg.log_dir, log_file)
    head = "%(asctime)-15s %(message)s"

    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    if cfg.log_level == "info":
        logger.setLevel(logging.INFO)
    elif cfg.log_level == "debug":
        logger.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError("Log level has to be one of info and debug")
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    return logger


def parse_logger(options: EasyDict):
    """parse the logger to get the validation epoch with the maximum accuracy.
    The path to the checkpoint file is updated.

    Args:
        options: a global variable.
    """
    # the phase of the logger file.
    phase = "train"

    # get the name of the logger file.
    log_file = "{}_{}.log".format(options.version, phase)
    final_log_file = os.path.join(options.log_dir, log_file)

    # get the number of epoch each training epoch.
    number_step_each_epoch = []
    with open(final_log_file, "r") as file:
        for line in file:
            if "Epoch 001," in line:
                number_step_each_epoch.append(line)
    number_step_each_epoch = len(number_step_each_epoch)

    # load the logger file.
    val_acc = []
    with open(final_log_file, "r") as file:
        for line in file:
            if ("Test" in line) and ("acc" in line):
                val_acc.append(line)

    # parse the accuracy
    val_acc_list = []
    for acc_line in val_acc:
        acc = acc_line[-9:-1]
        val_acc_list.append(float(acc))

    # get the step, epoch, and the checkpoint file name
    reverse_list = np.array(val_acc_list)[::-1]
    epoch = len(val_acc_list) - np.argmax(reverse_list)
    step = number_step_each_epoch * epoch
    checkpoint_file = "{:06d}_{:06d}.pt".format(step, epoch)

    # remove the useless checkpoint files
    remove_ckpt = True
    if remove_ckpt:
        ckpt_files = glob(options.checkpoint_dir + "/*.pt")
        for ckpt_file in ckpt_files:
            file_end = ckpt_file.split("/")[-1]
            if file_end != checkpoint_file:
                os.remove(ckpt_file)

    print(val_acc_list[epoch - 1])

    # optionally plot the accuracy
    visualization = False
    if visualization:
        import matplotlib.pyplot as plt
        plt.plot(val_acc_list)
        plt.ylabel("the accuracy")
        plt.show()
        print(val_acc_list[epoch - 1])

    # update the options.
    options.checkpoint_file = checkpoint_file
