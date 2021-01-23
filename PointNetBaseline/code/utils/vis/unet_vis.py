"""Implement this function across different project.
----ZY.2020.Oct.
"""
import os
from easydict import EasyDict
from torchvision.utils import save_image
from logging import Logger
from subprocess import call


def create_save_folders(root_folder, folder_list: list):
    """Create folders to save visualization image.

    :param root_folder: The root folder.
    :param folder_list: The list of folders
    """
    for folder in folder_list:
        os.makedirs(os.path.join(root_folder, folder), exist_ok=True)


def unet_vis(
        in_batch: dict, out_batch: tuple, training: bool, epoch: int, step: int, options: EasyDict, logger: Logger
):
    """The visualization function of UNet.

    :param in_batch: The input batch.
    :param out_batch: The output batch.
    :param training: Whether it is training stage.
    :param epoch: The epoch number start with 1.
    :param step: The step.
    :param logger: The logger.
    :param options: The options for visualization.
    """
    # Folders
    if training:
        vis_dir = os.path.join(options.vis.dir, "train_vis")
    else:
        vis_dir = os.path.join(options.vis.dir, "val_vis")
    out_dir = os.path.join(vis_dir, "epoch-{:04d}".format(epoch))
    # Customize the list of folders.
    dir_list = ["input_image", "info"]
    # Create the list folders.
    create_save_folders(out_dir, dir_list)
    # The list of key in input and output batch.
    key_list = ["input_image", ["loss"]]
    batch = {}
    batch.update(in_batch)
    batch.update(out_batch[0])
    batch.update(out_batch[1])

    # Get the batch size.
    if training:
        batch_size = options.train.batch_size
    else:
        batch_size = options.test.batch_size

    # Get number of steps each epoch.
    if training:  # Update the number of training samples in options.
        num_step_each_epoch = options.dataset.len_train // (options.train.batch_size * options.num_gpus)
    else:  # Update the number of validation samples in options.
        num_step_each_epoch = options.dataset.len_test // (options.test.batch_size * options.num_gpus)

    # Save images and info.
    for i in range(batch_size):
        batch_id = step % num_step_each_epoch
        fn = "data-{:04d}.png".format(batch_id * batch_size + i)  # file name.
        for key, folder in zip(key_list, dir_list):
            if folder == "info":
                with open(os.path.join(out_dir, folder, fn.replace('.png', '.txt')), 'w') as file:
                    for loss_item in key:
                        file.write("{}: {}\n".format(loss_item, batch[loss_item][i].item()))
            else:
                save_image(batch[key][i], os.path.join(out_dir, folder, fn))

    # Get the KC step interval.
    if training:
        kc_steps = options.train.kc_steps
    else:
        kc_steps = options.test.kc_steps

    # Generate HTML file.
    mod_step = step % num_step_each_epoch  # step starts ar 1.
    extra_step = (mod_step + kc_steps) / num_step_each_epoch
    if mod_step == 0 or extra_step > 1.0:
        # Visualize HTML.
        logger.info("Generating html visualization ...")
        sublist = ",".join(dir_list)
        script_path = os.path.join(os.path.abspath(os.getcwd()), "utils", "gen_html_hierarchy_local.py")
        if not os.path.exists(script_path):
            raise ValueError("{} this python script does not exist!".format(script_path))
        cmd = "cd {} && python {} . 10 htmls {} {} > /dev/null".format(
            out_dir, script_path, sublist, sublist
        )
        call(cmd, shell=True)
        logger.info("DONE")
