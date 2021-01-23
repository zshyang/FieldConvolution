"""This file is not needed to be changed anymore.
----ZY.2020.Oct
"""
from easydict import EasyDict
import importlib
from importlib import import_module


def pick_dataset(config, dataset: EasyDict, training: bool):
    """Pick the dataset to be used given the name.
    :param config:
    :param dataset:
    :param training:
    :return: A dataset is returned.
    """
    importlib.invalidate_caches()
    # Load dataset.
    dataset_class = getattr(import_module("datasets.{}".format(dataset.name[0])), "{}".format(dataset.name[1]))
    return dataset_class(config, dataset, training)


def pick_model(options: EasyDict):
    """Pick the model to be used given options.
    :param options:
    :return:
    """
    importlib.invalidate_caches()
    # Load model class.
    model_class = getattr(
        import_module("models.{}".format(options.model.name[0])), "{}".format(options.model.name[1])
    )
    return model_class(options)


def pick_loss(options: EasyDict):
    """Pick the loss function.

    :param options:
    :return:
    """
    importlib.invalidate_caches()
    # Load model class.
    loss_class = getattr(
        import_module("models.losses.{}".format(options.loss.name[0])), "{}".format(options.loss.name[1])
    )
    return loss_class(options)
    # raise NotImplementedError("Your loss is not found")


def pick_vis_func(options: EasyDict):
    """Pick the function to visualize one batch.

    :param options:
    :return:
    """
    importlib.invalidate_caches()
    vis_func = getattr(
        import_module("utils.vis.{}".format(options.vis.name[0])),
        "{}".format(options.vis.name[1])
    )
    return vis_func
