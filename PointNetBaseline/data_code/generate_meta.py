import os
import json
import argparse
import numpy as np
from glob import glob


np.random.seed(4124036635)

META_FLDR = "../data/meta"
SDF_FLDR = "../data/sdf/"


def parse_name(name: str) -> [str]:
    """Parse the name with structure like, "../data/sdf/NL_neg/007_S_1206_I326626_RHippo_60k.npz"

    Args:
        name: The input string.

    Returns:
        The output list.
    """
    stage = name.split("/")[-2]
    index = name.split("/")[-1][:-4]
    return [stage, index]


def generate_meta(stage: str):
    """Generate the meta information for the very first beginning.

    Args:
        stage: The stage of the brain.
    """

    name_list = glob(os.path.join(SDF_FLDR, stage, "*.npz"))

    meta_list = []
    with open(os.path.join(META_FLDR, stage + ".json"), "w") as file:
        for name in name_list:
            meta_list.append(parse_name(name))

        json.dump(meta_list, file)


def split_train_test(data_name):
    with open(os.path.join(META_FLDR, data_name + ".json"), "r") as file:
        data_list = json.load(file)
    data_np = np.array(data_list)
    np.random.shuffle(data_np)
    if len(data_list) < 1:
        raise ValueError("The length of the dataset is wrong: {}".format(len(data_list)))
    split_point = int(0.8 * len(data_list))
    return data_np[:split_point].tolist(), data_np[split_point:].tolist()


def gen_meta(data_first, data_second):
    train_list_1, test_list_1 = split_train_test(data_first)
    train_list_2, test_list_2 = split_train_test(data_second)

    train_list_1.extend(train_list_2)
    test_list_1.extend(test_list_2)

    train_fn = data_first + "_" + data_second + "_" + "train.json"
    test_fn = data_first + "_" + data_second + "_" + "test.json"

    with open(os.path.join(META_FLDR, train_fn), "w") as file:
        json.dump(train_list_1, file)
    print("{} with {} train data is generated.".format(train_fn, len(train_list_1)))

    with open(os.path.join(META_FLDR, test_fn), "w") as file:
        json.dump(test_list_1, file)
    print("{} with {} test data is generated.".format(test_fn, len(test_list_1)))


def main():
    """Generate the separate meta file and split them to save.
    """

    stage_list = ["AD_pos", "NL_neg"]
    for stage in stage_list:
        generate_meta(stage)

    # Generate the meta information given the dataset name.
    gen_meta(stage_list[0], stage_list[1])


if __name__ == '__main__':
    main()
