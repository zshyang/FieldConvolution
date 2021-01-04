import os
import json
import argparse
import numpy as np
from glob import glob


np.random.seed(4124036635)

META_FLDR = "../data/meta"
SDF_FLDR = "../data/sdf/"


def parse_name(name: str) -> list:
    """Parse the name with structure like, "../data/sdf/NL_neg/007_S_1206_I326626_RHippo_60k.npz"

    Args:
        name: The input string.

    Returns:
        The output list.
    """


def generate_meta(stage: str):
    """Generate the meta information for the very first beginning.

    Args:
        stage: The stage of the brain.
    """

    name_list = glob(os.path.join(SDF_FLDR, stage, "*.npz"))

    with open(os.path.join(META_FLDR, stage + ".json"), "w") as file:
        for name in name_list:
            print(name)


def main():
    stage_list = ["AD_pos", "NL_neg"]
    for stage in stage_list:
        generate_meta(stage)

    # # Get the two dataset name.
    # parser = argparse.ArgumentParser(description="Data preparation.")
    # parser.add_argument("--data_first", help="The first dataset name.", type=str, required=True)
    # parser.add_argument("--data_second", help="The second dataset name.", type=str, required=True)
    # args = parser.parse_args()
    #
    # # Generate the meta information given the dataset name.
    # gen_meta(args.data_first, args.data_second)


if __name__ == '__main__':
    main()
