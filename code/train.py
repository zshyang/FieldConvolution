"""Most part of this file does not need to be changed.
----ZY.Oct.2020
"""
import argparse
import sys
from options import update_options, reset_options, options
from functions.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--options", help="Experiment options file name", required=False, type=str)

    args, rest = parser.parse_known_args()

    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    # training
    parser.add_argument("--batch-size", help="batch size", type=int)
    parser.add_argument("--checkpoint", help="checkpoint file", type=str)
    parser.add_argument("--name", required=False, type=str)  # The required is changed from True to False.
    parser.add_argument("--num-epochs", help="number of epochs", type=int)
    parser.add_argument("--version", help="version of task (timestamp by default)", type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logger, writer = reset_options(options, args)

    trainer = Trainer(options, logger, writer, training=True)
    trainer.train()


if __name__ == "__main__":
    main()
