"""The test function.
----ZhangsihaoYang,Jan,24,2021
"""
import argparse
import sys

from options import update_options, options, reset_options
from logger import parse_logger


def parse_args() -> argparse.Namespace:
    """parse the arguments.

    Returns:
        args: the arguments.
    """
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()

    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    return args


def main():
    """main function.
    """
    args = parse_args()

    logger, writer = reset_options(options, args, phase="test")

    parse_logger(options)


if __name__ == '__main__':
    main()
