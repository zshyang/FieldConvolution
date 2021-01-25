"""The test function.
----ZhangsihaoYang,Jan,24,2021
"""
import argparse
import sys

from options import update_options, options, reset_options


def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()

    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    # parser.add_argument('--batch-size', help='batch size', type=int)
    # parser.add_argument('--shuffle', help='shuffle samples', default=False, action='store_true')
    # parser.add_argument('--checkpoint', help='trained checkpoint file', type=str, required=True)
    # parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    # parser.add_argument('--name', help='subfolder name of this experiment', required=True, type=str)
    # parser.add_argument('--gpus', help='number of GPUs to use', type=int)
    #
    # args = parser.parse_args()
    #
    return args


def main():
    args = parse_args()
    print(args)
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(options)


if __name__ == '__main__':
    main()
