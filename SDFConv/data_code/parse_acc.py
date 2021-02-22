import os
from parse import *
import numpy as np


version_name = "sparse2"
# stage_name = "nl"
stage_name = "mci"


def parse_test_log(filename):
    with open(filename, "r") as file:
        for line in file:
            if "Validation acc" in line:
                r = parse("{} Validation acc: {:f}", line)
                validation_acc = r[1]
            if ("acc" in line) and ("Validation acc" not in line):
                r = parse("{} Test {}: {:f}", line)
                test_acc = r[2]
    return validation_acc, test_acc


def main():
    log_file_path = os.path.join("../data/save_ckpt_log_summary", version_name)

    validation_acc_list = []
    test_acc_list = []
    for i in range(10):
        test_log_path = os.path.join(
            log_file_path, "{}_{}{:02d}".format("shapecad", stage_name, i),
            "log", "{}{:02d}_test.log".format(stage_name, i)
        )
        validation_acc, test_acc = parse_test_log(test_log_path)
        validation_acc_list.append(validation_acc)
        test_acc_list.append(test_acc)

    print("{} {}".format(stage_name, version_name))
    for validation_acc, test_acc in zip(validation_acc_list, test_acc_list):
        print("{:.4f} / {:.4f}".format(validation_acc * 100, test_acc * 100))

    print(
        "{:.4f} +- {:.4f}".format(
            np.mean(test_acc_list) * 100, (np.std(test_acc_list)) * 100
        )
    )


if __name__ == '__main__':
    main()
