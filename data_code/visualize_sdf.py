import numpy as np


def load_sdf(file_name: str) -> (np.ndarray, np.ndarray):
    """Load the signed distance file.

    Args:
        file_name: The name of the file.

    Returns:
        points: The locations. (N, 3)
        sdf: The distance. (N, 1)
    """

    data = np.load(file_name, mmap_mode="r")

    point_sfd = np.concatenate((data["pos"], data["neg"]), axis=0)

    points = point_sfd[:, :3]
    sdf = point_sfd[:, [-1]]

    return points, sdf


def visualize_sdf():
    pass


def main():
    output = load_sdf("../data/141_S_1255_I297902_RHippo_60k.npz")
    print(type(output))

    points, sdf = load_sdf("../data/141_S_1255_I297902_RHippo_60k.npz")




if __name__ == '__main__':
    main()
