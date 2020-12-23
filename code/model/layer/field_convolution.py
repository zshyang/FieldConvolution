import numpy as np
import torch
from data_code.visualize_sdf import load_sdf


def numpy_tensor(numpy_array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to a tensor.

    Args:
        numpy_array: The numpy array.

    Returns:
        torch_tensor: The torch tensor.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_tensor = torch.from_numpy(numpy_array).to(device)

    return torch_tensor


def make_batch(points: [np.ndarray], sdfs: [np.ndarray]) -> {str: torch.Tensor}:
    numbers = [point.shape[0] for point in points]
    min_number = min(numbers)

    batch_size = len(points)

    sampled_points = []
    sampled_sdfs = []
    for i in range(batch_size):
        index = np.random.randint(low=0, high=numbers[i], size=min_number)

        sampled_points.append(points[i][index])
        sampled_sdfs.append(sdfs[i][index])

    # Load on to gpu.
    batch_points = numpy_tensor(np.stack(sampled_points))
    batch_sdfs = numpy_tensor(np.stack(sampled_sdfs))

    return {
        "points": batch_points, "sdfs": batch_sdfs,
    }





def test():
    # Load the sdf.
    point_0, sdf_0 = load_sdf("../../../data/141_S_1255_I297902_RHippo_60k.npz")
    point_1, sdf_1 = load_sdf("../../../data/1a6f615e8b1b5ae4dbbc9440457e303e.npz")

    # Make the sdf a batch.
    batch = make_batch([point_0, point_1], [sdf_0, sdf_1])

    # Forward the batch.


    pass


if __name__ == '__main__':
    test()
