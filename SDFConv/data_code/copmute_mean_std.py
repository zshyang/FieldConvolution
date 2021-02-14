import numpy as np

# for sdf conv
test_acc = [
    80.7692, 75.6410, 65.3846, 78.2051, 71.7949, 76.9231, 82.0513, 78.2051, 73.0769, 79.4872
]

print(np.mean(test_acc))
print(np.std(test_acc))

# for pointnet++
test_acc_pn2 = [
    78.2051, 80.7692, 69.2308, 76.9231, 70.5128, 80.7692, 79.4872, 80.7692, 71.7949, 76.9231
]
print("For pointnet++:")
print(np.mean(test_acc_pn2))
print(np.std(test_acc_pn2))

# conv1 version
test_acc = [
    83.3333, 76.9231, 67.9487, 71.7949, 70.5128,
    74.3590, 66.6667, 69.2308, 76.9231, 73.0769
]
print("For version 1:")
print(np.mean(test_acc))
print(np.std(test_acc))
