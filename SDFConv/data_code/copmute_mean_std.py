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

# sparse1 version
test_acc = [
    76.9231, 71.7949, 78.2051, 82.0513, 75.6410,
    76.9231, 76.9231, 79.4872, 74.3590, 79.4872
]
print("For sparse 1:")
print(np.mean(test_acc))
print(np.std(test_acc))

# pointnet nl
test_acc = [
    60.5634,
    64.7887,
    66.1972,
    66.1972,
    61.9718,
    66.1972,
    63.3803,
    60.5634,
    60.5634,
    57.7465,
]
print("For pointnet nl 1:")
print(np.mean(test_acc))
print(np.std(test_acc))

# sparse 1 nl
test_acc = [
    64.7887,
    66.1972,
    57.7465,
    66.1972,
    60.5634,
    64.7887,
    64.7887,
    64.7887,
    57.7465,
    57.7465,
]
print("For sparse nl 1:")
print(np.mean(test_acc))
print(np.std(test_acc))

# pointnet mci
test_acc = [
    57.1429,
    61.4286,
    50.0000,
    55.7143,
    52.8571,
    61.4286,
    64.2857,
    61.4286,
    65.7143,
    65.7143,
]
print("For pointnet mci:")
print(np.mean(test_acc))
print(np.std(test_acc))
