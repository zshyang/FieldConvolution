## test.py

The test function to evaluate a checkpoint file. 

The plot from the test.py function.

![](../image/test_acc.png)

By running test.py with options in experiments/baseline_pointnet2_test.yml

We get validation accuracy 80.3279, while test accuracy just 62.8205.

So still a long way to go.

![](../image/test_acc_wo_da.png)

Test accuracy without rotation and jitter.

### The table of the PointNet++ baseline. 
 
Train data processing               | validation    | test 
---                                 | ---           | --- 
unit-scale & data augmentation (da) | 80.3279       | 62.8285
unit-scale & no da                  | 85.2459       | 75.6410
constant-scale & da                 | 78.6885       | 71.7949

