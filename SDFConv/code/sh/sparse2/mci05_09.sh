CUDA_VISIBLE_DEVICES=1 python train.py --options experiments/sparse2/mci/fold05.yml
CUDA_VISIBLE_DEVICES=1 python test.py  --options experiments/sparse2/mci/fold05_t.yml
CUDA_VISIBLE_DEVICES=1 python train.py --options experiments/sparse2/mci/fold06.yml
CUDA_VISIBLE_DEVICES=1 python test.py  --options experiments/sparse2/mci/fold06_t.yml
CUDA_VISIBLE_DEVICES=1 python train.py --options experiments/sparse2/mci/fold07.yml
CUDA_VISIBLE_DEVICES=1 python test.py  --options experiments/sparse2/mci/fold07_t.yml
CUDA_VISIBLE_DEVICES=1 python train.py --options experiments/sparse2/mci/fold08.yml
CUDA_VISIBLE_DEVICES=1 python test.py  --options experiments/sparse2/mci/fold08_t.yml
CUDA_VISIBLE_DEVICES=1 python train.py --options experiments/sparse2/mci/fold09.yml
CUDA_VISIBLE_DEVICES=1 python test.py  --options experiments/sparse2/mci/fold09_t.yml

