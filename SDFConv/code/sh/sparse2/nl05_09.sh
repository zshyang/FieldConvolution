CUDA_VISIBLE_DEVICES=3 python train.py --options experiments/sparse2/nl/fold05.yml
CUDA_VISIBLE_DEVICES=3 python test.py  --options experiments/sparse2/nl/fold05_t.yml
CUDA_VISIBLE_DEVICES=3 python train.py --options experiments/sparse2/nl/fold06.yml
CUDA_VISIBLE_DEVICES=3 python test.py  --options experiments/sparse2/nl/fold06_t.yml
CUDA_VISIBLE_DEVICES=3 python train.py --options experiments/sparse2/nl/fold07.yml
CUDA_VISIBLE_DEVICES=3 python test.py  --options experiments/sparse2/nl/fold07_t.yml
CUDA_VISIBLE_DEVICES=3 python train.py --options experiments/sparse2/nl/fold08.yml
CUDA_VISIBLE_DEVICES=3 python test.py  --options experiments/sparse2/nl/fold08_t.yml
CUDA_VISIBLE_DEVICES=3 python train.py --options experiments/sparse2/nl/fold09.yml
CUDA_VISIBLE_DEVICES=3 python test.py  --options experiments/sparse2/nl/fold09_t.yml
