CUDA_VISIBLE_DEVICES=0 python train.py --options experiments/sparse2/mci/fold00.yml
CUDA_VISIBLE_DEVICES=0 python test.py  --options experiments/sparse2/mci/fold00_t.yml
CUDA_VISIBLE_DEVICES=0 python train.py --options experiments/sparse2/mci/fold01.yml
CUDA_VISIBLE_DEVICES=0 python test.py  --options experiments/sparse2/mci/fold01_t.yml
CUDA_VISIBLE_DEVICES=0 python train.py --options experiments/sparse2/mci/fold02.yml
CUDA_VISIBLE_DEVICES=0 python test.py  --options experiments/sparse2/mci/fold02_t.yml
CUDA_VISIBLE_DEVICES=0 python train.py --options experiments/sparse2/mci/fold03.yml
CUDA_VISIBLE_DEVICES=0 python test.py  --options experiments/sparse2/mci/fold03_t.yml
CUDA_VISIBLE_DEVICES=0 python train.py --options experiments/sparse2/mci/fold04.yml
CUDA_VISIBLE_DEVICES=0 python test.py  --options experiments/sparse2/mci/fold04_t.yml

