CUDA_VISIBLE_DEVICES=3 python train.py --options experiments/sparse2widedeep/mci/fold05.yml
CUDA_VISIBLE_DEVICES=3 python test.py  --options experiments/sparse2widedeep/mci/fold05_t.yml
CUDA_VISIBLE_DEVICES=3 python train.py --options experiments/sparse2widedeep/mci/fold06.yml
CUDA_VISIBLE_DEVICES=3 python test.py  --options experiments/sparse2widedeep/mci/fold06_t.yml
CUDA_VISIBLE_DEVICES=3 python train.py --options experiments/sparse2widedeep/mci/fold07.yml
CUDA_VISIBLE_DEVICES=3 python test.py  --options experiments/sparse2widedeep/mci/fold07_t.yml
CUDA_VISIBLE_DEVICES=3 python train.py --options experiments/sparse2widedeep/mci/fold08.yml
CUDA_VISIBLE_DEVICES=3 python test.py  --options experiments/sparse2widedeep/mci/fold08_t.yml
CUDA_VISIBLE_DEVICES=3 python train.py --options experiments/sparse2widedeep/mci/fold09.yml
CUDA_VISIBLE_DEVICES=3 python test.py  --options experiments/sparse2widedeep/mci/fold09_t.yml

