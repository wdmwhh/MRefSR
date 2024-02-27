# LMR: A Large-Scale Multi-Reference Dataset for Reference-based Super-Resolution

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>


[[Paper](https://arxiv.org/pdf/2303.04970)]
[[ArXiv](https://arxiv.org/abs/2303.04970)]
[[Dataset Website](https://academictorrents.com/details/39424bb06d9172ac1c50fe4426eca51697bb4bfc)]

## Dependency

### Environment
This code is based on PyTorch.

By using Anaconda, you can setup the environment simply by

```
conda env create -f environment.yml
```

## Training

### Train on the proposed LMR
```
./scripts/dist_train.sh 4 options/train/stage3_5ref_restoration_mse.yml
```
The results will be saved in ./experiments

## Inference

### Test on CUFED5 and the proposed LMR
```
python mmsr/test.py -opt test.yml
```
The results will be saved in ./result

## Contact
Feel free to contact me if there is any question. (Lin Zhang,  zhanglin2019@ia.ac.cn)

