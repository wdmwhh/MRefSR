# LMR: A Large-Scale Multi-Reference Dataset for Reference-based Super-Resolution

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>


[[Paper (Coming soon)]()]
[[ArXiv (Coming soon)]()]
[[Dataset Website (Coming soon)]()]

## Dependency

### Environment
This code is based on PyTorch.

By using Anaconda, you can setup the environment simply by

```
conda env create -f environment.yml
```

## Inference

### Test on CUFED5 and the proposed LMR
```
python mmsr/test.py -opt test.yml
```
The results will be saved in ./result
