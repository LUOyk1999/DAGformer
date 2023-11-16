# Transformer over Directed Acyclic Graph (NeurIPS 2023) 

The repository implements the [Transformer over Directed Acyclic Graph](https://openreview.net/forum?id=g49s1N5nmO) (DAG transformer) in Pytorch Geometric.

## Installation
Tested with Python 3.7, PyTorch 1.13.1, and PyTorch Geometric 2.3.1.

The dependencies are managed by [conda]:

```
pip install -r requirements.txt
```

## Overview

* `./NA` Experiment code over the `NA` dataset. 

* `./ogbg-code2` Experiment code over the `ogbg-code2` data from OGB. 

* `./self-citation` Experiment code over the `self-citation` dataset.

* `./Node_classification_citation` Experiment code over the `Cora, Citeseer, Pubmed` datasets.

## Reference

If you find our codes useful, please consider citing our work

```
@inproceedings{
luo2023transformers,
title={Transformers over Directed Acyclic Graphs},
author={Yuankai Luo and Veronika Thost and Lei Shi},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=g49s1N5nmO}
}
```

## Poster

![DAG_poster](https://raw.githubusercontent.com/LUOyk1999/images/main/images/DAG_poster.png)
