# imputed-diffusion

## Installation

### Requirements

```
numpy
scikit-learn
torch
tqdm
```



## Data

```
python preprocess_daphnet.py
```

## Training

```
python main.py -c configs/config_daphnet.json
```

## Results

| Dataset | Method             | Sensitivity | Specificity | Accuracy |
|:-------:|-------------------|:-----------:|:-----------:|:--------:|
| Daphnet | Ashour et al. 2020 | - | - | 83.83 |
| Daphnet | Xia et al. 2018    | 90.60 | 69.29 | 79.95 |
| Daphnet | Li et al. 2020     | - | - | 91.90 |
| Daphnet | Noor et al. 2021   | 90.94 | 67.04 | 78.99 |
| Daphnet | This repository    | 82.24 | 97.51 | 92.39 |  |

Direct comparison between previous works and the method in this repository is not plausible because the evaluation method differs. For details, check [this document.](data/daphnet/README.md)