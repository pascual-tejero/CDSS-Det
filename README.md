# CDSS-Det
Official implementation of Cross-Domain Semi-Supervised Organ Detection, [MIDL 2026 submission](https://openreview.net/forum?id=NSjBDpsZqV#discussion)

## Description
CDSS-Det is a cross-domain semi-supervised framework for accurate 3D organ detection in CT scans. Built on the Transformer-based Organ-DETR model, it leverages labeled source data, a small set of labeled target samples, and unlabeled target data to improve detection across domains. By combining pseudo-labeling, curriculum learning, and domain adaptation, CDSS-Det effectively transfers knowledge from well-annotated sources to target domains with limited labels. Experiments show it significantly outperforms conventional semi-supervised and domain adaptation methods, achieving higher mean Average Precision (mAP) and improved detection of small organs, even surpassing fully supervised models trained solely on the target domain.


## Installation
```
conda create -n cdss-det python=3.8.18
conda activate cdss-det
conda install pytorch==2.3.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Configs
Config files are under config/aaai, adjust the dataset path according to your storage and other parameters as you needed.

## Datasets
Our preprocessed datasets can be downloaded from [here](https://pan.baidu.com/s/1OlRY_z7R4z2MbcMadYiEXQ).
We don't upload Abdomen-Atlas since it is too large, you can download it from the [official source](https://github.com/MrGiovanni/AbdomenAtlas) instead.

If you use your own dataset, make sure it is preprocessed in the same way as organdetr/data/preprocessor.py.


## Training

## Pre-training
First, pre-train the model using source data.
```
export ORGANDETR_DATA=${your_data_path}
python scripts/sd_train.py --config ${your_config_name}
```

## Cross-Domain training
Second, use the pre-trained model above to train the source, labeled target and unlabeled target data together.
```
export ORGANDETR_DATA=${your_data_path}
python scripts/aaai_ts_cd_train.py --config ${your_config_name}$
``` 

## Inference
```
python scripts/ts_test.py --run ${your_config_name} --model ${your_model_checkpoint_name}
```
