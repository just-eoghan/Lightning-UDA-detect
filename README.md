# Lightning-UDA-detect
A collection of popular unsupervised domain adaptation architechtures. Made easy to install and use for research.


Click the Weights & Biases logo to see logs from our runs.

<a href="https://wandb.ai/eoghan/Lightning-UDA-detect"><img alt="Weights and Biases" src="https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black"></a>

# Getting started

## Environment & Package Install

```
$ conda create --name Lightning-UDA-detect python=3.8
$ conda activate Lightning-UDA-detect
$ pip install -r requirements.txt
```

## Data Download

> **Note**
> You will need to create an account with cityscapes to access this data.

Go to https://www.cityscapes-dataset.com/downloads/ and download the following folders;

1. leftImg8bit_trainvaltest.zip https://www.cityscapes-dataset.com/file-handling/?packageID=3
2. leftImg8bit_trainvaltest_foggy.zip https://www.cityscapes-dataset.com/file-handling/?packageID=29

The zips contain the following files;

```
leftImg8bit
    val
        munster
        lindau
        frankfurt
        ...
    train
        aachen
        bochum
        bremen
        ...
    test
        berlin
        bielefeld
        bonn
        ...
```

Extract the leftimg8bit folder from leftImg8bit_trainvaltest.zip into data/cityscapes/images

Extract the leftimg8bit folder from leftImg8bit_trainvaltest_foggy.zip into data/foggy_cityscapes/images

## Weights and Biases

You can automatically log runs to your W&B account by editing configs/logger/wandb

Set `entity` to the name of your W&B account.
Set `project` to the name of the project you have created in W&B.

## Running Experiments

You can run models easily through the following command line calls

If you don't want to use W&B just add logger=csv to the end of your run command e.g.
`python run.py experiment=x logger=csv`

MIC
`python run.py experiment=mic_da_cityscapes`

SADA
`python run.py experiment=scale_aware_da_cityscapes`

DA
`python run.py experiment=original_da_cityscapes`

Edit these experiments by going to configs/experiment and editing the YAML files.

# References

## Papers

- https://ieeexplore.ieee.org/document/8578450/
- https://link.springer.com/10.1007/s11263-021-01447-x
- https://arxiv.org/abs/2212.01322

## Implementations

- https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch
- https://github.com/yuhuayc/sa-da-faster
- https://github.com/lhoyer/MIC

# Experiment Results

![Mean Average Precision \@50.](/paper/lit-uda-map50.png)

![Max Mean Average Precision \@50.](/paper/map50-bar-plot.png)