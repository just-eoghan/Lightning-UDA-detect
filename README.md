# Lightning-UDA-detect
A collection of popular unsupervised domain adaptation architectures. Made easy to install and use for research.


Click the Weights & Biases logo to see logs from our runs.

<a href="https://wandb.ai/eoghan/Lightning-UDA-detect"><img alt="Weights and Biases" src="https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black"></a>

# Getting started

## Complete Installation Steps

Install a relevant miniconda https://docs.conda.io/projects/miniconda/en/latest/

### Linux ~ Ubuntu 20.04

```
$ conda create --name Lightning-UDA-detect python=3.8
$ conda activate Lightning-UDA-detect
$ pip install pycocotools==2.0.6
$ pip install -r requirements.txt
```

### Windows 11

Using the Anaconda Prompt Terminal (Search Anaconda on Windows Search)

```
$ conda create --name Lightning-UDA-detect python=3.8
$ conda activate Lightning-UDA-detect
$ pip install cython
$ pip install pycocotools-windows==2.0
$ pip install -r requirements.txt
```

### MacOS (Apple Silicon)

```
$ conda create -n Lightning-UDA-detect
$ conda activate Lightning-UDA-detect
$ conda config --env --set subdir osx-64
$ conda install python=3.8
$ pip install pycocotools==2.0.6
$ pip install -r requirements.txt
```

Creating the conda environment ensures all packages are installed in an isolated virtual environment.

All of the required dependencies are stored within the requirements.txt file.

## Data Download

> **Note**
> You will need to create an account with cityscapes to access this data.

Go to https://www.cityscapes-dataset.com/downloads/ and download the following folders;

1. leftImg8bit_trainvaltest.zip https://www.cityscapes-dataset.com/file-handling/?packageID=3
2. leftImg8bit_trainvaltest_foggy.zip https://www.cityscapes-dataset.com/file-handling/?packageID=29


## Data Preprocessing

1. Extract the leftimg8bit folder from leftImg8bit_trainvaltest.zip into data/cityscapes/images

```
unzip leftImg8bit_trainvaltest.zip -d ./data/cityscapes/images/
```

2. Extract the leftimg8bit_foggy folder from leftImg8bit_trainvaltest_foggy.zip into data/foggy_cityscapes/images

```
unzip leftImg8bit_trainvaltest_foggy.zip -d ./data/foggy_cityscapes/images/
```

> **Note**
> leftimg8bit_foggy contains 15000 images, this is because it includes 3 different levels of fog set by a beta parameter. The value of beta we use is 0.02. 
>
> We provide a script `preprocess/foggy_data_beta_0.02.py` for re-creating the folder with only the 5000 images files with the 0.02 beta value.

3. Run the preprocessing script to reduce the foggy dataset to only images with a beta of 0.02

```
python preprocess/foggy_data_beta_0.02.py
```

After following the previous steps your data directory should look like this;

```
data/
├── cityscapes
│   ├── annotations
│   └── images
│       └── leftImg8bit
│           ├── test
│           │   ├── berlin
│           │   ├── bielefeld
│           │   ├── ...
│           ├── train
│           │   ├── aachen
│           │   ├── bochum
│           │   ├── ...
│           └── val
│               ├── frankfurt
│               ├── lindau
│               └── ...
└── foggy_cityscapes
    ├── annotations
    └── images
        └── leftImg8bit
            ├── test
            │   ├── berlin
            │   ├── bielefeld
            │   ├── ...
            ├── train
            │   ├── aachen
            │   ├── bochum
            │   ├── ...
            └── val
                ├── frankfurt
                ├── lindau
                └── ...
```

4. Verify there are 5000 image files in each leftImg8bit folder.

```
find ./data/cityscapes/images/leftImg8bit/ -type f | wc -l
```  
for original images

```
find ./data/foggy_cityscapes/images/leftImg8bit/ -type f | wc -l
```  
for foggy_images

## Weights and Biases

You can automatically log runs to your W&B account by editing configs/logger/wandb

Set `entity` to the name of your W&B account.
Set `project` to the name of the project you have created in W&B.

## Running Experiments

You can run models easily through the following command line calls

> **Note**
> If you don't want to use W&B just add logger=csv to the end of your run command

```
python run.py experiment=mic_da_cityscapes logger=csv
```

#### MIC

```
python run.py experiment=mic_da_cityscapes
```

#### SADA

```
python run.py experiment=scale_aware_da_cityscapes
```

#### DA

```
python run.py experiment=original_da_cityscapes
```

> **Note**
> You can edit the parameters of these experiments by going to configs/experiment and editing the YAML files.

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
