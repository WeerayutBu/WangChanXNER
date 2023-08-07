# WangChanXNER

Welcome to WangChanXNER! This repository contains a basic implementation for sequence labeling tasks. It provides functionalities for training and testing.

## Installation

To install the necessary dependencies, run the following:
python=3.8.16

```bash
pip install pythainlp
pip install seqeval
pip install torch==2.0.1
pip install protobuf==3.20.0
pip install transformers==4.29.2
```

## Downloading the Checkpoint

You can download the checkpoint and unzip the storage file from [LINK](https://vistec-my.sharepoint.com/:f:/g/personal/weerayut_b_s20_vistec_ac_th/EhCu1EJLsZJEpAVmT3c6D3oBy0y7lb0CzBN-9xsutlzdJg?e=zrYwzK).

Once downloaded, extract the files to ensure the appropriate directory structure.

```bash
unzip downloaded_checkpoint.zip -d storage/
```

## Directory tree

```
.
├── utils
├── model
├── storage
├── trainer
├── inference.ipynb
└── inference.py
```

## Train/Test

### Training

Run the `main.py` script to train the model:

```
python main.py --device 0 -c storage/config_lst20.json
```

- The `--device` flag specifies the GPU device to use during training. In this example, GPU 0 will be used.
- The `-c` flag points to the configuration file `config_lst20.json`, which holds the hyperparameters and settings for training.

### Testing

To test the model, use the `inference.py` script:

```
python inference.py --resume based-lst20/0804_224007_970844/checkpoint.pth
```

- The `--resume` flag points to the saved checkpoint file to load the model for testing.


## Acknowledgements

This project is inspired by [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) created by [Mahmoud Gemy](https://github.com/MrGemy95). We express our gratitude to Mahmoud Gemy for providing the foundation and ideas for this project.
