# WangChanXNER

Welcome to WangChanXNER! This repository contains a basic implementation for sequence labeling tasks. It provides functionalities for training and testing.

## Installation

To install the necessary dependencies, run the following:
python=3.8.16

```bash
seqeval
pythainlp
tabulate
pandas==2.0.3
torch==2.0.1
numpy==1.23.5
python=3.8.16
protobuf==3.20.0
transformers==4.29.2
```

## Downloading the Checkpoint

Test the model with [Colab](https://colab.research.google.com/drive/1XL8U6_u5bIIFNiyqfg9RMJcG__3kXtGi?usp=sharing).

You can download the checkpoint from our google drive: [checkpoint](https://drive.google.com/drive/folders/1u-Auuo7cph4I1d78ezzM3mW4oieZlZuU?usp=share_link).

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
python main.py --device 0 -c storage/config_base.json
```

- The `--device` flag specifies the GPU device to use during training. In this example, GPU 0 will be used.
- The `-c` flag points to the configuration file `config_lst20.json`, which holds the hyperparameters and settings for training.


### Pretrained

```
xlm-roberta-base
airesearch/wangchanberta-base-att-spm-uncased
```

### Testing

To test the model, use the `inference.py` script:

```
python inference.py --resume storage/best_model/model_best.pth
```

- The `--resume` flag points to the saved checkpoint file to load the model for testing.


## Acknowledgements

This project is inspired by [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) created by [Mahmoud Gemy](https://github.com/MrGemy95). We express our gratitude to Mahmoud Gemy for providing the foundation and ideas for this project.
