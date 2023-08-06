# WangChanXNER

Welcome to WangChanXNER! This repository contains a basic implementation for sequence labeling tasks. It provides functionalities for training and testing.

## Installation

To install the necessary dependencies, run the following:

```bash
pip install pythainlp seqeval torch==2.0.1 python=3.8.16 protobuf==3.20.0 transformers==4.29.2
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

## License

This project is licensed under CC-BY-SA 3.0. You can find more information about the license in the LICENSE file.

## Acknowledgements

This project is inspired by [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) created by [Mahmoud Gemy](https://github.com/MrGemy95). We express our gratitude to Mahmoud Gemy for providing the foundation and ideas for this project.