# WangChanXNER

Welcome to WangChanXNER! This repository contains a basic implementation for sequence labeling tasks. It provides functionalities for training and testing.

## Train/Test
To train and test the model, you can use the following commands:

### Training
```
python train.py --device 0 -c pc/config_lst20.json
```
- The `--device` flag specifies the GPU device to use during training. In this example, GPU 0 will be used.
- The `-c` flag points to the configuration file `config_lst20.json`, which holds the hyperparameters and settings for training.

### Testing
```
python test.py --resume [PATH]/checkpoint.pth
```
- The `--resume` flag points to the saved checkpoint file to load the model for testing.

## License
This project is licensed under CC-BY-SA 3.0. You can find more information about the license in the LICENSE file.

## Acknowledgements
This project is inspired by [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) created by [Mahmoud Gemy](https://github.com/MrGemy95). We express our gratitude to Mahmoud Gemy for providing the foundation and ideas for this project.