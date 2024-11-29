# SchoolMap: Mapping of school yard in Norway :school:

This repository contains the code necessary to train and run DeepLabV3 on satellite pictures. In particular, for our project we were interested in segmenting school yard in Norway.

## Install the required libraries

```bash
pip install poetry
poetry install
```

## Changing the CONFIG file

The `configs` folder contains a list of the config files necessary to use this repository's pipeline. The only config file you need to modify is the `/configs/paths/default.yaml`, which contains all the paths the pipeline is using.

## How to Use

1. Extract the Satellite Images

Use the script to download and prepare the satellite images used for training and testing. The exact source and format of the images should be configured in the config.yaml file.

```bash

poetry run python3 src/dataset/get_train_images.py
```

2. Rasterize the Masks

This step converts GeoJSON or vector data into raster format, creating segmentation masks that correspond to your satellite images. These masks are required for training the model.

```bash

poetry run python3 src/dataset/rasterize_masks.py
```

3. Train the Model

Once the dataset (images and masks) is ready, train the segmentation model using [DeepLabV3](https://arxiv.org/abs/1706.05587v3) by running the following:

```bash

poetry run python3 src/train.py
```

The training script supports early stopping, model checkpoints, and tracks the loss function and evaluation metrics (IoU, Precision, Recall, DICE, Pixel accurracy).

4. Predict New Areas

Once the model is trained, you can use it to generate predictions for new images. The following script takes an input image and outputs the corresponding segmentation mask:

```bash
poetry run python3 src/predict.py cfg.paths.INPUT_PATH=<input_image_path>
```

Instead of only predicting a single image at a time, it is also possible to predict on a whole folder using the following command:

```bash
find /PATH/TO/IMAGE/FOLDER -name "*.png" | xargs -I {} poetry run python src/predict.py paths.INPUT_IMAGE={}
```

5. Compute Evaluation Metrics

To evaluate the model’s performance on the test dataset, run the metrics script. This script computes common segmentation metrics, such as precision, recall, Dice coefficient, and IoU (Intersection over Union):

```bash

poetry run python3 src/metrics.py
```
