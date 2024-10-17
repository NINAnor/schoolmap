# SchoolMap: Mapping of school yard in Norway :school:

This repository contains the code necessary to train and run DeepLabV3 on satellite pictures. In particular, for our project we were interested in segmenting school yard in Norway.

## How to Use

Make sure to adjust the parameters in the config.yaml file to fit your environment and dataset paths before proceeding with the steps below.

1. Extract the Satellite Images

Use the script to download and prepare the satellite images used for training and testing. The exact source and format of the images should be configured in the config.yaml file.

```bash

python3 src/dataset/get_train_images.py
```

2. Rasterize the Masks

This step converts GeoJSON or vector data into raster format, creating segmentation masks that correspond to your satellite images. These masks are required for training the model.

```bash

python3 src/dataset/rasterize_masks.py
```

3. Train the Model

Once the dataset (images and masks) is ready, train the segmentation model using [DeepLabV3](https://arxiv.org/abs/1706.05587v3) by running the following:

```bash

python3 src/train.py
```

The training script supports early stopping, model checkpoints, and tracks the loss function and evaluation metrics (IoU, Precision, Recall, DICE, Pixel accurracy).

4. Predict New Areas

Once the model is trained, you can use it to generate predictions for new images. The following script takes an input image and outputs the corresponding segmentation mask:

```bash

python3 src/predict.py <input_image_path>
```


Make sure to update paths for the model checkpoint and the directory for output masks in the config.yaml file.
5. Compute Evaluation Metrics

To evaluate the model’s performance on the test dataset, run the metrics script. This script computes common segmentation metrics, such as precision, recall, Dice coefficient, and IoU (Intersection over Union):

```bash

python3 src/metrics.py
```

## Config.yaml parameters

### DATASET Parameters

**MASK:**
Path to the GeoJSON file containing the polygon geometries (masks) for the segmentation task.

**MASKS_DIR:**
Directory where the rasterized masks will be saved after converting from vector format (GeoJSON) to raster format. These masks are used for training the model.

**IMG_DIR:**
Directory where the satellite images that correspond to the masks are stored. The images should match the IDs in the GeoJSON or mask file for effective training.

**WMS_URL:**
The URL of the Web Map Service (WMS) used for retrieving satellite images. This points to an external service that provides access to geographic data layers (in this case, Norwegian imagery).

**CRS:**
The Coordinate Reference System (CRS) used when extracting satellite images. "EPSG:4326" is a standard CRS, often used for global datasets, corresponding to WGS84, with latitude and longitude.

**SAVE_IMAGE:**
A boolean flag (True or False) to determine whether the satellite images should be saved locally after retrieval. If True, images are saved in the specified directory.

### TRAINING Parameters

**NUM_CLASSES:**
The number of classes in the segmentation task. This corresponds to the different areas you want to classify, such as "grass", "parking", "playground", etc. In this case, you have 8 distinct classes.

**NUM_EPOCHS:**
The maximum number of epochs to train the model. One epoch corresponds to one complete pass through the entire training dataset.

**BATCH_SIZE:**
The number of samples processed at once during training. A batch size of 8 means that the model will update its parameters every 8 images.

**LR:**
The learning rate for the optimizer. This controls how much the model's weights are updated with respect to the gradient during training. A small learning rate ensures gradual convergence.

**PATIENCE:**
The number of epochs with no improvement in validation loss before early stopping is triggered. This is used to prevent overfitting by stopping the training once the model stops improving.

### PREDICT Parameters

**MODEL_PATH:**
Path to the saved checkpoint of the trained model. This checkpoint file contains the model's weights and other training state data needed to make predictions on new images.

**PREDICTED_MASKS:**
Directory where the predicted segmentation masks will be saved after running the predict.py script. These masks will represent the model's predictions for the input satellite images.
