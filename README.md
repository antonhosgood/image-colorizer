# Image Colourisation with PyTorch

This project contains a PyTorch framework to train deep learning models from scratch to automatically add colour to
black-and-white (greyscale) images.

## Repository Structure

### Core

```
src/
 ├── api/                        # Utilities to interact with data sources
 │   └── lorem_picsum.py
 ├── data/                       # Dataset and data loading utilities
 │   ├── stock_image_dataset.py
 │   └── utils.py
 ├── infer/                      # Inference pipeline
 │   ├── config.yaml
 │   └── inference.py
 ├── models/                     # Model architectures
 │   └── unet.py
 ├── train/                      # Training pipeline scripts and utilities
 │   ├── config.yaml             # Training parameters
 │   ├── train.py                # Entry point for training
 │   └── trainer.py              # Epoch training and validation logic
 └── utils/                      # General utilities
     ├── checkpoint.py           # Model saving/loading
     ├── config.py               # Config loading logic
     └── device.py               # Device selection (CPU/GPU)
```

### Other Components

#### Notebooks

Used for prototyping, experimentation, or showcasing model predictions.

* `demo.ipynb` — A sample notebook to train a model from scratch and visualise outputs interactively.

#### Scripts

Standalone utility scripts.

* `generate_dataset.py` — Script to generate an example image dataset.

## Setup and Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/antonhosgood/image-colorizer.git
    cd image-colorizer
    ```

2. Create a **Python 3.13 or higher** virtual environment (optional but recommended):

   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip3 install -r requirements.txt
   ```

## Training

Train the model using the provided training script and configuration file:

```bash
python3 -m src.train.train src/train/config.yaml
```

The configuration file defines parameters like data directories, batch size, learning rate, number of epochs, etc.

Model checkpoints will be saved at intervals defined in the config.

## Inference

Colourise a greyscale image using a trained model checkpoint:

```bash
python3 -m src.infer.inference src/infer/config.yaml path/to/grayscale_image.png path/to/checkpoint.pth --output path/to/save_colorized.png
```

## Generate Sample Dataset

[Lorem Picsum](https://picsum.photos) is an API to get random images. Although by default the API returns a random
image, an identifier can be provided to request a specific image.

`generate_dataset.py` creates a dataset of colour and greyscale image pairs by first obtaining a list of every image ID
and downloading every image into a `color` and `grayscale` folder. An image width and height must be provided. Altering
the width and/or height does not stretch or shrink the images. Instead, the original source images are cropped
appropriately.

```bash
python3 -m scripts.generate_dataset data <WIDTH> <HEIGHT>
```

## Future Improvements

* Create unit tests for dataset, model, and training components
* Add TensorBoard, Weights & Biases or MLFlow integration for better training monitoring
* Support batch inference on directories of images
* Add perceptual and adversarial loss functions for better colourisation quality
