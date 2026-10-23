# Variance-Guided Feature Projection

Variance-guided feature projection is a method developed to reduce complexity while preserving expressiveness. This is done by projecting input into a variance varying feature space through a module, namely the Variance-initialized Linear Projection (VLP) module. The publication of the method can be found at [link]().

## Required Packages

Current script is only tested to work on Python 3.12.4, other versions might require further testing and correction of syntaxes where needed. The required version of packages is Keras (3.7.0) and TensorFlow (2.17.0) Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install keras
pip install tensorflow[and-cuda]==2.17.0
```

## Usage

The repository contains the Python file VLP.py for the class module, main.py for the model loading and prediction, modify_MBV3.py for the modified VLP-MobileNetV3. For ShuffleNetV2, we have provided the modified version that matches the syntaxes of the required packages. Alternatively, you can find the original repository in [ShuffleNetV2](https://github.com/opconty/keras-shufflenetv2). The repository contains trained model checkpoints under [link]() that can be loaded via the following sample usage code. The performance of each models can be found in the publication, at [link]().

```python
# Import the VLP layer from VLP.py
from VLP import ProjectionLayer

# You need to include keras_shufflenetV2 under scripts as Root Sources (if using PyCharm) to be able to import
from keras_shufflenetV2 import shufflenetv2, utils
from shufflenetv2 import ShuffleNetV2

# Import Keras (v3.7.0) and TensorFlow (v2.17.0)
import keras
import tensorflow as tf


if __name__ == '__main__':
    # Initialize file path
    file_path = "" # Put the file path to the .keras file

    # For ShuffleNetV2
    # trained_model = keras.models.load_model(f"{file_path}",
    #                                         custom_objects={"channel_split": utils.channel_split,
    #                                                         "channel_shuffle": utils.channel_shuffle,
    #                                                         "shuffle_unit": utils.shuffle_unit,
    #                                                         "block": utils.block})

    # For all other models
    # trained_model = keras.models.load_model(f"{file_path}")

    # Call trained_model.predict() to perform prediction

    # You can create and connect the layer in Keras style by calling the following
    # input_layer = keras.layers.Input(shape=224, 224, 3)
    # layer = ProjectionLayer(projection_var=1.0, name="layer_1")(input_layer)
```

## Dataset
The repository contains datasets used to test the models and can be found inside [link](). Particularly, the repository contains only the modified datasets for reproducibility purposes. The original datasets can be found at [CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html), [CODEBRIM](https://zenodo.org/records/2620293), [PlantVillage](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) and [PneumoniaMNIST](https://www.tensorflow.org/datasets/catalog/pneumonia_mnist).
