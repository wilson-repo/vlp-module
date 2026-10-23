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
