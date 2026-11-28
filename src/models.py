"""Model-building functions. Each returns a new, untrained Keras model."""
import tensorflow as tf
from tensorflow.keras import layers


def build_mlp():
    """Dense baseline: Flatten -> 512 -> 256 -> 10. No knowledge of image structure."""
    return tf.keras.Sequential([
        tf.keras.Input(shape=(32, 32, 3)),        # declares the input shape, so summary() works now
        layers.Flatten(),                         # (32, 32, 3) -> (3072,)
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10, activation="softmax"),   # one probability per class
    ], name="mlp_baseline")