
import tensorflow as tf
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def load_cifar10(val_size=5000, seed=42):
    """Return (train, val, test) tuples of (x, y), scaled to [0, 1]."""
    # uint8 images (N, 32, 32, 3) and labels (N, 1), cached after the first download
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # (N, 1) -> (N,): sparse integer labels for sparse_categorical_crossentropy
    y_train, y_test = y_train.ravel(), y_test.ravel()

    # uint8 [0, 255] -> float32 [0, 1]. float32 is what TF computes in.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Stratified split: validation keeps the same class balance as training
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)