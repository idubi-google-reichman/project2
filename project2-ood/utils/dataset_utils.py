from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets
import app_const as const


def load_dataset(dataset_name, validation_split, dec_factor):
    # Load the CIFAR-10 dataset
    dts = getattr(datasets, dataset_name).load_data()
    (train_images, train_labels), (test_images, test_labels) = dts

    # Reduce the number of images by a factor of dec_factor
    train_images = train_images[::dec_factor]  # Take every Nth image
    train_labels = train_labels[::dec_factor]  # Corresponding labels
    test_images = test_images[::dec_factor]
    test_labels = test_labels[::dec_factor]

    # Split the training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=validation_split, random_state=const.SEED
    )

    # Normalize pixel values to be between 0 and 1
    train_images, val_images, test_images = (
        train_images / 255.0,
        val_images / 255.0,
        test_images / 255.0,
    )

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, 10)
    val_labels = to_categorical(val_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels
