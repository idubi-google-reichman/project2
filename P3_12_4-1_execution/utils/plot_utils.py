import matplotlib.pyplot as plt
from utils.logging_utils import LoggerUtility, LOG_LEVEL


def plot_training_history(history, save_path=None):
    """
    Plots the training and validation accuracy and loss per epoch.

    Parameters:
    history (History): The history object returned by the fit method of a Keras model.
    """
    # Extracting data from history
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    # Plotting training and validation accuracy
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "g-", label="Training Accuracy")
    plt.plot(epochs, val_acc, "b-", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plotting training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "r-", label="Training Loss")
    plt.plot(epochs, val_loss, "y-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    if not save_path:
        plt.show()
    else:
        LoggerUtility.log_message(
            "plot_utility",
            f"Saving plot to {save_path}.png",
            LOG_LEVEL["INFO"],
        )
        plt.savefig(f"{save_path}.png", format="png")
