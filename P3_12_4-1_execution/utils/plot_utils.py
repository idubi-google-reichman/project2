import matplotlib.pyplot as plt
from prettytable import PrettyTable
import app_const as const


def print_training_report(history, test_loss, test_accuracy, train_execution_time):
    history_dict = history.history
    # Extract the last epoch's metrics from the history
    val_accuracy = history_dict["val_accuracy"][-1]
    val_loss = history_dict["val_loss"][-1]

    table = PrettyTable()
    # Print the organized table
    table.field_names = [
        "Validation",
        "Test",
        "Inference",
    ]
    sub_headers = [
        "acc | loss",
        "acc | loss",
        "Time (sec)",
    ]
    table.add_row(sub_headers)
    table.add_row(["-" * 22, "-" * 22, "_" * 10])

    table.add_row(
        [
            f"{val_accuracy:.4f} | {val_loss:.4f}",
            f"{test_accuracy:.4f} | {test_loss:.4f}",
            f"{train_execution_time:.4f} sec",
        ]
    )
    print(table)

    print(" validation split : ", const.TRAIN_VALIDATION_SPLIT)
    print(" reduce factor    : ", const.TRAIN_DECREASE_FACTOR)
    print(" number of epocs  : ", const.TRAIN_EPOCHS)
    print(" batch size       : ", const.BATCH_SIZE)
    print(" learning rate    : ", const.LEARNING_RATE)
    print(" is freeze model  : ", const.FREEZE_MODEL)
    print(" loss function    : ", const.LOSS_FUNCTION)
    print(" dataset name     : ", const.DATASET_NAME)
    print(" random seed      : ", const.SEED)

    return


def plot_training_history(history):
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
    plt.show()
