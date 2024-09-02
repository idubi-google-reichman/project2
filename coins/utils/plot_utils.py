import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Example: Load curves from results and plot them
def plot_and_log_curves(results_dir, experiment):
    # Example paths, adapt based on actual saved files
    pr_curve_path = f"{results_dir}/pr_curve.png"
    f1_curve_path = f"{results_dir}/f1_curve.png"
    recall_curve_path = f"{results_dir}/recall_curve.png"
    precision_curve_path = f"{results_dir}/precision_curve.png"

    # Log PR Curve
    experiment.log_image(pr_curve_path, name="PR Curve")

    # Log F1 Curve
    experiment.log_image(f1_curve_path, name="F1 Curve")

    # Log Recall Curve
    experiment.log_image(recall_curve_path, name="Recall Curve")


# Call the function after training
# plot_and_log_curves("path/to/save/directory/my_experiment", experiment)


# Example: Load and plot confusion matrix
def plot_confusion_matrix(conf_matrix_path, experiment):
    conf_matrix = np.loadtxt(conf_matrix_path, delimiter=",")
    conf_matrix_df = pd.DataFrame(
        conf_matrix, columns=["Class 0", "Class 1"], index=["Class 0", "Class 1"]
    )

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    # Save and log confusion matrix
    plt.savefig("path/to/save/directory/confusion_matrix.png")
    experiment.log_image(
        "path/to/save/directory/confusion_matrix.png", name="Confusion Matrix"
    )


# Call the function after training
# plot_confusion_matrix(
#     "path/to/save/directory/my_experiment/confusion_matrix.txt", experiment
# )
