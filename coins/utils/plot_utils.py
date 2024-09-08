import matplotlib.pyplot as plt
import pandas as pd


def compare_labeled_and_prediction(labeled_summary, prediction_summary):

    # Prepare data for comparison
    image_summary_filtered = [d for d in prediction_summary if "file" in d]
    labeled_summary_filtered = [d for d in labeled_summary if "file" in d]
    # Extract filenames and values from both arrays
    image_files = [d["file"].replace(".jpg", "") for d in image_summary_filtered]
    image_values = [d["value"] for d in image_summary_filtered]
    labeled_values = [d["value"] for d in labeled_summary_filtered]

    # Calculate percentage deviation and success/fail status
    deviation = []
    status = []

    for img_val, lbl_val in zip(image_values, labeled_values):
        if lbl_val == 0:  # Avoid division by zero
            deviation.append(0 if img_val == 0 else 100)  # If both 0, no deviation
            status.append("Success" if img_val == 0 else "Fail")
        else:
            perc_deviation = abs(img_val - lbl_val) / lbl_val * 100
            deviation.append(perc_deviation)
            status.append("Success" if perc_deviation == 0 else "Fail")

    # Create a pandas dataframe
    data = pd.DataFrame(
        {
            "file": image_files,
            "image_values": image_values,
            "labeled_values": labeled_values,
            "deviation (%)": deviation,
            "status": status,
        }
    )

    # Print the results
    print(data)

    # Plot the data with success/fail status
    fig, ax = plt.subplots(figsize=(10, 6))

    # Function to determine the bar color based on success/fail status
    def get_fill_color(status):
        return "green" if status == "Success" else "red"

    # Plot the image and labeled values
    for i, row in data.iterrows():
        # Image (prediction) values
        ax.bar(
            i - 0.2,
            row["image_values"],
            width=0.4,
            color=get_fill_color(row["status"]),
            edgecolor="orange",
            linewidth=2,
            label="Prediction" if i == 0 else "",
        )
        # Labeled values
        ax.bar(
            i + 0.2,
            row["labeled_values"],
            width=0.4,
            color=get_fill_color(row["status"]),
            edgecolor="blue",
            linewidth=2,
            label="Labeled" if i == 0 else "",
        )

    # Add success/fail labels with colors
    for i, row in data.iterrows():
        ax.text(
            i,
            max(row["image_values"], row["labeled_values"]) + 0.5,
            row["status"],
            ha="center",
            color="black",
            fontsize=10,
        )

    # Customize plot
    plt.title(
        "Comparison of Image Values and Labeled Values with Success/Fail Status",
        fontsize=14,
    )
    plt.xticks(range(len(image_files)), image_files, rotation=45, ha="right")
    plt.ylabel("Value", fontsize=12)
    plt.xlabel("Files", fontsize=12)

    # Add legend for clarity
    plt.legend()

    # Adjust layout for better display
    plt.tight_layout()

    # Show plot
    plt.show()
