# import pandas as pd
# import logging
# import click
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import os 
# import matplotlib.pyplot as plt
# from sklearn.calibration import calibration_curve

# @click.command()
# @click.option(
#     "--pred",
#     "-p",
#     help="Path to predicted labels file",
#     type=click.Path(exists=True),
#     required=True,
# )
# @click.option(
#     "--true",
#     "-t",
#     help="Path to true labels file",
#     type=click.Path(exists=True),
#     required=True,
# )
# @click.option(
#     "--logfile",
#     "-lf",
#     help="Path to log output file (optional)",
#     type=click.Path(),
#     required=False,
# )
# @click.help_option("--help", "-h", help="Show this message and exit")

# def main(pred, true, logfile):
#     # Setup logger
#     logger = logging.getLogger("Binary FFNN")
#     logger.setLevel(logging.DEBUG)
#     logging.captureWarnings(True)

#     # Formatter for log messages
#     formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

#     # Console handler (for logging to console)
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)
#     console_handler.setLevel(logging.INFO)
#     logger.addHandler(console_handler)

#     # File handler (for logging to a file, if the logfile option is provided)
#     if logfile:
#         file_handler = logging.FileHandler(logfile)
#         file_handler.setFormatter(formatter)
#         file_handler.setLevel(logging.DEBUG)
#         logger.addHandler(file_handler)

#     # Load predicted and true label files
#     pred_df = pd.read_csv(pred)
#     true_df = pd.read_csv(true, usecols=['id', 'y_true'])

#     # # Create a dictionary of predictions
#     # pred_dict = {row["id"]: row["pred_label"] for _, row in pred_df.iterrows()}

#     # # Create a list to store true labels by class
#     # true_list = [[] for _ in range(2)]
#     # for _, row in true_df.iterrows():
#     #     true_list[row["y_true"]].append(row["id"])
   
#     # # Prepare true and predicted label lists
#     # predicted_labels = []
#     # true_labels = []
#     # for clz in range(2):
#     #     for ele in true_list[clz]:
#     #         predicted_labels.append(pred_dict[ele])
#     #     true_labels.extend(np.full(len(true_list[clz]), clz))
    

#     # Create a set of IDs from the true labels to filter predictions
#     valid_ids = set(true_df["id"])

#     # Create a dictionary of predictions, filtering out invalid IDs
#     # pred_dict = {row["id"]: row["pred_label"] for _, row in pred_df.iterrows() if row["id"] in valid_ids}

#     # Initialize an empty dictionary to store predictions
#     pred_dict = {}

#     # Iterate over each row in the predicted DataFrame
#     for _, row in pred_df.iterrows():
#         if row["id"] in valid_ids:  # Check if the ID exists in the set of valid IDs
#             pred_dict[row["id"]] = row["pred_label"]  # Add the ID and prediction to the dictionary


#     # Prepare lists for true and predicted labels
#     predicted_labels = []
#     true_labels = []

#     # Only include entries where the ID exists in both true and predicted labels
#     for _, row in true_df.iterrows():
#         if row["id"] in pred_dict:  # Ensure the ID has a corresponding prediction
#             true_labels.append(row["y_true"])
#             predicted_labels.append(pred_dict[row["id"]])


#     # Generate classification report
#     logger.info(
#         f'\n {classification_report(true_labels, predicted_labels, target_names=["Host", "Microbial"])}'
#     )

#     # Compute and log confusion matrix
#     cm = confusion_matrix(true_labels, predicted_labels)
#     cm_df = pd.DataFrame(
#         cm,
#         index=["Host", "Microbial"],
#         columns=["Host", "Microbial"],
#     )
#     logger.info(f"\nConfusion Matrix:\n{cm_df}")

#     # Normalize confusion matrix and log percentage version
#     cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
#     cm_df_normalized = pd.DataFrame(
#         cm_normalized,
#         index=["Host", "Microbial"],
#         columns=["Host", "Microbial"],
#     )
#     logger.info(f"\nConfusion Matrix (Percentages):\n{cm_df_normalized}")

#     # merge true and predicted labels
#     merged_df = pd.merge(true_df,pred_df, on='id')
    

#     # Save concatenated DataFrame to the same path as the predicted file
#     output_path = os.path.join(os.path.dirname(pred), "trueAndPredictedLabels.csv")
#     merged_df.to_csv(output_path, index=False)
#     # logger.info(f"Concatenated CSV saved to {output_path}")

#     # Calculate sensitivity and specificity
#     # Confusion matrix: [TN, FP], [FN, TP]
#     TN, FP, FN, TP = cm.ravel()

#     sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
#     specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

#     # Save sensitivity and specificity to logs
#     logger.info(f"Sensitivity: {sensitivity:.4f}")
#     logger.info(f"Specificity: {specificity:.4f}")


#     # # Ensure the predicted DataFrame contains probabilities
#     # if "pred_prob" not in pred_df.columns:
#     #     logger.error("Prediction file must include probabilities for calibration plots.")
#     #     return

#     # # Filter probabilities for valid IDs
#     # predicted_probs = []
#     # for _, row in true_df.iterrows():
#     #     if row["id"] in pred_dict:  # Ensure the ID has a corresponding prediction
#     #         predicted_probs.append(pred_df.loc[pred_df["id"] == row["id"], "pred_prob"].values[0])

#     # # Compute calibration curve
#     # fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, predicted_probs, n_bins=10)

#     # # Plot the calibration curve
#     # plt.figure(figsize=(8, 6))
#     # plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
#     # plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
#     # plt.title("Calibration Plot")
#     # plt.xlabel("Mean Predicted Probability")
#     # plt.ylabel("Fraction of Positives")
#     # plt.legend(loc="best")
#     # plt.grid()
#     # plt.tight_layout()

#     # # Save the plot
#     # calibration_plot_path = os.path.join(os.path.dirname(pred), "calibration_plot.png")
#     # plt.savefig(calibration_plot_path)
#     # logger.info(f"Calibration plot saved to {calibration_plot_path}")
#     # plt.show()

    
#     # Ensure the predicted DataFrame contains probabilities
#     # if "pred_prob" not in pred_df.columns:
#     #     logger.error("Prediction file must include probabilities for calibration plots.")
#     #     return

#     # logger.info("Prediction file contains probabilities. Proceeding with calibration plot.")

#     # # Filter probabilities for valid IDs
#     # predicted_probs = []
#     # for _, row in true_df.iterrows():
#     #     if row["id"] in pred_dict:  # Ensure the ID has a corresponding prediction
#     #         prob = pred_df.loc[pred_df["id"] == row["id"], "pred_prob"].values[0]
#     #         predicted_probs.append(prob)

#     # # Log the size of true_labels and predicted_probs to verify matching lengths
#     # logger.info(f"True labels count: {len(true_labels)}, Predicted probabilities count: {len(predicted_probs)}")

#     # # Check if the predicted_probs list is empty
#     # if not predicted_probs:
#     #     logger.error("No predicted probabilities found for calibration. Skipping plot generation.")
#     #     return

#     # # Compute calibration curve
#     # fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, predicted_probs, n_bins=10)

#     # # Log calibration curve details
#     # logger.debug(f"Fraction of Positives: {fraction_of_positives}")
#     # logger.debug(f"Mean Predicted Values: {mean_predicted_value}")

#     # # Plot the calibration curve
#     # plt.figure(figsize=(8, 6))
#     # plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
#     # plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
#     # plt.title("Calibration Plot")
#     # plt.xlabel("Mean Predicted Probability")
#     # plt.ylabel("Fraction of Positives")
#     # plt.legend(loc="best")
#     # plt.grid()
#     # plt.tight_layout()

#     # # Save the plot
#     # calibration_plot_path = os.path.join(os.path.dirname(pred), "calibration_plot.png")
#     # plt.savefig(calibration_plot_path)
#     # logger.info(f"Calibration plot saved to {calibration_plot_path}")

#     # # Show the plot explicitly to render it
#     # plt.show()


# if __name__ == "__main__":
#     main()

import pandas as pd
import logging
import click
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import os 

@click.command()
@click.option(
    "--pred",
    "-p",
    help="Path to predicted labels file",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--true",
    "-t",
    help="Path to true labels file",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--min_length",
    "-ml",
    help="Minimum read length to consider",
    type=int,
    required=True,
)
@click.option(
    "--logfile",
    "-lf",
    help="Path to log output file (optional)",
    type=click.Path(),
    required=False,
)
@click.help_option("--help", "-h", help="Show this message and exit")

def main(pred, true, min_length, logfile):
    # Setup logger
    logger = logging.getLogger("Binary FFNN")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)

    # Formatter for log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler (for logging to console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler (for logging to a file, if the logfile option is provided)
    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    # Load predicted and true label files
    pred_df = pd.read_csv(pred)
    true_df = pd.read_csv(true, usecols=['id', 'y_true', 'length'])

    # Filter by minimum read length
    true_df = true_df[true_df['length'] >= min_length]

    if "pred_prob" not in pred_df.columns:
        logger.error("The predicted file must include a 'pred_prob' column for calibration plot.")
        return

    pred_dict = {row["id"]: row for _, row in pred_df.iterrows()}

    true_labels = true_df["y_true"].tolist()
    predicted_probs = [pred_dict[row["id"]]["pred_prob"] for _, row in true_df.iterrows()]
    predicted_labels = [pred_dict[row["id"]]["pred_label"] for _, row in true_df.iterrows()]

    # Generate classification report
    logger.info(
        f'\n {classification_report(true_labels, predicted_labels, target_names=["Host", "Microbial"])}'
    )

    # Compute and log confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_df = pd.DataFrame(cm, index=["Host", "Microbial"], columns=["Host", "Microbial"])
    logger.info(f"\nConfusion Matrix:\n{cm_df}")

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_df_normalized = pd.DataFrame(cm_normalized, index=["Host", "Microbial"], columns=["Host", "Microbial"])
    logger.info(f"\nConfusion Matrix (Percentages):\n{cm_df_normalized}")

    # Generate calibration plot
    prob_true, prob_pred = calibration_curve(true_labels, predicted_probs, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label="Calibration Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
    plt.title("Calibration Plot")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    # output_plot_path = os.path.join(os.path.dirname(pred), "calibration_plot.png")
    # plt.savefig(output_plot_path)
    # logger.info(f"Calibration plot saved to {output_plot_path}")

if __name__ == "__main__":
    main()
