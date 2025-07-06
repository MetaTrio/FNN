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
    "--logfile",
    "-lf",
    help="Path to log output file (optional)",
    type=click.Path(),
    required=False,
)
@click.option(
    "--ignore",
    "-i",
    help="Optional CSV file containing IDs to ignore during evaluation",
    type=click.Path(exists=True),
    required=False,
)
@click.help_option("--help", "-h", help="Show this message and exit")

def main(pred, true, logfile, ignore):
    # Setup logger
    logger = logging.getLogger("Binary FFNN")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    # Load predicted and true label files
    pred_df = pd.read_csv(pred)
    true_df = pd.read_csv(true, usecols=['id', 'y_true'])

    # If ignore file is given, filter out those IDs
    if ignore:
        ignore_df = pd.read_csv(ignore, header=None, names=['id'])
        ignore_ids = set(ignore_df['id'])
        original_len = len(true_df)
        true_df = true_df[~true_df['id'].isin(ignore_ids)]
        logger.info(f"Ignored {original_len - len(true_df)} entries based on ignore file.")

    # Ensure probabilities are available in pred_df
    if "pred_prob" not in pred_df.columns:
        logger.error("The predicted file must include a 'pred_prob' column for calibration plot.")
        return

    # Merge true and predicted dataframes on 'id'
    merged_df = true_df.merge(pred_df[["id", "pred_prob", "pred_label"]], on="id", how="inner")
    logger.info(f"Matched {len(merged_df)} IDs between true and predicted labels.")


    # Extract lists for evaluation
    true_labels = merged_df["y_true"].tolist()
    predicted_labels = merged_df["pred_label"].tolist()

    # Generate classification report
    logger.info(
        f'\n {classification_report(true_labels, predicted_labels, target_names=["Host", "Microbial"])}'
    )

    # Compute and log confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_df = pd.DataFrame(cm, index=["Host", "Microbial"], columns=["Host", "Microbial"])
    logger.info(f"\nConfusion Matrix:\n{cm_df}")

    # Normalize confusion matrix and log percentage version
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_df_normalized = pd.DataFrame(
        cm_normalized, index=["Host", "Microbial"], columns=["Host", "Microbial"]
    )
    logger.info(f"\nConfusion Matrix (Percentages):\n{cm_df_normalized}")

if __name__ == "__main__":
    main()
