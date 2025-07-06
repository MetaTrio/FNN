from binary_ffnn_imbalanced import * 
import torch
import pandas as pd
from Bio import SeqIO
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import logging
import click
import numpy as np
import psutil
from sklearn.metrics import classification_report


@click.command()
@click.option(
    "--input_fasta_fastq",
    "-i",
    help="path to input (fasta/fastq) data",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--type_",
    "-t",
    help="type of the input file (fasta or fastq)",
    type=click.Choice(["fasta", "fastq"]),
    required=True,
)
@click.option(
    "--input_kmers",
    "-k",
    help="path to generated k-mers",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--model",
    "-m",
    help="path to the model",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output",
    "-o",
    help="path to a folder to save predictions",
    type=str,
    required=True,
)
@click.help_option("--help", "-h", help="Show this message and exit")
def main(input_fasta_fastq, type_, input_kmers, model, output):
    
    inputset = input_fasta_fastq
    type_ = type_
    k_mers = input_kmers
    modelPath = model
    
    resultPath = output.rstrip('/') + '/'
    
    # Initialize logger
    logger = logging.getLogger("BinaryClassifier")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(f"{resultPath}info.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load and prepare input data
    k_mer_arr = pd.read_csv(k_mers, header=None).to_numpy()
    # input_data = [np.reshape(row.astype(np.float32), (32,)) for row in k_mer_arr]  # 3mer
    # input_data = [np.reshape(row.astype(np.float32), (136,)) for row in k_mer_arr]  # 4mer
    input_data = [np.reshape(row.astype(np.float32), (42,)) for row in k_mer_arr]  # 2mer, 3mer
    
    accession_numbers = [seq.id for seq in SeqIO.parse(inputset, type_)]
    
    # Track time for data parsing
    logger.info("Initializing model and loading data...")
    start_time = time.time()

    # Initialize model and load state
    model = nn.DataParallel(FeedForwardNN())
    model.load_state_dict(torch.load(modelPath, device))
    model = model.to(device)
    model.eval()

    logger.info("predicting the classes...")

    # Track memory before prediction
    memory_before = psutil.Process().memory_info()

    dataLoader = DataLoader(input_data, shuffle=False, batch_size=2048)
    
    # Prediction
    predictions = []
    prdicted_probabilities = []
    # with torch.no_grad():
    #     for data in input_data:
    #         x = torch.tensor(data).to(device)
    #         output = model(x).cpu().item()
    #         predictions.append(output)

    with torch.no_grad():
        for step, test_x in enumerate(dataLoader):
            test_x = test_x.to(device)
            pred = model(test_x)
            # Apply Sigmoid activation to the raw logits
            pred = torch.sigmoid(pred)
            predictions.extend(pred.cpu().numpy())


    # Threshold for binary output
    # predicted_labels = [1 if p > 0.5 else 0 for p in predictions]

    # Prediction with labels and probabilities in one loop
    # predicted_labels_probs = [(1 if p > 0.9899 else 0, float(p)) for p in predictions]
    predicted_labels_probs = [(1 if p > 0.65 else 0, float(p)) for p in predictions]
    # predicted_labels_probs = [(1 if p > 0.5 else 0, float(p)) for p in predictions]

    # Unpack labels and probabilities for DataFrame creation
    predicted_labels, probabilities = zip(*predicted_labels_probs)

    elapsed_time = time.time() - start_time
    memory_after = psutil.Process().memory_info()
    
    # Log time and memory
    logger.info(f"Total time for predictions: {elapsed_time / 60:.2f} minutes")
    logger.info(f"Memory usage before predictions: {memory_before.rss / (1024 * 1024 * 1024):.2f} GB")
    logger.info(f"Memory usage after predictions: {memory_after.rss / (1024 * 1024 * 1024):.2f} GB")
    
    # Output predictions to file
    prediction_data = pd.DataFrame({"id": accession_numbers, "pred_label": predicted_labels, "pred_prob": probabilities})
    prediction_data.to_csv(f"{resultPath}predictions.csv", index=False)
    logger.info("Predictions saved successfully.")

if __name__ == "__main__":
    main()
