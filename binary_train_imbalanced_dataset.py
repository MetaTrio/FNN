from binary_ffnn_imbalanced import *
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import logging
import click
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import psutil
import torch
import torch.nn as nn

@click.command()
@click.option(
    "--input",
    "-i",
    help="path to training data",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--labels",
    "-l",
    help="path to labels of training data",
    type=click.Path(exists=True),
    required=True,
)
@click.option("--model", "-m", help="path to save model", type=str, required=True)
@click.option("--output", "-o", help="path to save output", type=str, required=True)
@click.option(
    "--batch_size",
    "-b",
    help="batch size",
    type=int,
    default=1024,
    show_default=True,
    required=False,
)
@click.option(
    "--epoches",
    "-e",
    help="number of epoches",
    type=int,
    default=50,
    show_default=True,
    required=False,
)
@click.option(
    "--learning_rate",
    "-lr",
    help="learning rate",
    type=float,
    default=0.001,
    show_default=True,
    required=False,
)
@click.help_option("--help", "-h", help="Show this message and exit")
def main(input, labels, model, output, batch_size, epoches, learning_rate):

    newModelPath = model
    inputset = input
    labelset = labels
    resultPath = output
    
    batchSize = batch_size
    epoches = epoches
    learningRate = learning_rate

    logger = logging.getLogger(f"FFNN")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    fileHandler = logging.FileHandler(f"{resultPath}.log")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    logger.info(f"Model path: {newModelPath}")
    logger.info(f"Input path: {inputset}")
    logger.info(f"Labels path: {labelset}")
    logger.info(f"Results path: {resultPath}")

    INIT_LR = learningRate
    BATCH_SIZE = int(batchSize)
    EPOCHES = epoches

    logger.info(f"Learning rate: {INIT_LR}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"# Epoches: {EPOCHES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading label data...")
    train_df = pd.read_csv(labelset)
    logger.info("Loading input data...")
    train_data_arr = pd.read_csv(inputset, header=None).to_numpy()

    logger.info("Parsing data...")
    startTime = time.time()

    X = train_data_arr.astype(np.float32)
    y = train_df["y_true"].to_numpy()

    endTime = time.time()
    encoding_time_diff = (endTime - startTime) / 60
    logger.info(f"Total time taken to parse data: {encoding_time_diff} min")

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))

    trainDataLoader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)

    logger.info("Initializing the FFNN model...")
    model = nn.DataParallel(FeedForwardNN()).to(device)

    # Calculate pos_weight for BCEWithLogitsLoss
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor(num_neg / num_pos).to(device)

    # Use BCEWithLogitsLoss for binary classification with pos_weight
    opt = Adam(model.parameters(), lr=INIT_LR, weight_decay=1e-5)
    lossFn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logger.info("Training the network...")

    startTime = time.time()
    max_val_acc = 0
    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    epoch_list = [e + 1 for e in range(0, EPOCHES)]

    for e in range(0, EPOCHES):
        model.train()

        total_loss = 0.0
        correct_train_predictions = 0

        for step, (x, y) in enumerate(trainDataLoader):
            (x, y) = (x.to(device), y.to(device))

            # Reshape labels for BCEWithLogitsLoss to [batch_size, 1]
            y = y.view(-1, 1).float()

            pred = model(x)
            loss = lossFn(pred, y)
            total_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Calculate correct predictions
            predicted_labels = (torch.sigmoid(pred) > 0.5).float()
            correct_train_predictions += (predicted_labels == y).sum().item()

        model.eval()

        total_val_loss = 0.0
        correct_val_predictions = 0

        with torch.no_grad():
            for step, (val_x, val_y) in enumerate(valDataLoader):
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_y = val_y.view(-1, 1).float()

                val_pred = model(val_x)
                val_loss = lossFn(val_pred, val_y)
                total_val_loss += val_loss.item()

                predicted_val_labels = (torch.sigmoid(val_pred) > 0.5).float()
                correct_val_predictions += (predicted_val_labels == val_y).sum().item()

        train_accuracy = correct_train_predictions / len(trainDataLoader.dataset)
        val_accuracy = correct_val_predictions / len(valDataLoader.dataset)
        total_val_loss = total_val_loss / len(valDataLoader.dataset)
        total_loss = total_loss / len(trainDataLoader.dataset)

        logger.info(
            f"Epoch {e+1}/{EPOCHES}, Training Loss: {total_loss}, Train Accuracy: {train_accuracy}, Validation Loss: {total_val_loss}, Validation Accuracy: {val_accuracy}"
        )

        train_losses.append(total_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(total_val_loss)
        validation_accuracies.append(val_accuracy)

        if max_val_acc < val_accuracy:
            max_val_acc = val_accuracy
            train_acc_at_max_val_acc = train_accuracy
            epoch_of_max_val_acc = e
            torch.save(model.state_dict(), newModelPath)

    endTime = time.time()
    memory = psutil.Process().memory_info()

    logger.info(
        "Total time taken to train the model: {:.2f} min".format(
            (endTime - startTime) / 60
        )
    )
    
    # Convert memory values from bytes to GB
    rss_gb = memory.rss / (1024 ** 3)
    vms_gb = memory.vms / (1024 ** 3)
    shared_gb = memory.shared / (1024 ** 3)
    text_gb = memory.text / (1024 ** 3)
    data_gb = memory.data / (1024 ** 3)

    # Log the values in GB
    logger.info(f"Memory usage: rss={rss_gb:.2f} GB, vms={vms_gb:.2f} GB, shared={shared_gb:.2f} GB, text={text_gb:.2f} GB, data={data_gb:.2f} GB")


    logger.info(f"Maximum training accuracy = {train_acc_at_max_val_acc:.7f}")
    logger.info(f"Maximum validation accuracy = {max_val_acc:.7f}")
    logger.info(f"Epoch of maximum validation accuracy = {epoch_of_max_val_acc}")

    # Plot training/validation losses vs. epochs
    plt.plot(epoch_list, train_losses, label="Training loss")
    plt.plot(epoch_list, validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(f"{resultPath}_losses.png", dpi=300, bbox_inches="tight")
    plt.clf()

    # Plot training/validation accuracies vs. epochs
    plt.plot(epoch_list, train_accuracies, label="Training accuracy")
    plt.plot(epoch_list, validation_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(f"{resultPath}_accuracies.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
