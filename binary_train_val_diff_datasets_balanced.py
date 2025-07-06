from binary_ffnn import *
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import logging
import click
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn


@click.command()
@click.option("--train_input", "-ti", help="path to training data", type=click.Path(exists=True), required=True)
@click.option("--train_labels", "-tl", help="path to labels of training data", type=click.Path(exists=True), required=True)
@click.option("--val_input", "-vi", help="path to validation data", type=click.Path(exists=True), required=True)
@click.option("--val_labels", "-vl", help="path to labels of validation data", type=click.Path(exists=True), required=True)
@click.option("--model", "-m", help="path to save model", type=str, required=True)
@click.option("--output", "-o", help="path to save output", type=str, required=True)
@click.option("--batch_size", "-b", help="batch size", type=int, default=1024, show_default=True, required=False)
@click.option("--epoches", "-e", help="number of epoches", type=int, default=50, show_default=True, required=False)
@click.option("--learning_rate", "-lr", help="learning rate", type=float, default=0.001, show_default=True, required=False)
@click.help_option("--help", "-h", help="Show this message and exit")

def main(train_input, train_labels, val_input, val_labels, model, output, batch_size, epoches, learning_rate):

    newModelPath = model

    logger = logging.getLogger("FFNN")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    fileHandler = logging.FileHandler(f"{output}.log")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    logger.info(f"Model path: {newModelPath}")
    logger.info(f"Training data path: {train_input}")
    logger.info(f"Training labels path: {train_labels}")
    logger.info(f"Validation data path: {val_input}")
    logger.info(f"Validation labels path: {val_labels}")
    logger.info(f"Results path: {output}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    startTime = time.time()

    # Load training data
    logger.info("Loading training data...")
    train_df = pd.read_csv(train_labels)
    train_data_arr = pd.read_csv(train_input, header=None).to_numpy()
    X_train = train_data_arr.astype(np.float32)
    y_train = train_df["y_true"].to_numpy()

    # Load validation data
    logger.info("Loading validation data...")
    val_df = pd.read_csv(val_labels)
    val_data_arr = pd.read_csv(val_input, header=None).to_numpy()
    X_val = val_data_arr.astype(np.float32)
    y_val = val_df["y_true"].to_numpy()

    # Create datasets
    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))

    trainDataLoader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valDataLoader = DataLoader(val_data, shuffle=True, batch_size=batch_size)

    logger.info("Initializing the FFNN model...")
    model = nn.DataParallel(FeedForwardNN()).to(device)

    opt = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    lossFn = nn.BCELoss()
    logger.info("Training the network...")

    train_losses = []
    max_val_acc = 0
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    epoch_list = []
    
    for e in range(epoches):
        model.train()
        total_loss = 0.0
        correct_train_predictions = 0
        
        for x, y in trainDataLoader:
            x, y = x.to(device), y.to(device)
            y = y.view(-1, 1).float()
            
            pred = model(x)
            loss = lossFn(pred, y)
            total_loss += loss.item()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            predicted_labels = (pred > 0.5).float()
            correct_train_predictions += (predicted_labels == y).sum().item()

        train_loss = total_loss / len(trainDataLoader)
        train_accuracy = correct_train_predictions / len(trainDataLoader.dataset)

        model.eval()
        total_val_loss = 0.0
        correct_val_predictions = 0
        
        with torch.no_grad():
            for val_x, val_y in valDataLoader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_y = val_y.view(-1, 1).float()
                
                val_pred = model(val_x)
                val_loss = lossFn(val_pred, val_y)
                total_val_loss += val_loss.item()
                
                predicted_val_labels = (val_pred > 0.5).float()
                correct_val_predictions += (predicted_val_labels == val_y).sum().item()

        val_loss = total_val_loss / len(valDataLoader)
        val_accuracy = correct_val_predictions / len(valDataLoader.dataset)

        # Store losses and accuracies
        train_losses.append(train_loss)
        validation_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)
        epoch_list.append(e + 1)

        logger.info(f"Epoch {e+1}/{epoches}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        if max_val_acc < val_accuracy:
            max_val_acc = val_accuracy
            train_acc_at_max_val_acc = train_accuracy
            epoch_of_max_val_acc = e
            torch.save(model.state_dict(), newModelPath)

    
    logger.info(f"Best validation accuracy: {max_val_acc}")
    logger.info(f"epoch: {epoch_of_max_val_acc}")

    endTime = time.time()
    memory = psutil.Process().memory_info()

    logger.info(
        "total time taken to train the model: {:.2f} min".format(
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
    plt.savefig(f"{output}_losses.png", dpi=300, bbox_inches="tight")
    plt.clf()

    # Plot training/validation accuracies vs. epochs
    plt.plot(epoch_list, train_accuracies, label="Training accuracy")
    plt.plot(epoch_list, validation_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(f"{output}_accuracies.png", dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    main()
