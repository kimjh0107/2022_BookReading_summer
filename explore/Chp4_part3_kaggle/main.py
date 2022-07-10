import itertools

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.kfold import kfold_learning
from src.loss import loss
from src.model import Net
from src.train import train


def extract_optimize_params(results):
    target = np.Inf
    learning_rate = 0
    weight_decay = 0
    epoch = 0
    batch_size = 0

    for result in results:
        if target > result["average_valid_rmse"]:
            target = result["average_valid_rmse"]
            learning_rate = result["learning_rate"]
            weight_decay = result["weight_decay"]
            epoch = result["epoch"]
            batch_size = result["batch_size"]

    return learning_rate, weight_decay, epoch, batch_size


def main():
    config = load_config()

    train_X, test_X = torch.load("data/processed/train_X.pt"), torch.load(
        "data/processed/test_X.pt"
    )
    train_y = torch.load("data/processed/train_y.pt")
    criterion = loss()

    results = []

    learning_rates = [5, 10]
    weight_decays = [0, 0.001, 0.01, 0.1]
    epochs = [50, 100, 500]
    batch_sizes = [256, 512, 1024]

    for learning_rate, weight_decay, epoch, batch_size in itertools.product(
        learning_rates, weight_decays, epochs, batch_sizes
    ):
        metrics = kfold_learning(
            5,
            train_X,
            train_y,
            Net(train_X.shape[1]),
            criterion,
            learning_rate,
            weight_decay,
            epoch,
            batch_size,
        )
        average_train_rmse = np.mean(
            [metric.train_loss_rmse for metric in metrics]
        )

        average_valid_rmse = np.mean(
            [metric.valid_loss_rmse for metric in metrics]
        )

        results.append(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epoch": epoch,
                "batch_size": batch_size,
                "average_train_rmse": average_train_rmse,
                "average_valid_rmse": average_valid_rmse,
            }
        )

    pd.DataFrame(results).to_csv("data/processed/kfold_output.csv")

    learning_rate, weight_decay, epoch, batch_size = extract_optimize_params(
        results
    )

    net = Net(train_X.shape[1])

    train_losses, _ = train(
        train_X,
        train_y,
        None,
        None,
        net,
        criterion,
        learning_rate,
        weight_decay,
        epoch,
        batch_size,
    )

    print(net(test_X))


if __name__ == "__main__":
    main()
