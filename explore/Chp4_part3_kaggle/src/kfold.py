from typing import Callable, Type
import torch
from sklearn.model_selection import KFold
from src.metric import Metric
from src.train import train


def kfold_learning(
    k: int,
    X: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module,
    criterion: Callable,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
):
    kf = KFold(n_splits=k)
    metrics = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]

        train_losses, valid_losses = train(
            X_train,
            y_train,
            X_valid,
            y_valid,
            model,
            criterion,
            learning_rate,
            weight_decay,
            epochs,
            batch_size,
        )
        metrics.append(Metric(fold, train_losses, valid_losses))

    return metrics
