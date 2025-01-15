"""Module containing metrics."""

import itertools

import numpy as np
import torch
from nltk.metrics.distance import edit_distance as ed
from torchmetrics import Metric, MetricCollection


def get_metrics() -> MetricCollection:
    """Returns a list of metrics.

    :return: torchmetrics.MetricCollection
    """
    return MetricCollection(
        {
            "string_match": StringMatchMetric(),
            "edit_distance": EditDistanceMetric(),
        }
    )


class StringMatchMetric(Metric):
    """Computes the string matches between two tensors."""

    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total = 0
        self.add_state(
            "correct",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Calculate metric.

        :param preds: predicted value
        :param target: true value
        :return: None
        """
        batch_size = torch.tensor(target.shape[0])
        metric = torch.tensor(string_match(preds, target))
        self.correct += metric * batch_size
        self.total += batch_size

    def compute(self) -> torch.Tensor:
        """Compute the metric.

        :return: torch.Tensor
        """
        return self.correct / self.total


class EditDistanceMetric(Metric):
    """Computes the edit distance metric."""

    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total = 0
        self.add_state(
            "correct",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric value.

        :param preds: predicted values
        :param target: true values
        :return: None
        """
        batch_size = torch.tensor(target.shape[0])
        metric = torch.tensor(edit_distance(preds, target))
        self.correct += metric * batch_size
        self.total += batch_size

    def compute(self) -> torch.Tensor:
        """Computes the edit distance metric.

        :return: edit distance torch.Tensor
        """
        return self.correct / self.total


def string_match(pred_data: torch.Tensor, true_data: torch.Tensor) -> float:
    """Calculates precision of pedicted word.

    :param pred_data: tensor with predicted values
    :param true_data: tesnory with true values
    :return: precision float value
    """
    pred_data = pred_data.permute(1, 0, 2)
    pred_data = torch.Tensor.argmax(pred_data, dim=2).detach().cpu().numpy()

    true_data = true_data.detach().cpu().numpy()

    valid = 0
    for j in range(pred_data.shape[0]):
        p3 = [k for k, g in itertools.groupby(pred_data[j])]
        p3 = [k for k in p3 if k > 0]
        t = [k for k in true_data[j] if k > 0]
        valid += float(np.array_equal(p3, t))

    return valid / pred_data.shape[0]


def edit_distance(pred_data: torch.Tensor, true_data: torch.Tensor) -> float:
    """Compute the edit distance between true and predicted values.

    :param pred_data: predicted tensor of word
    :param true_data: true tensor of word
    :return: edit distance
    """
    pred_data = pred_data.permute(1, 0, 2)
    pred_data = torch.Tensor.argmax(pred_data, dim=2).detach().cpu().numpy()

    true_data = true_data.detach().cpu().numpy()

    dist = 0
    for j in range(pred_data.shape[0]):
        p3 = [k for k, g in itertools.groupby(pred_data[j])]
        p3 = [k for k in p3 if k > 0]
        t = [k for k in true_data[j] if k > 0]

        s_pred = "".join(list(map(lambda x: chr(x), p3)))
        s_true = "".join(list(map(lambda x: chr(x), t)))

        dist += ed(s_pred, s_true)

    return dist / pred_data.shape[0]
