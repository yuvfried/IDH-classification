from abc import ABC, abstractmethod
from typing import Union

import torch
from sklearn import metrics
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _plot(
        x,
        y,
        thresholds,
        best_point,
        best_threshold,
        no_skill_points,
        x_name,
        y_name,
        auc_score,
        curve_name,
        backend="plotly"
):
    # if backend == "matplotlib":
    #     display = RocCurveDisplay(fpr=self.fpr, tpr=self.tpr, roc_auc=self.auc())
    #     out_plot = display.plot()
    #
    #     plt.scatter(x=best_point[0], y=best_point[1], c='orange', marker="*", s=200)
    #
    #     return out_plot

    if backend == "plotly":
        # roc curve
        df_to_plot = pd.DataFrame({
            x_name: x,
            y_name: y,
            'threshold': thresholds
        })
        fig = px.line(
            data_frame=df_to_plot, x=x_name, y=y_name,
            hover_data=[x_name, y_name, 'threshold'],
            title=f'{curve_name} Curve (AUC={auc_score:.2f})',
            # labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )

        # diagonal
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=no_skill_points[0][0], x1=no_skill_points[1][0],
            y0=no_skill_points[0][1], y1=no_skill_points[1][1]
        )

        # best threshold
        fig.add_trace(
            go.Scatter(
                x=best_point[0], y=best_point[1], mode="markers",
                marker=dict(size=10, symbol='star'),
                name=f"Best threshold: {best_threshold:.2f}", hoverinfo='skip'
            )
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        return fig
    else:
        raise NotImplementedError


class CurveMetric(ABC):
    def __init__(self):
        self.probs = None
        self.targets = None
        self.thresholds = None
        self.tpr = None  # called also recall

    @abstractmethod
    def compute(self, targets, probs):
        self.targets = targets
        self.probs = probs

    @abstractmethod
    def auc(self, **kwargs):
        pass

    @abstractmethod
    def best_point_idx(self, *args, **kwargs):
        pass

    @abstractmethod
    def best_threshold(self, *args, **kwargs):
        pass


class RocCurve(CurveMetric):
    def __init__(self):
        super().__init__()
        self.fpr = None

    def compute(self, targets, probs):
        super().compute(targets, probs)
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(targets, probs)

    def auc(self, **kwargs):
        return metrics.roc_auc_score(self.targets, self.probs, **kwargs)

    def best_point_idx(self, method: str = "G-mean"):
        if method == "l2-norm":
            return np.argmin([
                np.linalg.norm(np.array([self.fpr[i], self.tpr[i]]) - np.array([0, 1]))
                for i in range(len(self.thresholds))
            ])
        elif method == "G-mean":
            return np.argmax(self.tpr - self.fpr)  # Youden’s J statistic

    def best_threshold(self, method: str = "G-mean"):
        """
        Returns the best threshold. Calculates ROC curve, finds the threshold that is responsible,
        for the best (fpr,tpr) point, and returns this threshold.
        There are different method to determine the best (fpr, tpr) point:
        See https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/

        The best (fpr, tpr) point
        is assumed to be the nearest point to (0,1) in the ROC curve.
        :param method: One of ["l2-norm", "g-mean"]
            * "l2-norm": The nearest point to the point (0,1).
            * "G-mean": The geometric mean of sensitivity and specificity.
        """
        return float(self.thresholds[self.best_point_idx(method)])

    def plot(self, backend="plotly"):
        best_idx = self.best_point_idx()
        best_point = [self.fpr[best_idx]], [self.tpr[best_idx]]
        best_threshold = self.thresholds[best_idx]
        return _plot(
            x=self.fpr,
            y=self.tpr,
            thresholds=self.thresholds,
            best_point=best_point,
            best_threshold=best_threshold,
            no_skill_points=((0, 0), (1, 1)),
            x_name='False Positive Rate',
            y_name='True Positive Rate',
            auc_score=self.auc(),
            curve_name='ROC',
        )


class PrecisionRecallCurve(CurveMetric):
    def __init__(self):
        super().__init__()
        self.precision = None

    def compute(self, targets, probs):
        super().compute(targets, probs)
        self.precision, self.tpr, self.thresholds = metrics.precision_recall_curve(targets, probs)

    def auc(self):
        return metrics.average_precision_score(self.targets, self.probs)

    def best_point_idx(self, min_recall: Union[float, None] = 0.9):
        """

        :param min_recall: set a fix minimum recall score, and choose the threshold which gives
        the best precision within this recall range. If None, threshold that maximizes F1 score is chosen.
        """
        if min_recall is not None:
            best_precision = self.precision[self.tpr > min_recall].max()
            best_idx = np.argwhere(self.precision == best_precision).flatten()[0]
        else:
            f1_scores = 2 * self.precision * self.tpr / (self.precision + self.tpr)
            best_idx = f1_scores.argmax()
        return best_idx

    def best_threshold(self, min_recall=0.9):
        return self.thresholds[self.best_point_idx(min_recall)]

    def plot(self, backend="plotly"):
        best_idx = self.best_point_idx()
        best_point = [self.tpr[best_idx]], [self.precision[best_idx]]
        best_threshold = self.thresholds[best_idx]
        pos_prop = (self.targets == 1).mean()

        return _plot(
            x=self.tpr,
            y=self.precision,
            thresholds=self.thresholds,
            best_point=best_point,
            best_threshold=best_threshold,
            no_skill_points=((0, pos_prop), (1, pos_prop)),
            x_name='Recall',
            y_name='Precision',
            auc_score=self.auc(),
            curve_name='PR',
        )


def specificity(y_true, y_pred, zero_division=0):
    negatives = (y_true == 0).sum()
    if negatives == 0 and zero_division == 0:
        return 0
    score = (y_pred[y_true == 0] == 0).sum() / negatives
    if not isinstance(score, float):
        score = float(score)
    return score


class ThresholdMetrics:
    """
    Metrics that gets predictions (not probs) as input, thus they depend on threshold.
    see https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification
    """

    METRIC_NAME_MAP = {
        "accuracy": metrics.accuracy_score,
        "recall": metrics.recall_score,
        "precision": metrics.precision_score,
        "f1": metrics.f1_score,
        "specificity": specificity,
        "balanced_accuracy": metrics.balanced_accuracy_score
    }

    def __init__(self):
        self.targets = None
        self.threshold = None
        self.preds = None
        self.__conf_matrix = None

    def compute(self, targets, probs, threshold):
        self.targets = targets
        self.threshold = threshold
        self.preds = torch.gt(probs, threshold).type(torch.int8)

    def confusion_matrix(self):
        if self.__conf_matrix is None:
            self.__conf_matrix = metrics.confusion_matrix(self.targets, self.preds)
        return self.__conf_matrix

    @staticmethod
    def __score_func(metric: str, targets, preds):
        if metric in ["accuracy", "balanced_accuracy"]:
            kwargs = {}
        else:
            kwargs = dict(zero_division=0)
        return ThresholdMetrics.METRIC_NAME_MAP[metric](targets, preds, **kwargs)

    def score(self, metric: str):
        return self.__score_func(metric, self.targets, self.preds)

