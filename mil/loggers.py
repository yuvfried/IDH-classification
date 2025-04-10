import os
from collections import UserDict
from typing import Dict, Union, List, Any

import torch
import numpy as np
import pandas as pd

from mil.metrics import PrecisionRecallCurve, RocCurve, ThresholdMetrics

def confusion_matrix_to_df(mat: np.ndarray):
    return pd.DataFrame(
        mat,
        index=['actual_0', 'actual_1'],
        columns=['pred_0', 'pred_1'])


def print_verbose_conf_mat(conf_mat):
    print(
        f"TN: {conf_mat[0, 0]}/{conf_mat.sum(axis=1)[0]}",
        f"TP: {conf_mat[1, 1]}/{conf_mat.sum(axis=1)[1]}"
    )



class EpochLogger:
    """
    Log metrics of epoch based on cumulative targets, probs and losses.
    After each step of training use the method `log`. When you want to output epoch metrics
    call `compute`. This class also has two attributes: `roc_curve` and `metrics` that have
    additional utilities in their APIs.
    """

    def __init__(self):
        self.__losses = list()
        self.__targets = list()
        self.__samples = list()
        self.__probs = list()
        self.attention_scores = list()
        self.coords_list = list()
        self.roc_curve = RocCurve()
        self.pr_curve = PrecisionRecallCurve()
        self.t_metrics = ThresholdMetrics()

    def log(self, target, prob, loss, sample=None, A=None, coords=None):
        self.__losses.append(loss)
        self.__targets.append(target)
        self.__probs.append(prob)
        if sample is not None:
            self.__samples.append(sample)
        if A is not None:
            self.attention_scores.append(A)
        if coords is not None:
            self.coords_list.append(coords)

    def compute(self, threshold: float = None, threshold_method='PR') -> Dict[str, float]:
        """
        Freeze logger on epoch end and output summary of metrics.
        :param threshold: specify threshold to calculate metrics
        that depend on threshold (accuracy F1 etc...). Leave None in train mode -
        the logger will take the best threshold as calculated by ROC curve.
        On inference mode - load a predefined threshold that was optimized in training.
        :return: dict with metric names and values.
        """
        self.__targets = torch.tensor(self.__targets).type(torch.uint8)
        self.__probs = torch.tensor(self.__probs)
        self.roc_curve.compute(self.__targets, self.__probs)
        self.pr_curve.compute(self.__targets, self.__probs)
        if threshold is None:
            if threshold_method == 'half':
                threshold = 0.5
            elif threshold_method == 'ROC':
                threshold = self.roc_curve.best_threshold()
            elif threshold_method == 'PR':
                threshold = self.pr_curve.best_threshold()
            else:
                raise NotImplementedError(f"Threshold method {threshold_method} is not implemented.")
        self.t_metrics.compute(self.__targets, self.__probs, threshold)

        scores = dict(
            loss=torch.tensor(self.__losses).mean().item(),
            auc=self.roc_curve.auc(),
        )
        # add threshold metric scores
        scores.update({
            m: getattr(self.t_metrics, 'score')(m) for m in self.t_metrics.METRIC_NAME_MAP.keys()
        })

        return scores

    def per_sample_info(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Output a verbose dict of samples as keys and their model output as vals.
        Output contains target, pred and score.
        :return: {sample: {'target': val1, 'pred': val2, 'score': val3}, ...}
        """
        # check validity
        if self.__samples is None:
            raise ValueError('cannot output raw summary of preds when track samples disabled')
        if self.preds is None:
            raise ValueError('cannot output raw summary of preds before calling "compute"')

        return {
            sample:
                {
                    'loss': loss.item(),
                    'target': target.item(),
                    'prob': prob.item(),
                    'pred': pred.item()
                }
            for sample, loss, target, prob, pred in
            zip(self.__samples, self.__losses, self.targets, self.probs, self.preds)
        }

    def per_sample_attention(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            self.__samples[i]: {'A': self.attention_scores[i], 'coords': self.coords_list[i]}
            for i in range(len(self.__samples))
        }

    @property
    def targets(self):
        return self.__targets

    @property
    def probs(self):
        return self.__probs

    @property
    def preds(self):
        return self.t_metrics.preds



# TODO
class EpochCallback(UserDict):
    pass


class History:
    # [{'train':{'scores':{'loss': 0.8, ...}, ...}, {'valid': {'scores':{'loss': 0.8, ...}, ...}]

    def __init__(self, callbacks: List[Dict[str, Dict[str, Any]]], best_iteration: int = -1):
        self.__best_iteration = best_iteration

        self.__scores, self.__samples, self.__roc_plots = self.__clean(callbacks)
        self.__best_val_logger = None
        if 'valid' in callbacks[0].keys():
            self.__best_val_logger = callbacks[best_iteration]['valid']['logger']

    @staticmethod
    def __clean(callbacks):
        phases = list(callbacks[0].keys())
        is_samples = 'samples' in callbacks[0][phases[0]].keys()
        is_roc = 'roc_plot' in callbacks[0][phases[0]].keys()
        scores = list()
        roc_plots = list()
        samples = list()
        for i in range(len(callbacks)):
            scores.append({
                phase: callbacks[i][phase]['scores'] for phase in phases
            })
            if is_samples:
                samples.append({
                    phase: callbacks[i][phase]['samples'] for phase in phases
                })
            if is_roc:
                roc_plots.append({
                    phase: callbacks[i][phase]['roc_plot'] for phase in phases
                })

        return scores, samples, roc_plots

    def plot(self, path, metrics=None, title_prefix=""):
        if metrics is None:
            metrics = ["loss", "auc"]
        for metric in metrics:
            fig = pd.DataFrame(
                {
                    phase: [self.scores[i][phase][metric] for i in range(len(self.scores))]
                    for phase in self.scores[0].keys()
                }
            ).plot(backend="plotly", title=f"{title_prefix} {metric}")
            fig.write_html(os.path.join(path, f"{metric}.html"))

    def log_to_wandb(self, run, fold_title_prefix=""):
        for epoch_num in range(len(self.scores)):
            to_log = {f"{fold_title_prefix}{phase}_{metric}": self.scores[epoch_num][phase][metric]
                      for phase in self.scores[epoch_num].keys()
                      for metric in self.scores[epoch_num][phase].keys()}
            run.log(to_log)

    def to_dict(self):
        d_out = {'best_iteration': self.best_iteration}
        d_out.update({attr: getattr(self, attr) for attr in ['scores', 'samples']})
        return d_out

    @property
    def scores(self):
        return self.__scores

    @property
    def samples(self):
        return self.__samples

    @property
    def roc_plots(self):
        return self.__roc_plots

    @property
    def best_iteration(self):
        return self.__best_iteration

    @property
    def best_val_logger(self):
        return self.__best_val_logger