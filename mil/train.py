import os
from copy import deepcopy
from typing import Dict

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from mil.loggers import EpochLogger, History, confusion_matrix_to_df


# TODO replace with pytorch ignite? https://pytorch.org/ignite/generated/ignite.handlers.early_stopping.EarlyStopping.html
class EarlyStopping:

    def __init__(self, patience: int, min_epochs: int, min_delta: float = 1e-6):
        self.__patience: int = patience
        self.__min_epochs: int = min_epochs
        self.__min_delta: float = min_delta
        self.current_epoch: int = 0
        self.__best_iteration: int = -1
        self.__best_score: float = np.inf
        self.__triggered: int = 0
        self.__best_logger: EpochLogger = None
        self.__best_state_dict: Dict = None

    def update(self, score: float, logger: EpochLogger, state_dict: Dict) -> bool:
        if score < self.__best_score + self.__min_delta:
            self.__triggered = 0
            self.__best_score = score
            self.__best_iteration = self.current_epoch
            self.__best_logger = deepcopy(logger)
            self.__best_state_dict = deepcopy(state_dict)
            ret_val = True
        else:
            self.__triggered += 1
            ret_val = False
        self.current_epoch += 1
        return ret_val

    def stop(self) -> bool:
        return (self.current_epoch > self.__min_epochs + self.__patience) \
            and (self.__triggered > self.__patience)

    @property
    def best_score(self):
        return self.__best_score

    @property
    def best_iteration(self):
        return self.__best_iteration

    @property
    def best_logger(self) -> EpochLogger:
        return self.__best_logger

    @property
    def best_state_dict(self) -> Dict:
        return self.__best_state_dict


class Trainer:

    def __init__(self,
                 device,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 accumulation_steps=1,
                 loss_weights=None,
                 early_stopping: EarlyStopping = None,
                 threshold_method='ROC',
                 track_samples=False,
                 writer: SummaryWriter = None
                 ):

        self.device = device
        self.track_samples = track_samples
        self.writer = writer

        self.early_stopping = early_stopping

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weights)

        # model
        self.__model = model.to(self.device)

        # optimizer
        self.__optimizer = optimizer

        # batch_size
        self.accumulation_steps = accumulation_steps

        # threshold picking method
        self.threshold_method = threshold_method


    def preprocess_loss(self, logits, target):
        """
        process model outputs to be inserted to torch bce loss obj
        """
        return logits.view(-1, ), target.float().to(self.device)

    def y_prob(self, logits):
        return torch.sigmoid(logits)

    def y_hat(self, logits, threshold=0.5):
        return self.y_prob(logits).ge(threshold).long()

    def step(self, batch, logger):
        embeddings = batch['embeddings'].to(self.device)

        # forward
        out = self.model(embeddings)

        # loss
        logits = out['logits']
        target = batch['target']
        logits, target = self.preprocess_loss(logits, target)
        loss = self.criterion(logits, target)
        if out['l2_reg'] is not None:
            loss += out['l2_reg']

        # log
        to_log = {
            'sample': batch['id'][0],   # assumes batch_size=1
            "target": target,
            'prob': self.y_prob(logits),
            "loss": loss
            }
        logger.log(**to_log)

        return loss

    def train_epoch(self, loader, epoch):
        logger = EpochLogger()
        self.model.train()
        self.optimizer.zero_grad()
        for i, batch in enumerate(loader):
            loss = self.step(batch, logger)
            self.writer.add_scalar('train_loss_step', loss, epoch * len(loader) + i)
            loss /= self.accumulation_steps
            loss.backward()
            # accumulate gradients and performs back-prop only after `accumulation_steps` bags.
            # Mimic training in batches, which is currently impossible due to different bag sizes.
            if (i + 1) % self.accumulation_steps == 0:
                self.__optimizer.step()
                self.__optimizer.zero_grad()

        return logger

    def validate_epoch(self, loader, pbar=False):
        logger = EpochLogger()
        self.model.eval()
        with torch.inference_mode():
            iterator = loader
            if pbar:
                iterator = tqdm(loader)
            for batch in iterator:
                self.step(batch, logger)
        return logger


    def train(self, train_loader, val_loader=None, epochs: int = 200) -> History:
        if epochs < 1:
            raise ValueError(f"epochs == {epochs}, but should be a positive integer")

        history = list()  # list of epoch scores
        best_iteration = -1

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.__optimizer,
            num_warmup_steps=0,
            num_training_steps=epochs * len(train_loader),
        )

        # train and validate one epoch
        for epoch in range(epochs):

            epoch_callback = dict() # init containers for loggers

            # train one epoch
            epoch_callback['train'] = {'logger': self.train_epoch(train_loader, epoch)}
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            lr_scheduler.step()

            # validation
            if val_loader is not None:
                epoch_callback['valid'] = {'logger': self.validate_epoch(val_loader)}

            # compute epoch scores
            epoch_callback['train']['scores'] = epoch_callback['train']['logger'].compute(
                threshold_method=self.threshold_method)
            # fix threshold based on train scores
            threshold = epoch_callback['train']['logger'].t_metrics.threshold
            self.writer.add_scalar('threshold', threshold, epoch)
            # evaluate validation based on this threshold
            epoch_callback['valid']['scores'] = epoch_callback['valid']['logger'].compute(
                threshold=threshold)

            # save scores and artifacts
            for phase_name in ['train', 'valid']:
                epoch_callback[phase_name]['roc_plot'] = epoch_callback[phase_name]['logger'].roc_curve.plot()
                if self.track_samples:
                    epoch_callback[phase_name]['samples'] = epoch_callback[phase_name]['logger'].per_sample_info()


                for s in epoch_callback[phase_name]['scores'].keys():
                    self.writer.add_scalar(
                        f'{phase_name}_{s}', epoch_callback[phase_name]['scores'][s], epoch)

            self.print_on_epoch_end(epoch_callback, epoch)
            history.append(epoch_callback)

            # early stopping
            if self.early_stopping is not None and val_loader is not None:
                score = epoch_callback['valid']['scores']["loss"]
                logger = epoch_callback['valid']['logger']
                state_dict = self.model.state_dict()
                self.early_stopping.update(score, logger, state_dict)

                # stop training
                if self.early_stopping.stop():
                    best_iteration = self.early_stopping.best_iteration
                    self.__model.load_state_dict(self.early_stopping.best_state_dict)
                    self.print_on_early_stopping()
                    break

        # save checkpoint
        # This will save the checkpoint of the last epoch where early stopping is off or the best
        # epoch if it's on.
        torch.save(
            {
                'best_epoch': best_iteration,
                'threshold': history[best_iteration]['train']['logger'].t_metrics.threshold,
                'state_dict': self.model.state_dict(),
            },
            os.path.join(self.writer.log_dir, f"checkpoint.pth.tar")
        )
        self.print_on_train_end(history[best_iteration]['valid']['logger'])

        history = History(history, best_iteration=best_iteration)
        return history

    def print_on_epoch_end(self, epoch_callback, epoch_num):
        print("=" * 5, f"Epoch {epoch_num}")
        for k in epoch_callback.keys():
            print(
                f"{k}: loss={epoch_callback[k]['scores']['loss']:.4f}",
                f"AUC={epoch_callback[k]['scores']['auc']:.4f}"
            )

    def print_on_train_end(self, logger: EpochLogger):
        # print epoch conf_mat and AUC
        print(f"best AUC: {logger.roc_curve.auc():.2f}")
        print(f"Confusion Matrix: (threshold={logger.t_metrics.threshold:.2f})\n")
        print(confusion_matrix_to_df(logger.t_metrics.confusion_matrix()).to_markdown())
        print()

    def print_on_early_stopping(self):
        print(
            f"Early Stopping best iter:{self.early_stopping.best_iteration}, "
            f"best_score: {self.early_stopping.best_score:.2f}")


    def load_state_dict(self, state_dict):
        self.__model.load_state_dict(state_dict)

    @property
    def model(self):
        return self.__model

    @property
    def optimizer(self):
        return self.__optimizer
