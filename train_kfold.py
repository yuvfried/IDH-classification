import argparse
import json
import os
from time import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.tensorboard import SummaryWriter

from mil.models import ABMIL
from mil.data.dataset import SlideBagDataset, CaseBagDataset, EmbeddingLoaderFactory
from mil.train import Trainer, EarlyStopping
from mil.utils import seed_everything


def plot_roc_kfold(cv_output, out_path):
    """
    Plots onw figure displaying all fold's ROC curves, with another curve representing the mean curve.
    """
    fprs = [cv_output[i].best_val_logger.roc_curve.fpr for i in range(len(cv_output))]
    tprs = [cv_output[i].best_val_logger.roc_curve.tpr for i in range(len(cv_output))]
    auc_scores = [cv_output[i].best_val_logger.roc_curve.auc() for i in range(len(cv_output))]
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
    # Calculate the standard deviation of the TPR values at each FPR point
    tpr_stds = np.std([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    # Create the plot
    plt.figure(figsize=(8, 6))

    # ROC Curves for each fold (with mid opacity level)
    for i in range(len(cv_output)):
        plt.plot(fprs[i], tprs[i], label=f'Fold {i + 1} (AUC = {auc_scores[i]:.2f})', alpha=0.5)

    # Mean ROC curve (thicker line)
    plt.plot(mean_fpr, mean_tpr, color='b', linewidth=3, label=f'Mean ROC (AUC = {mean_auc:.2f}±{std_auc:.2f})')

    # Shaded region for standard deviation
    plt.fill_between(mean_fpr, mean_tpr - tpr_stds, mean_tpr + tpr_stds, color='b', alpha=0.2,
                     label='Standard Deviation')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Naive Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of KFold Cross Validation')
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.savefig(out_path)
    plt.close()
    # return mean_fpr, mean_tpr, mean_auc


def train_fold(
        train_loader,
        val_loader,
        model_kwargs,
        trainer_kwargs,
        writer: SummaryWriter,
        early_stopping: bool = False,
        lr=0.005,
        weight_decay=0.001,
        epochs: int = 100,
):
    # os.makedirs(results_path, exist_ok=True)

    model = ABMIL(**model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if early_stopping:
        early_stopping = EarlyStopping(patience=20, min_epochs=20, min_delta=1e-3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        writer=writer,
        early_stopping=None if not early_stopping else early_stopping,
        **trainer_kwargs
    )

    # train
    history = trainer.train(train_loader, val_loader, epochs)

    return history

def parse_args():
    parser = argparse.ArgumentParser(description='Attention Based Deep MIL training')
    parser.add_argument('--csv', required=True,
                        help="cols ['id', 'path', 'target', 'case_id'] "
                             "'id' is slide id, 'path' is embedding path, 'target' is the label."
                                    "'case_id is an optional column. If it is passed, "
                                    "then GroupStratified KFold CV is performed which means "
                                    "same case_id cannot appear "
                                    "both in train and val sets in the same fold")
    parser.add_argument("--k", type=int, default=3, help="number of folds for CV")
    parser.add_argument('--track_samples', action='store_true',
                        help="output csv for each epoch with its predictions")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--early-stopping', action='store_true', default=False,
                        help='enables early stopping')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='path to store ckpt and scores.')
    parser.add_argument('--cache', action='store_true', default=False,
                        help="Pre load embedding vectors to RAM before training")

    # hyperparameters

    # input
    parser.add_argument('--format', type=str, default='pt',
                        choices=EmbeddingLoaderFactory.list_loaders(), help='Format of embedding files')
    parser.add_argument(
        '--embedding_size',type=int, default=1024,
        help="Embedding size of pretrained output vector. "
             "Default is 1024 which fits UNI-v1 embeddings")
    parser.add_argument('--bag_size', type=int, default=None)
    parser.add_argument('--case_bag', action='store_true', default=False,
                        help="Each bag corresponds to case_id with instances from multiple WSIs")

    # model arch
    parser.add_argument('--mlp_dim', default=128, type=int,
                        help="MLP layer size before ABMIL")
    parser.add_argument('--D', type=int, default=128,
                        help='hidden dim of attention')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout value. default is 0.0 (no dropout)')
    parser.add_argument('--no-gate', action='store_true', default=False,
                        help='Use this flag to turn off gated attention.')

    # training
    parser.add_argument('-w_loss', action="store_true", default=False,
                        help='Use class distribution of dataset as weights for loss')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay. Default 0.0001')
    parser.add_argument('--t', type=float, default=1,
                        help='temperature value (default = 1) for softmax over attention scores')
    parser.add_argument('--l2', type=float, default=None,
                        help="Add L2 regularization term for classification loss. This value will"
                             "multiply the sum of squares of Attention scores, forcing them to shrinkage.")
    parser.add_argument('--accum', type=int, default=1,
                        help='Gradient Accumulation steps. How many bags will form a batch'
                             'for taking one back-prop step. This could emulate a batch even '
                             'though only one bag can be forwarded through the model at a time due to '
                             'differences in bag sizes. Default is 1 meaning no accumulation steps are taken.')
    parser.add_argument('--threshold', default='ROC', choices=['half','ROC', 'PR'],
                        help='Method for picking threshold: `half` for hard-coding 0.5 as a threshold,'
                             '`ROC` for optimizing Receiver Operating Characteristic curve or '
                             '`PR` for optimizing Precision Recall curve')

    args = parser.parse_args()

    return args

def main(args):
    # determine device
    device = torch.device('cpu')
    if not args.no_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('\nGPU is ON')
        else:
            print("Using CPU since CUDA is not available")
    else:
        print("GPU is OFF, Using CPU")

    # Create a root folder SummaryWriter to log data for TensorBoard
    run_name = os.path.basename(args.csv).split(".")[0] # name of run determined by csv filename
    run_datetime = pd.to_datetime('today').strftime('%Y-%m-%d_%H-%M-%S')
    run_root: str = os.path.join(args.results_dir, run_name, run_datetime) # this will serve as the root path for all run artifacts

    # config logging
    os.makedirs(run_root, exist_ok=True)
    with open(os.path.join(run_root, "config.json"), 'w') as jf:
        json.dump(vars(args), jf)
        print(json.dumps(vars(args), indent=4))

    # set seed
    seed_everything(args.seed)

    # read training data
    df_slides = pd.read_csv(args.csv, index_col=False)
    # enforce required columns
    required_cols = {'id', 'case_id', 'path', 'target'}
    diff_cols = required_cols.difference(set(df_slides.columns.tolist()))
    if len(diff_cols) > 0:
        raise ValueError(f"The following required columns are not present in the dataset: {diff_cols}")

    # print data stats
    df_slides.to_csv(os.path.join(run_root, "slides.csv"), index=False)
    print("Slide counts:\n", df_slides['target'].value_counts().to_dict())
    print("Case counts:\n", df_slides.groupby('target')['case_id'].nunique().to_dict())

    # init ds
    if args.case_bag:
        ds_cls = CaseBagDataset
    else:
        ds_cls = SlideBagDataset
    ds = ds_cls(df_slides, args.bag_size, args.cache, args.format)

    # weighted loss - deal with imbalanced data
    if args.w_loss:
        weights = ds.balance().to(device)
        print(f"Using dataset balance as loss weights: {weights.item()}")
    else:
        weights = None

    model_kwargs = {
        "L": args.embedding_size,
        "D":args.D,
        "mlp_dim": args.mlp_dim,
        "dropout": args.dropout,
        "gated": not args.no_gate,
        "temperature": args.t
    }

    trainer_kwargs = {
        "device": device,
        "accumulation_steps": args.accum,
        "threshold_method": args.threshold,
        "track_samples": args.track_samples,
        "loss_weights": weights,
    }

    kfold = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    cv_output = list()

    for fold, (train_ids, val_ids) in enumerate(kfold.split(ds, y=ds.targets, groups=ds.groups)):
        print(f'FOLD {fold}\n--------------------------------')

        # folder to store fold's output
        fold_path = os.path.join(run_root, f"fold{fold}")
        fold_writer = SummaryWriter(fold_path)

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        # Note that in our setting each batch consists of a single bag.
        train_loader, val_loader = [DataLoader(
            ds, batch_size=1, sampler=sampler, num_workers=args.workers, pin_memory=True)
            for sampler in [train_subsampler, val_subsampler]]

        history = train_fold(
            train_loader,
            val_loader,
            model_kwargs,
            trainer_kwargs,
            writer=fold_writer,
            early_stopping=args.early_stopping, # default is False
            lr=args.lr,
            weight_decay=args.wd,
            epochs=args.epochs
        )

        # save fold output
        cv_output.append(history)
        with open(os.path.join(fold_path, f"results.json"), 'w') as jf:
            json.dump(history.to_dict(), jf)
        # TODO: should be replaced with some tensorboard.
        #  Possibly other lines of code in this module should be replaced by it
        history.plot(fold_path, title_prefix=f"fold {fold}")

    # save ROC plot
    plot_roc_kfold(cv_output, os.path.join(run_root, "ROC.png"))

    # plot kfold score for each metric in each epoch
    for metric_name in ['loss','auc','balanced_accuracy', 'f1']:
        dfs = list()
        for i in range(args.epochs):
            # metric score of all folds in epoch=i
            metric_score_list = [
                cv_output[fold_idx].to_dict()['scores'][i]['valid'][metric_name]
                for fold_idx in range(args.k)]

            dfs.append(pd.DataFrame({
                'values': metric_score_list,
                'fold': list(range(args.k)),
                'epoch': i
            }))

        df_metric = pd.concat(dfs)
        df_metric[['fold', 'epoch']] = df_metric[['fold', 'epoch']].astype(int)
        # plot line for each fold
        sns.lineplot(df_metric, x='epoch', y='values', hue='fold',
                     palette=sns.color_palette()[:args.k], alpha=0.5, linestyle='--')
        # plot the mean score across all fold for each epoch with errorbar
        sns.lineplot(df_metric, x='epoch', y='values', color='black', linewidth=3, label=f'Mean')
        mean_score = df_metric.loc[df_metric['epoch'] == args.epochs-1, 'values'].mean()
        plt.title(f"{metric_name} mean at epoch {args.epochs} = {mean_score:.2f}")
        plt.savefig(os.path.join(run_root, metric_name + ".png"))
        plt.close()

    # print mean results
    best_valid_scores = [
        cv_output[i].scores[cv_output[i].best_iteration]['valid']
        for i in range(len(cv_output))
    ]
    kfold_mean_results = pd.DataFrame.from_records(best_valid_scores).mean()
    print("mean Kfold CV scores\n",
          kfold_mean_results.to_json(indent=4))

    print(f"saved results to {run_name}")

if __name__ == '__main__':
    start = time()
    args = parse_args()
    main(args)
    len_in_secs = time() - start
    print(f"Completed in {int(len_in_secs // 3600):02d}h "
          f"{int(len_in_secs % 3600 // 60):02d}m {int(len_in_secs % 60):02d}s")


