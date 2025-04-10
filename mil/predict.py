import json
from typing import Tuple

import pandas as pd
import torch
from tqdm import tqdm

from mil.data.dataset import SlideBagDataset
from mil.models import ABMIL


def load_pretrained_abmil(ckpt_path:str, config_path:str, device='cuda') -> Tuple[ABMIL, float]:
    """
    Loads a pre-trained ABMIL model from a checkpoint and configuration file,
    sets it to evaluation mode, and moves it to the specified device (e.g., GPU or CPU).

    Parameters:

    ckpt_path (str):
    Path to the checkpoint file containing the model's saved state and additional metadata: `best epoch` and `threshold`.
    Typically, it's the checkpoint that automatically output from training a KFold.

    config_path (str):
    Path to the JSON configuration file containing model parameters. required fields:
    `embedding size`, `mlp_dim`, `D`, `t`, `no_gate`, `l2`.

    device (str, optional, default='cuda'):
    The device on which to load the model (e.g., 'cuda' for GPU or 'cpu' for CPU).

    Returns:

    model (ABMIL):
    The ABMIL model, loaded with the pre-trained weights and set to evaluation mode.

    threshold (float):
    The threshold value retrieved from the checkpoint, which is used for model decision-making or classification.

    Raises:

    FileNotFoundError:
    If either the checkpoint file or configuration file cannot be found at the provided paths.

    KeyError:
    If the configuration file or checkpoint is missing required keys like 'embedding_size', 'mlp_dim', or 'state_dict'.
    """

    with open(config_path, 'r') as jf:
        config = json.load(jf)
    model = ABMIL(
        L=config['embedding_size'],
        mlp_dim=config['mlp_dim'],
        D=config['D'],
        temperature=config['t'],
        l2=config['l2'],
        gated=not config['no_gate']
    )

    ckpt = torch.load(ckpt_path, weights_only=True)
    threshold = ckpt['threshold']
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model = model.to(device)
    print('loaded ckpt\n', f"best epoch: {ckpt['best_epoch']}", f"threshold: {threshold:.2f}")

    return model, threshold

# TODO: What about the situation where the Dataset is CaseBagDataset?
#     # load validation dataset
#     if args.case_bag:
#         ds_cls = CaseBagDataset
#     else:
#         ds_cls = SlideBagDataset
#     ds = ds_cls(
#         df_slides,
#         bag_size=None   # TODO Understand what bag size should be in inference
#     )
def make_predictions(df_slides, ckpt_path, config_path, device, return_attention: bool = False):
    """
    Predicts `df_slides` using ABMIL model.

    df_slides (pandas.DataFrame):
    A DataFrame containing slide-level data with columns [id, target, case_id and path].

    ckpt_path (str):
    Path to the checkpoint file containing the model's saved state and additional metadata.
    Typically, it's the checkpoint that automatically output from training a KFold.
    Required fields:
    -  `best epoch`
    - `threshold`
    - `state_dict`

    config_path (str):
    Path to the JSON configuration file containing model parameters. Required fields in the configuration:
    - `embedding size`
    - `mlp_dim`
    - `D`
    - `t`
    - `no_gate`
    - `l2`

    device (str, optional, default='cuda'):
    The device on which to load the model (e.g., 'cuda' for GPU or 'cpu' for CPU).

    return_attention (bool, optional, default False):
    Whether to return the attention scores for each slide.
    If True, the function will return a dictionary of attention scores along with the predictions.
    """
    with open(config_path, 'r') as jf:
        config = json.load(jf)

    ds = SlideBagDataset(
        df_slides,
        bag_size=None,  # TODO Understand what bag size should be in inference
        file_format=config['format']
    )

    model, threshold = load_pretrained_abmil(ckpt_path, config_path, device)

    eval_list = list()
    attention_scores_dict = dict()
    for i in tqdm(range(len(ds)), desc="Predict"):
        item = ds[i]
        slide_id, embeddings = item['id'], item['embeddings']

        # Predict
        with torch.inference_mode():
            out = model(embeddings.to(device))  # batch size of 1
        prob = torch.nn.functional.sigmoid(out['logits'])
        pred = prob.gt(threshold).type(torch.uint8)

        # save predictions of this fold
        target = df_slides.loc[df_slides['id'] == slide_id, 'target'].item()
        case_id = df_slides.loc[df_slides['id'] == slide_id, 'case_id'].item()
        eval_list.append({
            'id': slide_id,
            'case_id': case_id,
            'target': target,
            'pred': pred.item(),
            'prob': prob.item()
        })

        if return_attention:
            attention_scores_dict[slide_id] = out['A'].squeeze().cpu().numpy()

    # Save all predictions as a single df
    df_eval = pd.DataFrame(eval_list)

    if return_attention:
        return df_eval, attention_scores_dict
    return df_eval