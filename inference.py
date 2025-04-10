import argparse
import json
import os

import h5py
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, f1_score, log_loss

from mil.heatmap import plot_heatmap
from mil.predict import make_predictions
from mil.utils import seed_everything, min_max


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True,
                        help="cols ['id', 'path', 'target', 'case_id'] "
                             "'id' is slide id, 'path' is embedding path, 'target' is the label."
                                    "'case_id is an optional column. If it is passed, "
                                    "then GroupStratified KFold CV is performed which means "
                                    "same case_id cannot appear "
                                    "both in train and val sets in the same fold")
    parser.add_argument("--results_dir", required=True, type=str,
                        help="Path to the results directory of KFold training")
    parser.add_argument('--target_root', required=True, help="root for outputs")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold index of model for inference')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--case_bag', action='store_true', default=False,
                        help="Each bag corresponds to case_id with instances from multiple WSIs")
    parser.add_argument('--no_attn', action='store_true', default=False,
                        help="Disable saving attention scores. Use it when there is low disk space or RAM")
    parser.add_argument('--heatmap', action='store_true',
                        help="Extract attention heatmaps for predicted slides")
    # required if --heatmap
    parser.add_argument('--patches', type=str,
                        help="Path to dir of patches in h5 format")
    parser.add_argument('--raw', type=str,
                        help="Path to dir of raw WSIs")
    parser.add_argument('--ext', type=str, default='tiff')

    args = parser.parse_args()
    return args

def main(args):
    print("Config:")
    print(json.dumps(vars(args), indent=4))

    # set seed
    seed_everything(args.seed)

    # read data
    df_slides = pd.read_csv(args.csv, index_col=False)
    # enforce required columns
    required_cols = {'id', 'case_id', 'path', 'target'}
    diff_cols = required_cols.difference(set(df_slides.columns.tolist()))
    if len(diff_cols) > 0:
        raise ValueError(f"The following required columns are not present in the dataset: {diff_cols}")

    # create output dir
    os.makedirs(args.target_root, exist_ok=True)

    # print data stats
    df_slides.to_csv(os.path.join(args.target_root, "slides.csv"), index=False)
    print("Slide counts:\n", df_slides['target'].value_counts().to_dict())
    print("Case counts:\n", df_slides.groupby('target')['case_id'].nunique().to_dict())

    ckpt_path = os.path.join(args.results_dir, f"fold{args.fold}", f'checkpoint.pth.tar')
    config_path = os.path.join(args.results_dir, "config.json")
    df_eval, d_attn = make_predictions(df_slides, ckpt_path, config_path, args.device, return_attention=True)
    df_eval.to_csv(os.path.join(args.target_root, "predictions.csv"), index=False)
    np.save(os.path.join(args.target_root, "attention_scores.npy"), d_attn)


    # Print results
    for m in [log_loss, roc_auc_score]:
        print(m.__name__)
        print(m(df_eval['target'].values, df_eval['prob'].values))
        print()

    for m in [balanced_accuracy_score, f1_score, confusion_matrix]:
        print(m.__name__)
        print(m(df_eval['target'].values, df_eval['pred'].values))
        print()


    # Create heatmaps
    if args.heatmap:
        heatmap_root = os.path.join(args.target_root, 'heatmaps')
        os.makedirs(heatmap_root, exist_ok=True)
        for i in tqdm(range(len(df_eval)), desc="Create heatmaps"):
            slide_id = df_eval.iloc[i]['id']

            # load WSI and coords
            try:
                wsi = openslide.open_slide(os.path.join(args.raw, f"{slide_id}.{args.ext}"))
                with h5py.File(os.path.join(args.patches, f"{slide_id}.h5"), 'r') as f:
                    coords = f['coords'][()]
            except Exception as e:
                print(f"\nProblem loading {slide_id}: {type(e).__name__}: {str(e)}")
                continue

            # Plot
            scaled_attention_scores = min_max(d_attn[slide_id])  # A scaled to be within [0,1]
            plotly_heatmap = plot_heatmap(scaled_attention_scores, coords, wsi.level_dimensions[0], title=slide_id)
            plotly_heatmap.write_html(os.path.join(heatmap_root, f'{slide_id}.html'))


    print(f"Results saved to {args.target_root}")

if __name__=="__main__":
    # TODO unify this script with heatmap.py
    args = parse_args()
    main(args)