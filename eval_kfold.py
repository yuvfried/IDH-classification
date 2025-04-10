import argparse
import json
import os

import h5py
import openslide
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from mil.heatmap import plot_heatmap
from mil.predict import make_predictions
from mil.utils import min_max


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=str,
                        help="Path to the results directory of KFold training")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--patches', type=str, required=True,
                        help="Path to dir of patches in h5 format")
    parser.add_argument('--raw', type=str, required=True,
                        help="Path to dir of raw WSIs")
    parser.add_argument('--ext', type=str, default='tiff')

    args = parser.parse_args()
    return args

def search_tcga_slide_path(root, slide_id):
    """
    Slide paths of TCGA are typically added with another file if after identifier. This method
    helps to handle it.
    """
    for p in os.scandir(root):
        if p.name.startswith(slide_id):
            return p.path

def main(args):
    df_slides = pd.read_csv(os.path.join(args.results_dir, 'slides.csv'))

    # infer k from dirs in results_dir
    k = len([f for f in os.listdir(args.results_dir) if f.startswith('fold')])
    print(f"{k} folds")

    # Loop for evaluation of each validation set for each fold
    target_root = os.path.join(args.results_dir, 'evaluations')
    eval_list = list()  # saves predictions of all samples
    for fold_idx in range(k):
        print(f"\n === FOLD {fold_idx}")

        # load validation dataset
        with open(os.path.join(args.results_dir, f"fold{fold_idx}", "results.json"), 'r') as jf:
            fold_results = json.load(jf)
        valid_samples = fold_results['samples'][-1]['valid'].keys()
        df_fold = df_slides[df_slides['id'].isin(valid_samples)]
        fold_output_dir = os.path.join(target_root, f'fold{fold_idx}')

        ckpt_path = os.path.join(args.results_dir, f"fold{fold_idx}", f'checkpoint.pth.tar')
        config_path = os.path.join(args.results_dir, "config.json")

        # Predict this fold
        df_eval_fold, d_attn = make_predictions(
            df_fold, ckpt_path, config_path, args.device, return_attention=True)
        df_eval_fold['fold'] = fold_idx
        eval_list.append(df_eval_fold)

        # Extract plots for every slide_id
        for i in tqdm(range(len(df_eval_fold)), desc="Create Heatmaps"):
            slide_id = df_eval_fold.iloc[i]['id']

            # load WSI and coords
            try:
                if args.ext == 'tiff':
                    slide_path = os.path.join(args.raw, f"{slide_id}.{args.ext}")
                elif args.ext == 'svs':
                    slide_path = search_tcga_slide_path(args.raw, slide_id)
                else:
                    raise ValueError(f"Unknown extension {args.ext}")
                wsi = openslide.open_slide(slide_path)
                coords_path = df_fold.loc[df_fold['id'] == slide_id, 'path'].item()
                # with h5py.File(os.path.join(args.patches, f"{slide_id}.h5"), 'r') as f:
                with h5py.File(coords_path, 'r') as f:
                    coords = f['coords'][()]

            except Exception as e:
                print(f"\nProblem loading {slide_id}: {type(e).__name__}: {str(e)}")
                continue

            # Create output folder
            slide_output_dir = os.path.join(fold_output_dir, slide_id)
            os.makedirs(slide_output_dir, exist_ok=True)

            # Histogram of raw attention_scores
            fig, ax = plt.subplots()
            ax.hist(d_attn[slide_id], bins=1000, density=True)
            fig.savefig(os.path.join(slide_output_dir, f"attention_weights.png"))
            plt.close(fig)

            # Create heatmap
            try:
                scaled_attention_scores = min_max(d_attn[slide_id]) # A scaled to be within [0,1]
                plotly_heatmap = plot_heatmap(scaled_attention_scores, coords, wsi.level_dimensions[0], title=slide_id)
                plotly_heatmap.write_html(os.path.join(slide_output_dir, 'plotly_heatmap.html'))
            except Exception as e:
                print(f"\nProblem Plotting heatmap for {slide_id}: {type(e).__name__}: {str(e)}")
                continue

            # Show most attended patches
            TOP_K = 12
            NCOLS = 6
            nrows = TOP_K // NCOLS + bool(TOP_K % NCOLS)

            highest_scores, highest_indices = torch.topk(torch.from_numpy(d_attn[slide_id]), k=TOP_K)
            highest_coords = coords[highest_indices]

            # Create a 2x6 grid of subplots
            fig, axes = plt.subplots(nrows, NCOLS, figsize=(15, 6))
            # Set a main title for the entire figure
            target = df_eval_fold.iloc[i]['target']
            prob = df_eval_fold.iloc[i]['prob']
            pred = df_eval_fold.iloc[i]['pred']
            fig.suptitle(
                f'{slide_id} MG {int(target)} fold={fold_idx}\n'
                f'prob={prob:.3f}, pred={int(pred)}\n',
                fontsize=16
            )

            # plot attended patches
            for col_ind in range(TOP_K):
                global_index = col_ind
                if col_ind < axes.shape[1]:
                    row_ind = 0
                else:
                    row_ind = 1
                    col_ind = col_ind % axes.shape[1]

                current_coords = highest_coords[global_index]
                patch = wsi.read_region(
                    location=current_coords, level=1, size=(256,256))  # read patch
                axes[row_ind, col_ind].set_title(
                    f'coords = {current_coords}\nscore={highest_scores[global_index]:.3f}')
                axes[row_ind, col_ind].set_xticks([]) # Remove axes
                axes[row_ind, col_ind].set_yticks([])
                axes[row_ind, col_ind].imshow(patch)

            # Adjust layout for better spacing
            plt.tight_layout(pad=1, w_pad=0, h_pad=5)
            fig.savefig(os.path.join(slide_output_dir, f"attended_patches.png"))
            plt.close(fig)

    # Save all predictions as a single df
    pd.concat(eval_list, ignore_index=True).to_csv(
        os.path.join(target_root, f"df_eval.csv"), index=False)


if __name__=="__main__":
    args = parse_args()
    main(args)
