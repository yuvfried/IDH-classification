# IDH Classification
An example of IDH mutation status classification in Glioma Whole Slide Images (WSI).

Here we use Attention Deep Multiple Instance Learning Based Classifier. See
[Original Paper](https://arxiv.org/abs/1802.04712) and [code](https://github.com/AMLab-Amsterdam/AttentionDeepMIL).

## Installation
### Conda  (Recommended)
```
conda env create -f environment.yml
conda activate mil
```
### Pip
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Usage
### Scripts
- **train_kfold.py**: trains KFold CV for a given csv and hyperparams.
- **eval_kfold.py**: Extract attention maps and other plots for KFold training, and also creates a single df of all predictions for validation sets.
- **inference.py**: inference with one of the KFold models on an external set. Currently, supports only model that was trained using kfold script.

### KFold Training Example
```
python train_kfold.py --csv train_tcga_IDH.csv --format h5 --cache --workers 4 --k 3 --track_samples --seed 1 \
    --results_dir ./kfold_tcga_IDH  --embedding_size 1024 --bag_size 4096 --mlp_dim 32 --D 32 --dropout 0.3 --epochs 5 \
    --lr 0.0001 --wd 0.001
```
### Inference on Test set
```
python inference.py --csv test_tcga_IDH.csv --target_root ./test_tcga_IDH --fold 1 --results_dir \
 ./kfold_tcga_IDH/train_tcga_IDH/2025-03-23_15-47-29
```

### Out Of The Box Inference
#### Load model
```python
import torch
from mil.predict import load_pretrained_abmil

ckpt_path = "assets/checkpoint.pth.tar" # replace with path to checkpoint file
config_path = "assets/config.json"  # replace with path to configuration JSON file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ABMIL model
model, threshold = load_pretrained_abmil(ckpt_path, config_path, device)
```
#### Inference
```python
import h5py

# Load embeddings
embedding_path = "assets/TCGA-DB-A4XG-01Z-00-DX1.76683FE4-150C-44B9-99C1-DFCCAB337CDE.h5"   # Replace to path of h5 file with 'features' key
with h5py.File(embedding_path, 'r') as f:
    embeds = torch.from_numpy(f['features'][()])

# Predict
with torch.inference_mode():
    out = model(embeds.to(device))
```
### Plot Heatmap
```python
import openslide
from mil.utils import min_max
from mil.heatmap import plot_heatmap

wsi_path = ""   # Replace a path to WSI
wsi = openslide.open_slide(wsi_path)
wsi_original_dim = wsi.level_dimensions[-1]
prob = torch.nn.functional.sigmoid(out['logits']).item() # Extracts prediction
scaled_attention_scores = min_max(out['A']) # min-max scaling for projeting scores into [0,1] range

# Load coords
with h5py.File(embedding_path, 'r') as f:
    coords = f['coords'][()]

# Plot
slide_id = "<some-id>"
fig = plot_heatmap(
    scores=scaled_attention_scores,
    coords=coords,
    wsi_original_dim=wsi_original_dim,
    title=f"{slide_id}={prob:.3f}",
)

# Show figure in notebook (Plotly will not be shown in GitHub)
fig.show(renderer='notebook')
```

## TCGA IDH Benchmark
You can view the results in the notebook: [IDH_Benchmark.ipynb](https://github.com/yuvfried/IDH-classification/blob/master/IDH_Benchmark.ipynb).
To run the notebook yourself:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/yuvalfriedmann/tcga-glioma-molecular-classification).
2. Use [Trident](https://github.com/mahmoodlab/TRIDENT) with `uni-v1` model to preprocess the data.
3. In the notebook, point to the folder containing the preprocessed features.


## TODO
- [ ] Fix Tensorboard.
- [ ] Support multi-class classification.
- [ ] TransMIL and other aggregators.
- [ ] Customise ABMIL architecture such as extending pre and post attention MLPs. Implemented [here](https://kaiko-ai.github.io/eva/main/reference/vision/models/networks/#eva.vision.models.networks.ABMIL) so just need to embed their model in this repo.
- [ ] Test early-stopping.
- [ ] Save checkpoint every n epochs.
- [ ] Test BagCase input data.
- [ ] Case level aggregations.
