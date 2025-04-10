from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px


def stitch(A, coords, wsi_original_dim, patch_size_level0) -> pd.DataFrame:
    """
    @param A:
    @param coords:
    @param original_dim:
    @param patch_size_level0: patch pixel size on magnification of level0.
    For example if level1 is 20x and level0 is 40x,
    and patches were extracted at level1 with 256 px, then patch_size_level0=512.
    @return:

    """
    orig_width, orig_height = wsi_original_dim

    # attentions = np.interp(A, [A.min(), A.max()], [0, 255]) # converting to 256 (8 bits) distinct colors
    colors = deepcopy(A)
    colors[colors == colors.min()] += 1e-4 # set darkest color to zero attention locations in image.

    locations = [
        (int(coords[i, 0]) // patch_size_level0, int(coords[i, 1]) // patch_size_level0)
        for i in range(len(coords))
    ]

    width = orig_width // patch_size_level0
    height = orig_height // patch_size_level0

    stitched_image = np.zeros((height, width), dtype=np.float32)
    xs = np.arange(width)
    ys = np.arange(height)

    for index, location in enumerate(locations):
        x, y = location
        stitched_image[y, x] = colors[index]
        origin_x, origin_y = coords[index]
        xs[x] = origin_x
        ys[y] = origin_y

    stitch_map = pd.DataFrame(stitched_image.squeeze(), columns=xs.astype('U'), index=ys.astype('U'))
    # stitch_map /= 255
    return stitch_map

def plotly_interactive_attn_map(stitch_map, title="", cmap="balance") -> go.Figure:
    fig = px.imshow(stitch_map,
                    labels=dict(color="Scaled Attention"),
                    color_continuous_scale=cmap, title=title,
                    zmin=np.nanmin(stitch_map), zmax=np.nanmax(stitch_map))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def plot_heatmap(
        scores,
        coords,
        wsi_original_dim,
        patch_size_level0: int = 512,
        title="",
        cmap="Turbo"
) -> go.Figure:
    """
    Plots a heatmap of attention scores for a whole slide image (WSI), where each coordinate from `coords`
    is colored based on its corresponding score.

    Parameters:
    - scores (array-like): Array of attention scores for the patches.
    - coords (array-like): List of (x, y) coordinates for the patch positions.
    - wsi_original_dim (tuple): Dimensions (width, height) of the original WSI.
    - patch_size_level0 (int, optional): Size of each patch at the base level (default is 512).
    - title (str, optional): Title of the heatmap plot (default is "").
    - cmap (str, optional): Colormap for the heatmap (default is "Turbo").

    Returns:
    - plotly_heatmap: An interactive Plotly heatmap object.
    """
    df_stitch = stitch(scores, coords, wsi_original_dim=wsi_original_dim, patch_size_level0=patch_size_level0)
    df_stitch = df_stitch.replace(to_replace=0, value=np.nan)  # ignore wsi areas without patches
    plotly_heatmap = plotly_interactive_attn_map(df_stitch, title=title, cmap=cmap)
    return plotly_heatmap