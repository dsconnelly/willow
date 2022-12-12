import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score

from ..utils.datasets import load_datasets
from ..utils.plotting import COLORS, format_latitude, format_name
from ..utils.wrappers import MiMAModel

def plot_R2_scores(
    data_dir: str,
    model_dirs: list[str],
    output_path: str,
    n_samples: int=int(1e4),
) -> None:
    """
    Plot training and test R2 scores by level and latitude.

    Parameters
    ----------
    data_dir : Directory containing training and test datasets.
    model_dirs : Directories containing trained models.
    output_path : Path where the plot should be saved.
    n_samples : How many samples to use in calculating training and test scores.

    """

    X_tr, Y_tr_df = load_datasets(data_dir, 'tr', n_samples)
    X_te, Y_te_df = load_datasets(data_dir, 'te', n_samples)
    Y_tr, Y_te = Y_tr_df.to_numpy(), Y_te_df.to_numpy()

    lats = np.linspace(-90, 90, len(X_tr['latitude'].unique()))
    pressures = [s.split(' @ ')[-1].split()[0] for s in Y_tr_df.columns]
    y = np.arange(-len(pressures), 0)

    fig = plt.figure(constrained_layout=True)
    axes = _make_axes(fig, pressures)
    fig.set_size_inches(9, 6)

    for model_dir, color in zip(model_dirs, COLORS):
        path = os.path.join(model_dir, 'model.pkl')
        model: MiMAModel = joblib.load(path)
        name = format_name(model.name)

        by_lev, by_lat = _get_scores(X_tr, Y_tr, model)
        axes[0].plot(by_lev, y, color=color)
        axes[1].plot(lats, by_lat, color=color, label=name)

        by_lev, by_lat = _get_scores(X_te, Y_te, model)
        axes[0].plot(by_lev, y, color=color, ls='dashed')
        axes[1].plot(lats, by_lat, color=color, ls='dashed')

    axes[1].plot([], [], color='gray', label='training')
    axes[1].plot([], [], color='gray', ls='dashed', label='test')
    axes[1].legend()

    plt.savefig(output_path)

def _configure_lat_axis(ax: Axes) -> None:
    """
    Configure an axis to plot R2 score against latitude.

    Parameters
    ----------
    ax : Axis to configure.

    """
    
    lats = np.linspace(-90, 90, 7)
    labels = list(map(format_latitude, lats))
    scores = [0.2, 0.4, 0.6, 0.8, 1]

    ax.set_xticks(lats)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlim(lats[0], lats[-1])

    ax.set_yticks(scores)
    ax.set_ylim(scores[0], scores[-1])

    ax.set_xlabel('latitude')
    ax.set_ylabel('$R^2$')

def _configure_lev_axis(ax: Axes, pressures: list[str]) -> None:
    """
    Configure an axis to plot level against R2 score.

    Parameters
    ----------
    ax : Axis to configure.
    pressures : List of formatted pressures at each level, in hPa. Should be
        subsampled to include only the pressures to include as ticks.

    """
    
    scores = [0.2, 0.4, 0.8, 0.6, 1]
    y = np.arange(-len(pressures), 0)
    
    ax.set_xticks(scores)
    ax.set_xticklabels(scores, rotation=45)
    ax.set_xlim(scores[0], scores[-1])

    ax.set_yticks(y)
    ax.set_yticklabels(pressures)
    ax.set_ylim(y[0], y[-1])

    ax.set_xlabel('$R^2$')
    ax.set_ylabel('pressure (hPa)')

def _get_scores(
    X: pd.DataFrame,
    Y: np.ndarray,
    model: MiMAModel
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate R2 scores by level and latitude.

    Parameters
    ----------
    X : `DataFrame` containing input features.
    Y : `ndarray` containing target drags.
    model : Model to assess.

    Returns
    -------
    by_lev : Scores by level, from the top of the atmosphere down.
    by_lat : Scores by latitude, from the south pole to the north pole.

    """

    lats = np.sort(X['latitude'].unique())
    by_lev = np.zeros(Y.shape[1])
    by_lat = np.zeros(len(lats))

    output = model.predict(X)
    for i, lat in enumerate(lats):
        idx = X['latitude'] == lat
        weight = idx.sum() / len(X)

        with np.errstate(divide='ignore'):
            scores = r2_score(Y[idx], output[idx], multioutput='raw_values')
            scores[np.isinf(scores)] = np.nan

        by_lev = by_lev + weight * scores
        by_lat[i] = np.nanmean(scores)

    return by_lev, by_lat

def _make_axes(fig: Figure, pressures: list[str]) -> list[Axes]:
    """
    Create and configure the axes for the level and latitude plots.

    Parameters
    ----------
    fig : Figure to place axes in.
    pressures : List of formatted pressures at each level, in hPa.

    Returns
    -------
    axes : List of properly-formatted axes.

    """

    gs = GridSpec(ncols=2, nrows=1, width_ratios=[1, 2], figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]

    _configure_lev_axis(axes[0], pressures[::3])
    _configure_lat_axis(axes[1])

    for ax in axes:
        ax.tick_params(direction='in', top=True, right=True)
        ax.grid(True, color='lightgray')

    return axes
    