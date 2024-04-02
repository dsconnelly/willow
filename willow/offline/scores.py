import os

import dask
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from sklearn.decomposition import PCA

from ..preprocessing.mima import _sample_dataset
from ..utils.ad99 import AlexanderDunkerton
from ..utils.datasets import load_datasets
from ..utils.mima import get_paths, open_mima_output
from ..utils.plotting import COLORS, format_latitude, format_name
from ..utils.statistics import R2_score, standardize
from ..utils.wrappers import MiMAModel

def plot_emulator_drift(
    data_dir: str,
    case_dirs: list[str],
    output_path: str,
    n_samples: int=int(1e6)
) -> None:
    """
    Do a PCA analysis to show if emulators respect training bounds.

    Parameters
    ----------
    data_dir : Directory containing training and test datasets.
    case_dirs : Directories containing MiMA runs.
    output_path : Path where the plot should be saved.
    n_samples : How many samples to search for extreme errors.

    """

    X, Y_df = load_datasets(data_dir, 'te', component='u')
    keep = (abs(X['latitude']) <= 5).values
    X, Y_df = X.iloc[keep], Y_df.iloc[keep]
    Y = Y_df.to_numpy()

    case_dirs = case_dirs[1:]
    n_subplots = len(case_dirs)
    fig, axes = plt.subplots(ncols=n_subplots)
    fig.set_size_inches(4.5 * n_subplots, 4.5)

    Y_std, means, stds = standardize(Y)
    pca = PCA(n_components=2).fit(Y_std)
    x, y = pca.transform(Y_std).T

    amax = 8
    ticks = np.linspace(-amax, amax, 5)
    range = ((-amax, amax), (-amax, amax))
    h_ref, x_edges, y_edges = np.histogram2d(x, y, bins=64, range=range)

    values = h_ref.flatten()
    idx = np.argsort(values)
    shade = (np.cumsum(values[idx][::-1]) < 0.95 * values.sum())[::-1]
    shade = shade[np.argsort(idx)].reshape(h_ref.shape).astype(float)
    shade[shade == 0] = np.nan

    for case_dir, ax, color in zip(case_dirs, axes, COLORS):
        with open_mima_output(get_paths(case_dir), 16) as ds:
            ds = ds.sel(lat=slice(-5, 5))
            Y_model = ds['gwfu_cgwd'].values
            Y_model = Y_model.transpose(0, 2, 3, 1)
            Y_model = Y_model.reshape(-1, 40)

            idx = np.random.permutation(Y_model.shape[0])[:n_samples]
            Y_model = Y_model[idx]

            x, y = pca.transform(standardize(Y_model, means, stds)[0]).T
            ax.scatter(x, y, s=5, c=color, alpha=0.3, linewidth=0, zorder=10)
    
        ax.pcolormesh(
            x_edges, y_edges, shade.T,
            vmin=0, vmax=4, cmap='Greys',
            zorder=5
        )

        ax.set_xlim(-amax, amax)
        ax.set_ylim(-amax, amax)

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        name = format_name(case_dir, True)
        name = name[:name.index(' (')]
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(output_path)

def plot_R2_scores(
    data_dir: str,
    model_dirs: list[str],
    output_path: str,
    n_samples: int=int(1e6),
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

    # X_tr, Y_tr_df = load_datasets(data_dir, 'te', n_samples, phase='west')
    # X_te, Y_te_df = load_datasets(data_dir, 'te', n_samples, phase='east')
    Y_tr, Y_te = Y_tr_df.to_numpy(), Y_te_df.to_numpy()

    lats = np.linspace(-90, 90, len(X_tr['latitude'].unique()))
    pressures = [s.split(' @ ')[-1].split()[0] for s in Y_tr_df.columns]
    y = -np.arange(len(pressures))

    fig = plt.figure(constrained_layout=True)
    axes = _make_axes(fig, pressures)
    fig.set_size_inches(9, 6)

    z = 10 + len(model_dirs)
    for model_dir, color in zip(model_dirs, COLORS):
        path = os.path.join(model_dir, 'model.pkl')
        model: MiMAModel = joblib.load(path)
        name = format_name(model.name)

        by_lev, by_lat = _get_scores(X_tr, Y_tr, model)
        axes[0].plot(by_lev, y, color=color, ls='dashed', zorder=z)
        axes[1].plot(lats, by_lat, color=color, ls='dashed', zorder=z)

        by_lev, by_lat = _get_scores(X_te, Y_te, model)
        axes[0].plot(by_lev, y, color=color, zorder=z)
        axes[1].plot(lats, by_lat, color=color, label=name, zorder=z)

        z = z - 1

    axes[1].plot([], [], color='gray', ls='dashed', label='training')
    axes[1].plot([], [], color='gray', label='test')

    # axes[1].plot([], [], color='gray', ls='dashed', label='QBOW')
    # axes[1].plot([], [], color='gray', label='QBOE')
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
    # scores = [0.2, 0.4, 0.6, 0.8, 1]
    scores = [0.6, 0.7, 0.8, 0.9, 1]

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
    pressures : List of formatted pressures at each level, in hPa.

    """
    
    scores = [0.2, 0.4, 0.8, 0.6, 1]
    y = -np.arange(len(pressures))
    
    ax.set_xticks(scores)
    ax.set_xticklabels(scores, rotation=45)
    ax.set_xlim(scores[0], scores[-1])

    ax.set_yticks(y[::3])
    ax.set_yticklabels(pressures[::3])
    ax.set_ylim(y[-1], y[0])

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

    output = model.predict(X)
    by_lev = R2_score(Y, output, reduce=False)

    lats = np.sort(X['latitude'].unique())
    by_lat = np.zeros(len(lats))

    for i, lat in enumerate(lats):
        idx = X['latitude'] == lat
        by_lat[i] = R2_score(Y[idx], output[idx])

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

    _configure_lev_axis(axes[0], pressures)
    _configure_lat_axis(axes[1])

    for ax in axes:
        ax.tick_params(direction='in', top=True, right=True)
        ax.grid(True, color='lightgray')

    return axes
    
