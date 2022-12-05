import os

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

from matplotlib.colors import SymLogNorm

from ..utils.ad99 import AlexanderDunkerton
from ..utils.datasets import load_datasets
from ..utils.mima import R_dry
from ..utils.offline import get_scores_by_lev
from ..utils.plotting import (
    colors,
    format_latitude,
    format_name,
    format_pressure,
    get_bounds_and_cmap
)

def plot_offline_scores(data_dir, model_dirs, output_path):
    """
    Plot training and test R-squared scores by level and latitude.

    Parameters
    ----------
    data_dir : str
        Directory containing training and test datasets.
    model_dirs : list of str
        Directories containing models trained on the data found in data_dir.
    output_path : str
        Path where the plot should be saved.

    """
    
    X_tr, Y_tr = load_datasets(data_dir, 'tr', int(1e5))
    X_te, Y_te = load_datasets(data_dir, 'te', int(1e5))
    
    scores_by_lev, scores_by_lat = {}, {}
    for model_dir in model_dirs:
        path = os.path.join(model_dir, 'model.pkl')
        model = joblib.load(path)

        scores_by_lev[model.name] = {
            'tr' : get_scores_by_lev(X_tr, Y_tr, model),
            'te' : get_scores_by_lev(X_te, Y_te, model)
        }
        
        scores_by_lat[model.name] = {
            'tr' : _get_scores_by_lat(X_tr, Y_tr, model),
            'te' : _get_scores_by_lat(X_te, Y_te, model)
        }
            
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(8, 16 / 3)
    
    gs = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 2], figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]

    lats = np.linspace(-90, 90, 64)
    pressures = [s.split('@')[-1].split()[0] for s in X_tr.columns if 'T' in s]
    y = np.arange(len(pressures))
    
    ax = _setup_lev_axis(axes[0], pressures, y)
    for color, (model_name, scores) in zip(colors, scores_by_lev.items()):
        ax.plot(scores['tr'], -y, color=color)
        ax.plot(scores['te'], -y, color=color, ls='dashed')
        
    ax = _setup_lat_axis(axes[1])
    for color, (model_name, scores) in zip(colors, scores_by_lat.items()):
        ax.plot(lats, scores['tr'], color=color, label=format_name(model_name))
        ax.plot(lats, scores['te'], color=color, ls='dashed')

    ax.plot([], [], color='gray', label='training')
    ax.plot([], [], color='gray', ls='dashed', label='test')
    ax.legend()

    for ax in axes:
        ax.tick_params(direction='in', top=True, right=True)
        ax.grid(True, color='lightgray')
       
    plt.savefig(output_path)

def plot_tropical_drag(data_dir, model_dir, output_path):
    """
    A temporary function for various analyses.

    Parameters
    ----------
    data_dir : str
        Directory containing training and test datasets.
    model_dir : str
        Directory containing a model trained on the data found in data_dir.
    output_path : str
        Path where the plot should be saved.

    """

    X_tr, Y_tr = load_datasets(data_dir, 'tr', int(1e6), component='u')
    X_te, Y_te = load_datasets(data_dir, 'te', int(1e6), component='u')

    tropics = abs(X_tr['latitude']) < 5
    X_tr, Y_tr = X_tr[tropics], Y_tr[tropics]

    tropics = abs(X_te['latitude']) < 5
    X_te, Y_te = X_te[tropics], Y_te[tropics]

    model = joblib.load(os.path.join(model_dir, 'model.pkl'))
    out_tr = model.predict_online(X_tr.to_numpy())
    out_te = model.predict_online(X_te.to_numpy())

    out_tr = pd.DataFrame(out_tr, index=Y_tr.index, columns=Y_tr.columns)
    out_te = pd.DataFrame(out_te, index=Y_te.index, columns=Y_te.columns)

    X = pd.concat((X_tr, X_te), ignore_index=True)
    Y = 60 * 60 * 24 * pd.concat((Y_tr, Y_te), ignore_index=True)
    out = 60 * 60 * 24 * pd.concat((out_tr, out_te), ignore_index=True)
    diff = out - Y

    time = X['time']
    pressures = [s.split('@')[-1].split()[0] for s in Y.columns][:25]
    y = np.arange(len(pressures))

    Y = Y.groupby(time).mean().iloc[:, :25]
    out = out.groupby(time).mean().iloc[:, :25]
    diff = diff.groupby(time).mean().iloc[:, :25]
    years = (Y.index - Y.index[0]) / 360

    cbar_scale = 0.1
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(9, 3 * (3 + cbar_scale))

    height_ratios = [1] * 3 + [cbar_scale]
    gs = gridspec.GridSpec(
        ncols=1, nrows=(3 + 1),
        height_ratios=height_ratios,
        figure=fig
    )

    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    data = {'AD99' : Y.to_numpy(), 'model' : out.to_numpy()}
    _, _, cmap = get_bounds_and_cmap('gwf_u', data)
    norm = SymLogNorm(linthresh=1e-2, vmin=-30, vmax=30)
    
    powers = np.linspace(-2, 1, 7)
    powers = np.concatenate((powers[::-1], np.array([0]), powers))
    
    signs = np.ones(len(powers))
    signs[:7], signs[7] = -1, 0
    levels = 3 * signs * (10.0 ** powers)
    
    axes[0].contourf(
        years, -y, Y.T,
        cmap=cmap,
        norm=norm,
        levels=levels
    )

    axes[1].contourf(
        years, -y, out.T,
        cmap=cmap,
        norm=norm,
        levels=levels
    )

    axes[2].contourf(
        years, -y, diff.T,
        cmap=cmap,
        norm=norm,
        levels=levels
    )

    axes[0].set_title('AD99')
    axes[1].set_title(format_name(model.name))
    axes[2].set_title('emulator $-$ AD99')

    line_x = 4 * np.ones(100)
    line_y = np.linspace(-y[-1], -y[0], 100)

    for ax in axes:
        ax.set_xticks(np.linspace(years.min(), years.max(), 9))
        ax.set_xticklabels(np.linspace(0, 8, 9))
        ax.set_xlabel('year')

        ax.set_yticks(-y[::3])
        ax.set_yticklabels(pressures[::3])
        ax.set_ylabel('pressure (hPa)')

        ax.plot(line_x, line_y, color='k', ls='dashed')

    cax = fig.add_subplot(gs[3])

    plt.savefig(output_path)

def _get_scores_by_lat(X, Y, model):
    lats = np.sort(X['latitude'].unique())
    scores = np.zeros(len(lats))
    
    for i, lat in enumerate(lats):
        idx = X['latitude'] == lat
        scores[i] = np.nanmean(get_scores_by_lev(X[idx], Y[idx], model))

    return scores
     
def _setup_lat_axis(ax):
    ax.set_xlim(-90, 90)
    ax.set_ylim(0.2, 1)
    
    ticks = np.linspace(-90, 90, 7)
    labels = list(map(format_latitude, ticks))
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(np.linspace(0.2, 1, 5))

    ax.set_xlabel('latitude')
    ax.set_ylabel('$R^2$')
    
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    return ax
    
def _setup_lev_axis(ax, pressures, y):
    ax.set_xlim(0.2, 1)
    ax.set_ylim(-y[-1], -y[0])
    
    ax.set_xticks(np.linspace(0.2, 1, 5))
    ax.set_xticklabels([0.2, 0.4, 0.6, 0.8, 1.0], rotation=45)

    ax.set_yticks(-y[::3])
    ax.set_yticklabels(pressures[::3])
    
    ax.set_xlabel('$R^2$')
    ax.set_ylabel('pressure (hPa)')
    
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    return ax