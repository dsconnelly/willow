import os

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from ..utils.data import load_data, prepare_data
from ..utils.plotting import colors, format_latitude, get_pressures
from ..utils.scores import R_squared

def plot_offline(data_dir, model_dirs, output_path):
    """
    Plot training and test R-squared by level and latitude.

    Parameters
    ----------
    data_dir : str
        Directory containing training and test datasets.
    model_dirs : list of str
        Directories containing models trained on the data found in data_dir.
    output_path : str
        Path where the plot should be saved.

    """
    
    X_tr, Y_tr = load_data(data_dir, 'tr')
    X_te, Y_te = load_data(data_dir, 'te')
    
    scores_by_lev, scores_by_lat = {}, {}
    for model_dir in model_dirs:
        path = os.path.join(model_dir, 'model.pkl')
        model = joblib.load(path)

        scores_by_lev[model.name] = {
            'tr' : _score_model(X_tr, Y_tr, model),
            'te' : _score_model(X_te, Y_te, model)
        }
        
        scores_by_lat[model.name] = {
            'tr' : _get_scores_by_lat(X_tr, Y_tr, model),
            'te' : _get_scores_by_lat(X_te, Y_te, model)
        }
            
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(12, 8)
    
    gs = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 2], figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    
    ax, y, pressures = _setup_lev_axis(axes[0])
    for color, (model_name, scores) in zip(colors, scores_by_lev.items()):
        ax.plot(scores['tr'], -y, color=color, label=model_name)
        ax.plot(scores['te'], -y, color=color, ls='dashed')
        
    ax, lats = _setup_lat_axis(axes[1]), np.linspace(-90, 90, 64)
    for color, (model_name, scores) in zip(colors, scores_by_lat.items()):
        ax.plot(lats, scores['tr'], color=color, label=model_name)
        ax.plot(lats, scores['te'], color=color, ls='dashed')
        
    ax.plot([], [], color='gray', label='training')
    ax.plot([], [], color='gray', ls='dashed', label='test')
    ax.legend()
       
    plt.savefig(output_path, dpi=400)
    
def _get_scores_by_lat(X, Y, model):
    lats = np.sort(X['latitude'].unique())
    scores = np.zeros(len(lats))
    
    for i, lat in enumerate(lats):
        idx = X['latitude'] == lat
        scores[i] = _score_model(X[idx], Y[idx], model).mean()
        
    return scores
    
def _score_model(X, Y, model):
    X, Y = prepare_data(X, Y, model.name)
    
    return R_squared(Y, model.predict(X))

def _setup_lat_axis(ax):
    ax.set_xlim(-90, 90)
    ax.set_ylim(0.2, 1)
    
    ticks = np.linspace(-90, 90, 7)
    labels = list(map(format_latitude, ticks))
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.linspace(0.2, 1, 5))

    ax.set_xlabel('latitude')
    ax.set_ylabel('$R^2$')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax
    
def _setup_lev_axis(ax):
    pressures = get_pressures()
    y = np.arange(len(pressures))
    
    ax.set_xlim(0.2, 1)
    ax.set_ylim(-y[-1], -y[0])
    
    ax.set_xticks(np.linspace(0.2, 1, 5))
    ax.set_yticks(-y[::3])
    ax.set_yticklabels(pressures[::3])
    
    ax.set_xlabel('$R^2$')
    ax.set_ylabel('pressure (hPa)')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax, y, pressures