import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import linregress
from sklearn.decomposition import PCA

from ..utils.ad99 import AlexanderDunkerton
from ..utils.datasets import load_datasets, prepare_datasets
from ..utils.mima import R_dry
from ..utils.offline import get_scores_by_lev
from ..utils.plotting import colors
from ..utils.qbo import load_qbo, qbo_statistics
from ..utils.shapley import get_lmis

def plot_oracle(
        model_dirs,
        case_dirs,
        output_path,
        data_dir=None,
        oracle='momentum',
        metric='period'
    ):
    """
    Plot an offline metric against an online metric.

    Parameters
    ----------
    model_dirs : list of str
        Directories where models are saved to disk. If offline == 'LMI', then
        the precomputed Shapley values should also be saved here.
    case_dirs : list of str
        Directories where coupled runs are saved.
    output_path : str
        Path where image should be saved.
    data_dir : str
        Directory containing training data. Only needed if offline == 'R2'.
    oracle : str
        The particular offline metric to use as an orcale. Must be one of
        {'R2', 'LMI'}.
    metric : str
        The online metric to display. At present, only 'period' is supported.

    """

    if data_dir is not None:
        component = 'u' if oracle == 'momentum' else 'both'
        X, Y = load_datasets(data_dir, 'te', int(1e5), component=component)

        tropics = abs(X['latitude']) < 5
        X, Y = X[tropics], Y[tropics]

        ad99 = AlexanderDunkerton()
        p_surf = X['surface pressure'].to_numpy()
        
        p = np.vstack([ad99.get_vertical_coords(v)[1:] for v in p_surf])
        T = X[[s for s in X.columns if 'T' in s]].to_numpy()
        rho = p / (R_dry * T)

        # if oracle == 'test':
        #     scores = np.zeros((len(model_dirs), 39 + 64))

            # tropics = abs(X['latitude']) < 5
            # # X, Y = X[tropics], Y[tropics]

            # # idx = X['time'].argsort()
            # # X, Y = X.iloc[idx], Y.iloc[idx]
            # # time = X.groupby(X['time']).mean().index

    xs = np.zeros(len(model_dirs))
    ys = np.zeros(len(case_dirs))
    kinds = []

    def _get_scores_by_lat(X, Y, model):
        lats = np.sort(X['latitude'].unique())
        scores = np.zeros(len(lats))
        
        for i, lat in enumerate(lats):
            idx = X['latitude'] == lat
            scores[i] = np.nanmean(get_scores_by_lev(X[idx], Y[idx], model))

        return scores

    for i, model_dir in enumerate(model_dirs):
        kinds.append(os.path.basename(model_dir).split('-')[0])

        if oracle == 'R2':
            model = joblib.load(os.path.join(model_dir, 'model.pkl'))
            xs[i] = get_scores_by_lev(X, Y, model)[:24].mean()

        elif oracle == 'LMI':
            with xr.open_dataset(os.path.join(model_dir, 'shapley.nc')) as ds:
                scores = abs(ds['scores']).mean('sample')

            k_lo, k_hi = 11, 37
            levels, lmis = get_lmis(model_dir, scores)
            xs[i] = linregress(levels[k_lo:k_hi], lmis[k_lo:k_hi])[0]

        elif oracle == 'momentum':
            model = joblib.load(os.path.join(model_dir, 'model.pkl'))
            momentum = rho * model.predict_online(X.to_numpy())
            xs[i] = abs(momentum[:, :24]).sum(axis=1).mean()

    for i, case_dir in enumerate(case_dirs):
        u = load_qbo(case_dir, n_years=36)
        period, _, amp, _ = qbo_statistics(u)

        if metric == 'period':
            ys[i] = abs(period - 28.921428571428574)

        elif metric == 'amplitude':
            ys[i] = amp

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    for kind, color in zip(set(kinds), colors):
        idx = [i for i, s in enumerate(kinds) if s == kind]
        ax.scatter(xs[idx], ys[idx], color=color, label=kind)

    m, b, r, *_ = linregress(xs, ys)
    line = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(line, m * line + b, color='k', ls='dashed', label=f'r = {r:.2f}')
    
    xlabel = {
        'R2' : 'tropical stratosphere $R^2$',
        'LMI' : 'LMI slope',
        'momentum' : 'tropical stratosphere momentum'
    }[oracle]

    ylabel = {
        'period' : 'QBO period error (month)',
        'amplitude' : 'QBO amplitude (m s$^{-1}$)'
    }[metric]

    _, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)

    ax.tick_params(direction='in', top=True, right=True)
    ax.grid(True, color='gray', zorder=-1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
