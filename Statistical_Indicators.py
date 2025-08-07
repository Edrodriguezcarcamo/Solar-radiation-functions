# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:09:42 2022

@author: edrod
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def statistical_indicators(predictions, actual, model_names=None, nbins=1000):
    """
    References:
      - U95, GPI: https://doi.org/10.1016/j.jclepro.2018.09.246
      - KSI, OVER, CPI: https://doi.org/10.1016/j.rser.2014.07.117
    """

    # -----------------------------
    # helper: Phi(N) ≈ 1.628 for large N
    def _phi_asymptotic(N):
        return 1.628

    # helper: KSI and OVER per Gueymard (2014)
    def ksi_over(act, pred, nbins):
        a = np.asarray(act, float).ravel()
        p = np.asarray(pred, float).ravel()
        mask = (~np.isnan(a)) & (~np.isnan(p))
        a, p = a[mask], p[mask]
        if a.size == 0:
            return np.nan, np.nan

        xmin_raw = min(a.min(), p.min())
        xmax_raw = max(a.max(), p.max())
        if xmax_raw <= xmin_raw:
            return np.nan, np.nan

        a_red = (a - xmin_raw) / (xmax_raw - xmin_raw)
        p_red = (p - xmin_raw) / (xmax_raw - xmin_raw)

        edges = np.linspace(0.0, 1.0, nbins + 1)
        dx = 1.0 / nbins

        ha, _ = np.histogram(a_red, bins=edges, density=True)
        hp, _ = np.histogram(p_red, bins=edges, density=True)
        cdfa = np.cumsum(ha) * dx
        cdfp = np.cumsum(hp) * dx

        Dn = np.abs(cdfp - cdfa)
        N = a.size
        Dc = _phi_asymptotic(N) / np.sqrt(N)
        Ac = Dc * 1.0  # reduced domain => Xmax - Xmin = 1

        KSI = 100.0 / Ac * (Dn.sum() * dx)
        OVER = 100.0 / Ac * (np.maximum(Dn - Dc, 0.0).sum() * dx)
        return KSI, OVER

    # -----------------------------
    # normalize input into dict of {model_name: predictions}
    a = np.asarray(actual, dtype=float).ravel()
    if isinstance(predictions, dict):
        pred_dict = {str(k): np.asarray(v, dtype=float).ravel() for k, v in predictions.items()}
    else:
        p = np.asarray(predictions, dtype=float)
        if p.ndim == 1:
            pred_dict = {"Model_1": p.ravel()}
        elif p.ndim == 2:
            n_models = p.shape[0]
            names = model_names if model_names is not None else [f"Model_{i+1}" for i in range(n_models)]
            if len(names) != n_models:
                raise ValueError("Length of model_names must match number of models.")
            pred_dict = {names[i]: p[i, :].ravel() for i in range(n_models)}
        else:
            raise ValueError("predictions must be 1D, 2D, or dict.")

    # -----------------------------
    # compute metrics for one model
    def _metrics_one(pred, act):
        mask = ~np.isnan(pred) & ~np.isnan(act)
        pred, act = pred[mask], act[mask]
        n = len(act)
        if n == 0:
            raise ValueError("No valid pairs.")

        a_mean = act.mean()
        if a_mean == 0:
            raise ValueError("Mean of actual is zero; relative metrics undefined.")

        res = pred - act
        rmse = np.sqrt(np.mean(res**2))
        mbe = np.mean(res)
        mae = np.mean(np.abs(res))

        nz = act != 0
        if np.any(nz):
            rel = res[nz] / act[nz]
            mpe = np.mean(rel) * 100.0
            mape = np.mean(np.abs(rel)) * 100.0
            ermax = np.max(np.abs(rel))
            mare = np.mean(np.abs(rel))
        else:
            mpe = mape = ermax = mare = np.nan

        nrmse = rmse / a_mean * 100.0
        nmbe = mbe / a_mean * 100.0
        rrmse = rmse / a_mean

        sd = np.std(res, ddof=1) if n > 1 else 0.0
        u95 = 1.96 * np.sqrt(sd**2 + rmse**2)

        s2 = rmse**2 - mbe**2
        t_stats = np.sqrt((n - 1) * mbe**2 / s2) if s2 > 0 and n > 1 else np.nan

        mabe = abs(a_mean - pred.mean())
        r2 = r2_score(act, pred) if np.std(act) > 0 else np.nan
        r = np.corrcoef(act, pred)[0, 1] if np.std(act) > 0 and np.std(pred) > 0 else np.nan

        ksi, over = ksi_over(act, pred, nbins)
        cpi = (ksi + over + 2 * nrmse) / 4.0 if ksi is not np.nan else np.nan

        return {
            'NRMSE (%)': nrmse,
            'R2': r2,
            'R': r,
            'NMBE (%)': nmbe,
            'RMSE': rmse,
            'MBE': mbe,
            'MAE': mae,
            'MPE (%)': mpe,
            'MAPE (%)': mape,
            'U95': u95,
            'RRMSE': rrmse,
            'T-Stats': t_stats,
            'ERMAX': ermax,
            'MARE': mare,
            'MABE': mabe,
            'KSI (%)': ksi,
            'OVER (%)': over,
            'CPI': cpi
        }

    # -----------------------------
    # loop models
    rows = {}
    for name, preds in pred_dict.items():
        if preds.shape[0] != a.shape[0]:
            raise ValueError(f"Model '{name}': length mismatch.")
        rows[name] = _metrics_one(preds, a)

    df = pd.DataFrame.from_dict(rows, orient='index')

    # -----------------------------
    # compute GPI if multi-model
    if df.shape[0] >= 2:
        mins = df.min(axis=0)
        maxs = df.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        norm = (df - mins) / ranges
        med_norm = (df.median(axis=0) - mins) / ranges
        weights = pd.Series(1.0, index=df.columns)
        if 'R' in weights:
            weights['R'] = -1.0
        dev = norm.subtract(med_norm, axis=1)
        gpi = (dev * weights).sum(axis=1)
        df['GPI'] = gpi
    else:
        df['GPI'] = np.nan

    # -----------------------------
    # reorder columns as requested
    ordered_cols = [
        'NRMSE (%)', 'R2', 'NMBE (%)', 'RMSE', 'MBE', 'MAE', 'MPE (%)',
        'MAPE (%)', 'U95', 'RRMSE', 'T-Stats', 'ERMAX', 'MARE', 'MABE',
        'GPI', 'KSI (%)', 'OVER (%)', 'CPI'
    ]
    return df[ordered_cols]
