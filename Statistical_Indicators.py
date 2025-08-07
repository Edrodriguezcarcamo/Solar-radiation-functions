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
    Compute statistical indicators + distribution metrics (KSI/OVER/CPI) and GPI.
    References:
      - U95, GPI: https://doi.org/10.1016/j.jclepro.2018.09.246
      - KSI, OVER, CPI: https://doi.org/10.1016/j.rser.2014.07.117
    DIRECTIONAL GUIDE (↑ = higher is better, ↓ = lower is better)
    ----------------------------------------------------------------
    Class A / classic regression:
      - NRMSE (%) ............... ↓  (normalized RMSE; smaller error is better)
      - R2 ....................... ↑  (coefficient of determination)
      - R ........................ ↑  (Pearson correlation coefficient)
      - NMBE (%) ................ ↓  (normalized mean bias; closer to 0 is better)
      - RMSE .................... ↓
      - MBE ..................... →  (closer to 0 is better; sign shows bias)
      - MAE ..................... ↓
      - MPE (%) ................. →  (closer to 0 is better; undefined if any actual==0)
      - MAPE (%) ................ ↓  (undefined if any actual==0; we guard it)
      - U95 ..................... ↓  (95% expanded uncertainty of residuals)
      - RRMSE ................... ↓  (fraction; same as NRMSE but not in %)
      - T-Stats ................. ↓  (smaller suggests MBE not statistically different from 0)
      - ERMAX ................... ↓  (max relative error magnitude)
      - MARE .................... ↓  (mean absolute relative error)
      - MABE .................... ↓  (|mean(actual) - mean(pred)|)
      - StdDev Ratio ............ →  (σ_pred / σ_actual; closer to 1 is better)
      - Slope ................... →  (slope of pred vs actual; closer to 1 is better)

    Class B / complementary:
      - NSE ..................... ↑  (Nash–Sutcliffe Efficiency; 1 is perfect, <0 worse than mean)
      - WIA ..................... ↑  (Willmott’s index of agreement; 0–1)
      - LCE ..................... ↑  (Legates–McCabe efficiency; 1 is perfect, can be <0)

    Distribution comparison (Gueymard/Espinar family):
      - KSI (%) ................. ↓  (integral KS index; smaller distance between CDFs is better)
      - OVER (%) ................ ↓  (exceedance over the critical KS distance; smaller is better)
      - CPI ..................... ↓  (combined performance index = (KSI+OVER+2*NRMSE%)/4)

    Aggregate ranking:
      - GPI ..................... ↓  (Global Performance Indicator; computed across models only)
                                   Lower is better; uses R with negative weight per paper.
    """

    # --- helpers ---------------------------------------------------------------
    def _phi_asymptotic(N):
        # Φ(N) ~ 1.628 for large N (common approximation used in the literature)
        return 1.628

    def ksi_over(act, pred, nbins):
        """
        KSI/OVER per Gueymard (2014): integrate differences between CDFs
        in the *reduced* domain x∈[0,1].
        KSI: integral of |ΔCDF|
        OVER: integral of max(|ΔCDF| - Dc, 0)
        """
        a = np.asarray(act, float).ravel()
        p = np.asarray(pred, float).ravel()
        mask = (~np.isnan(a)) & (~np.isnan(p))
        a, p = a[mask], p[mask]
        if a.size == 0:
            return np.nan, np.nan

        # Reduced irradiance: linear map both series to [0, 1]
        xmin_raw = min(a.min(), p.min())
        xmax_raw = max(a.max(), p.max())
        if xmax_raw <= xmin_raw:
            return np.nan, np.nan
        a_red = (a - xmin_raw) / (xmax_raw - xmin_raw)
        p_red = (p - xmin_raw) / (xmax_raw - xmin_raw)

        # Uniform bins on [0,1] → CDFs
        edges = np.linspace(0.0, 1.0, nbins + 1)
        dx = 1.0 / nbins
        ha, _ = np.histogram(a_red, bins=edges, density=True)
        hp, _ = np.histogram(p_red, bins=edges, density=True)
        cdfa = np.cumsum(ha) * dx
        cdfp = np.cumsum(hp) * dx

        # |ΔCDF| and critical distance Dc
        Dn = np.abs(cdfp - cdfa)
        N = a.size
        Dc = _phi_asymptotic(N) / np.sqrt(N)

        # In reduced domain, Xmax-Xmin = 1 → Ac = Dc
        Ac = Dc

        # Riemann sums
        KSI  = 100.0 / Ac * (Dn.sum() * dx)
        OVER = 100.0 / Ac * (np.maximum(Dn - Dc, 0.0).sum() * dx)
        return KSI, OVER

    # --- normalize input into dict of models ----------------------------------
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

    # --- metrics for one model -------------------------------------------------
    def _metrics_one(pred, act):
        # Align and drop NaN pairs
        mask = ~np.isnan(pred) & ~np.isnan(act)
        pred, act = pred[mask], act[mask]
        n = len(act)
        if n == 0:
            raise ValueError("No valid pairs.")
        a_mean = act.mean()
        if a_mean == 0:
            # Relative metrics (NRMSE%, NMBE%, etc.) undefined if mean(actual)==0
            raise ValueError("Mean of actual is zero; relative metrics undefined.")

        # Residuals
        res = pred - act

        # --- Core errors -------------------------------------------------------
        rmse = np.sqrt(np.mean(res**2))           # ↓
        mbe  = np.mean(res)                       # → (closer to 0)
        mae  = np.mean(np.abs(res))               # ↓

        # --- Relative errors (guard actual==0) --------------------------------
        nz = act != 0
        if np.any(nz):
            rel = res[nz] / act[nz]
            mpe   = np.mean(rel) * 100.0          # → (closer to 0)
            mape  = np.mean(np.abs(rel)) * 100.0  # ↓
            ermax = np.max(np.abs(rel))           # ↓ (max relative error)
            mare  = np.mean(np.abs(rel))          # ↓
        else:
            mpe = mape = ermax = mare = np.nan

        # --- Normalized errors -------------------------------------------------
        nrmse = rmse / a_mean * 100.0             # ↓
        nmbe  = mbe  / a_mean * 100.0             # ↓ (bias%; sign shows over/under)
        rrmse = rmse / a_mean                     # ↓ (fraction)

        # --- Uncertainty -------------------------------------------------------
        sd  = np.std(res, ddof=1) if n > 1 else 0.0
        u95 = 1.96 * np.sqrt(sd**2 + rmse**2)     # ↓ (expanded uncertainty, 95%)

        # --- Bias significance -------------------------------------------------
        s2 = rmse**2 - mbe**2
        t_stats = np.sqrt((n - 1) * mbe**2 / s2) if s2 > 0 and n > 1 else np.nan  # ↓

        # --- Means difference --------------------------------------------------
        mabe = abs(a_mean - pred.mean())          # ↓

        # --- Association -------------------------------------------------------
        r2 = r2_score(act, pred) if np.std(act) > 0 else np.nan  # ↑
        # Pearson correlation (sign & strength)
        r  = np.corrcoef(act, pred)[0, 1] if np.std(act) > 0 and np.std(pred) > 0 else np.nan  # ↑

        # --- Class A additions -------------------------------------------------
        # StdDev ratio (spread match): closer to 1 is better ( → )
        sigma_a = np.std(act, ddof=1) if n > 1 else np.nan
        sigma_p = np.std(pred, ddof=1) if n > 1 else np.nan
        std_ratio = (sigma_p / sigma_a) if (sigma_a not in [0, np.nan]) else np.nan  # →

        # Slope of best-fit line: pred = a + b*actual; closer to 1 is better ( → )
        var_a = np.var(act, ddof=1) if n > 1 else 0.0
        if var_a > 0 and n > 1:
            cov_ap = np.cov(act, pred, ddof=1)[0, 1]
            slope = cov_ap / var_a                                                    # →
        else:
            slope = np.nan

        # --- Class B additions -------------------------------------------------
        sse = np.sum((act - pred)**2)
        sst = np.sum((act - a_mean)**2)
        nse = 1.0 - (sse / sst) if sst > 0 else np.nan   # ↑ (1 perfect; <0 worse than mean)

        # Willmott’s index of agreement (0–1), higher is better
        denom_wia = np.sum((np.abs(pred - a_mean) + np.abs(act - a_mean))**2)
        wia = 1.0 - (sse / denom_wia) if denom_wia > 0 else np.nan  # ↑

        # Legates–McCabe efficiency (can be <0); higher is better
        sae = np.sum(np.abs(act - pred))
        sad = np.sum(np.abs(act - a_mean))
        lce = 1.0 - (sae / sad) if sad > 0 else np.nan   # ↑

        # --- Distribution metrics (Gueymard/Espinar) --------------------------
        ksi, over = ksi_over(act, pred, nbins)           # ↓, ↓
        cpi = (ksi + over + 2 * nrmse) / 4.0 if np.isfinite(ksi) and np.isfinite(over) else np.nan  # ↓

        return {
            'NRMSE (%)':  nrmse,     # ↓
            'R2':         r2,        # ↑
            'R':          r,         # ↑ Pearson correlation
            'NMBE (%)':   nmbe,      # ↓ (bias %)
            'RMSE':       rmse,      # ↓
            'MBE':        mbe,       # → (closer to 0)
            'MAE':        mae,       # ↓
            'MPE (%)':    mpe,       # → (closer to 0)
            'MAPE (%)':   mape,      # ↓
            'U95':        u95,       # ↓
            'RRMSE':      rrmse,     # ↓
            'T-Stats':    t_stats,   # ↓
            'ERMAX':      ermax,     # ↓
            'MARE':       mare,      # ↓
            'MABE':       mabe,      # ↓
            'StdDev Ratio': std_ratio,  # → (σ_pred / σ_actual)
            'Slope':        slope,      # → (pred vs actual)
            'NSE':          nse,        # ↑
            'WIA':          wia,        # ↑
            'LCE':          lce,        # ↑
            'KSI (%)':      ksi,        # ↓
            'OVER (%)':     over,       # ↓
            'CPI':          cpi         # ↓
        }

    # --- compute all models ----------------------------------------------------
    rows = {}
    for name, preds in pred_dict.items():
        if preds.shape[0] != a.shape[0]:
            raise ValueError(f"Model '{name}': length mismatch.")
        rows[name] = _metrics_one(preds, a)
    df = pd.DataFrame.from_dict(rows, orient='index')

    # --- GPI across models (lower is better; only meaningful with ≥2 rows) ----
    if df.shape[0] >= 2:
        # Min–max normalize each metric across models, then subtract median position
        mins = df.min(axis=0)
        maxs = df.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        norm = (df - mins) / ranges
        med_norm = (df.median(axis=0) - mins) / ranges

        # Weights: +1 for most metrics (higher normalized → worse), -1 for R (higher is better)
        weights = pd.Series(1.0, index=df.columns)
        if 'R' in weights:
            weights['R'] = -1.0

        dev = norm.subtract(med_norm, axis=1)
        gpi = (dev * weights).sum(axis=1)
        df['GPI'] = gpi       # ↓ lower is better
    else:
        df['GPI'] = np.nan    # not meaningful for a single model

    # --- final column order ---------------------------------------------------
    ordered_cols = [
        'NRMSE (%)', 'R2', 'NMBE (%)', 'RMSE', 'MBE', 'MAE', 'MPE (%)',
        'MAPE (%)', 'U95', 'RRMSE', 'T-Stats', 'ERMAX', 'MARE', 'MABE',
        'GPI', 'KSI (%)', 'OVER (%)', 'CPI',
        # extras grouped at the end:
        'R', 'StdDev Ratio', 'Slope', 'NSE', 'WIA', 'LCE'
    ]
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    return df[ordered_cols]
