# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:09:42 2022

@author: edrod
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import ks_2samp

def statistical_indicators(predictions, actual):

    if len(predictions) != len(actual):
        raise ValueError("The length of predictions and actual must be the same.")
        
    actual_mean = np.mean(actual)
    if actual_mean == 0:
        raise ValueError("Mean of actual values is zero, which could lead to division by zero.")
        
    # Calculate statistical metrics
    nrmse = np.sqrt(((predictions - actual) ** 2).mean()) / np.mean(actual) * 100
    nmbe = np.mean(predictions - actual) / np.mean(actual) * 100
    rmse = np.sqrt(((predictions - actual) ** 2).mean())
    mbe = np.mean(predictions - actual)
    mae = np.mean(np.abs(actual - predictions))
    mpe = np.mean((predictions - actual) / actual) * 100
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    u95 = 1.96 * (predictions.std() ** 2 + rmse ** 2) ** 0.5
    rrmse = rmse / np.mean(actual)
    t_stats = (((len(predictions) - 1) * mbe ** 2) / (rmse ** 2 - mbe ** 2)) ** 0.5
    ermax = (np.abs((actual - predictions) / actual)).max()
    mare = np.mean(np.abs((predictions - actual) / actual))
    mabe = np.abs(np.mean(actual) - np.mean(predictions))
    r2 = r2_score(actual, predictions)
    
    # Normalize the metrics to a 0-1 scale
    metrics = np.array([nrmse, nmbe, rmse, mbe, mae, mpe, mape, u95, rrmse, t_stats, ermax, mare, mabe, r2])
    normalized_metrics = (metrics - np.min(metrics)) / (np.max(metrics) - np.min(metrics))

    # Calculate Global Performance Indicator (GPI) url: https://doi.org/10.1016/j.jclepro.2018.09.246
    weights = np.ones_like(metrics)
    weights[-1] = -1
    gpi = np.sum(weights * normalized_metrics)
    
    # Calculate KSI and OVER url: https://doi.org/10.1016/j.solener.2008.07.009
    def calculate_ksi_over(actual, predictions):
        n = len(actual)
        
        # Empirical cumulative distribution functions (ECDFs)
        ecdf_true = np.arange(1, n+1) / n
        ecdf_pred = np.arange(1, n+1) / n

        # Sort the values
        sorted_true = np.sort(actual)
        sorted_pred = np.sort(predictions)

        # Calculate Dn for KSI
        d, _ = ks_2samp(actual, predictions)
        
        # Characteristic quantity (Ac) calculation
        Ac = 1.63 * np.sqrt(n)
        
        # KSI calculation
        ksi = (d * 100) / Ac
        
        # OVER calculation
        over = np.sum(np.maximum(np.abs(ecdf_true - ecdf_pred) - d, 0)) * 100 / Ac

        return ksi, over

    # Calculate KSI and OVER
    ksi, over = calculate_ksi_over(actual, predictions)
    
    # Calculate CPI url: https://doi.org/10.1016/j.rser.2014.07.117
    cpi = (ksi + over + 2 * nrmse) / 4

    # Prepare results
    data = {
        'NRMSE (%)': [nrmse],
        'R2': [r2],
        'NMBE (%)': [nmbe],
        'RMSE': [rmse],
        'MBE': [mbe],
        'MAE': [mae],
        'MPE (%)': [mpe],
        'MAPE (%)': [mape],
        'U95': [u95],
        'RRMSE': [rrmse],
        'T-Stats': [t_stats],
        'ERMAX': [ermax],
        'MARE': [mare],
        'MABE': [mabe],
        'GPI': [gpi],
        'KSI (%)': [ksi],
        'OVER (%)': [over],
        'CPI': [cpi]
    }

    # Create DataFrame from dictionary
    result_df = pd.DataFrame(data)

    return result_df
