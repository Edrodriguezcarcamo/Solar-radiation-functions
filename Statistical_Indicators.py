# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:09:42 2022

@author: edrod
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def statistical_indicators(predictions, actual):
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
    r2 = r2_score(actual, predictions)
    mabe = np.abs(np.mean(actual) - np.mean(predictions))

    # Normalize the metrics to a 0-1 scale
    metrics = np.array([nrmse, nmbe, rmse, mbe, mae, mpe, mape, u95, rrmse, t_stats, ermax, mare, mabe, r2])
    normalized_metrics = (metrics - np.min(metrics)) / (np.max(metrics) - np.min(metrics))

    # Calculate GPI
    weights = np.ones_like(metrics)
    weights[-1] = -1  # Assuming the last metric is R2 and needs to be inversely weighted
    gpi = np.sum(weights * normalized_metrics)

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
        'GPI': [gpi]  # Added GPI
    }

    # Create DataFrame from dictionary
    result_df = pd.DataFrame(data)

    return result_df



