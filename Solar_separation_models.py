"""
Solar Radiation Separation Models

This module provides various models to calculate the diffuse fraction (kd) 
from global horizontal irradiance (GHI) measurements.

Required DataFrame columns:
    - Clearness_index: Instantaneous clearness index
    - Daily_KT: Daily clearness index
    - AST: Apparent Solar Time
    - Solar_altitud: Solar altitude angle
    - Persistence: Persistence parameter
    - k_tc: Difference between clear-sky and actual clearness index
    - k_de: Portion of diffuse fraction from cloud enhancement events
    - Global_clear_sky_rad: Global clear sky radiation
    - K_csi: Clear sky index (for Starke models)
    - Hourly_kT: Hourly clearness index (for Starke3 models)
    - Diffuse_Fraction: Actual diffuse fraction (only for Boland model calibration)

Author: edrod
Created: 2022-09-05
Refactored: 2025-10-02
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import List, Dict, Optional, Union
from enum import Enum


class ModelCategory(Enum):
    """Categories of separation models."""
    UNIVERSAL = "universal"
    REGIONAL = "regional"
    CLIMATE_SPECIFIC = "climate_specific"


class ClimateZone(Enum):
    """Köppen-Geiger climate classification zones."""
    # Main climate groups
    TROPICAL = "A"
    DRY = "B"
    MILD_TEMPERATE = "C"
    CONTINENTAL = "D"
    POLAR = "E"
    
    # Specific classifications for Every2
    AM = "Am"  # Tropical monsoon
    AW = "Aw"  # Tropical savanna
    BSH = "BSh"  # Hot semi-arid
    BSK = "BSk"  # Cold semi-arid
    BWH = "BWh"  # Hot desert
    CFA = "Cfa"  # Humid subtropical
    CFB = "Cfb"  # Oceanic
    CSA = "Csa"  # Hot-summer Mediterranean
    CSB = "Csb"  # Warm-summer Mediterranean


class SeparationModel:
    """Base class for separation models."""
    
    category: ModelCategory = ModelCategory.UNIVERSAL
    climate_zone: Optional[str] = None
    region: Optional[str] = None
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.Series:
        """Calculate diffuse fraction."""
        raise NotImplementedError
    
    @classmethod
    def get_info(cls) -> Dict:
        """Return model information."""
        return {
            'name': cls.__name__.replace('Model', '').lower(),
            'category': cls.category.value,
            'climate_zone': cls.climate_zone,
            'region': cls.region,
            'description': cls.__doc__.strip() if cls.__doc__ else ''
        }


class BolandModel(SeparationModel):
    """Boland logistic model with curve fitting (requires calibration data).
    
    Reference: https://doi.org/10.1016/j.renene.2009.07.018
    """
    
    category = ModelCategory.UNIVERSAL
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.Series:
        def logistic_func(x, b0, b1, b2, b3, b4, b5):
            return 1 / (1 + np.exp(
                b0 + b1*x['Clearness_index'] + b2*x['AST'] + 
                b3*x['Solar_altitud'] + b4*x['Daily_KT'] + b5*x['Persistence']
            ))
        
        if 'Diffuse_Fraction' not in df.columns:
            raise ValueError("Boland model requires 'Diffuse_Fraction' column for calibration")
        
        params, _ = curve_fit(logistic_func, df, df['Diffuse_Fraction'], method='trf')
        
        kd = 1 / (1 + np.exp(
            params[0] + params[1]*df['Clearness_index'] + params[2]*df['AST'] +
            params[3]*df['Solar_altitud'] + params[4]*df['Daily_KT'] + 
            params[5]*df['Persistence']
        ))
        
        return kd


class Engerer2Model(SeparationModel):
    """Engerer2 quasi-universal model.
    
    Reference: https://doi.org/10.1016/j.solener.2015.04.012
    """
    
    category = ModelCategory.UNIVERSAL
    
    PARAMS = {
        'c': 0.042336,
        'b0': -3.7912,
        'b1': 7.5479,
        'b2': -0.010036,
        'b3': 0.003148,
        'b4': -5.3146,
        'b5': 1.7073
    }
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.Series:
        p = Engerer2Model.PARAMS
        kd = p['c'] + (1 - p['c']) / (1 + np.exp(
            p['b0'] + p['b1']*df['Clearness_index'] + p['b2']*df['AST'] +
            p['b3']*df['Solar_altitud'] + p['b4']*df['k_tc']
        )) + p['b5']*df['k_de']
        
        return kd


class Engerer4Model(SeparationModel):
    """Engerer4 model.
    
    Reference: https://doi.org/10.1063/1.5097014
    """
    
    category = ModelCategory.UNIVERSAL
    
    PARAMS = {
        'c': 0.10562,
        'b0': -4.1332,
        'b1': 8.2578,
        'b2': 0.010087,
        'b3': 0.00088801,
        'b4': -4.9302,
        'b5': 0.44378
    }
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.Series:
        p = Engerer4Model.PARAMS
        kd = p['c'] + (1 - p['c']) / (1 + np.exp(
            p['b0'] + p['b1']*df['Clearness_index'] + p['b2']*df['AST'] +
            p['b3']*df['Solar_altitud'] + p['b4']*df['k_tc']
        )) + p['b5']*df['k_de']
        
        return kd


class Yang4Model(SeparationModel):
    """Yang4 universal model (automatically calculates engerer2 if not present).
    
    Reference: https://doi.org/10.1016/j.rser.2022.112195
    """
    
    category = ModelCategory.UNIVERSAL
    
    PARAMS = {
        'c2': 0.0361,
        'b0': -0.5744,
        'b1': 4.3184,
        'b2': -0.0011,
        'b3': 0.0004,
        'b4': -4.7952,
        'b5': 1.4414,
        'b6': -2.8396
    }
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.Series:
        # Calculate engerer2 if not present
        if 'engerer2' not in df.columns:
            df = df.copy()
            df['engerer2'] = Engerer2Model.calculate(df)
        
        # Calculate hourly average of engerer2
        hourly_avg = (df['engerer2'].resample('H').mean()
                      .rename('engerer2_hourly'))
        
        # Merge hourly data back to original resolution
        df_merged = pd.merge_asof(df, hourly_avg.to_frame(), 
                                   left_index=True, right_index=True, 
                                   direction='backward')
        
        # Fill any remaining NaN values
        if df_merged['engerer2_hourly'].isna().any():
            df_merged['engerer2_hourly'] = df_merged['engerer2_hourly'].fillna(
                df_merged['engerer2_hourly'].mean()
            )
        
        p = Yang4Model.PARAMS
        kd = p['c2'] + (1 - p['c2']) / (1 + np.exp(
            p['b0'] + p['b1']*df_merged['Clearness_index'] + p['b2']*df_merged['AST'] +
            p['b3']*df_merged['Solar_altitud'] + p['b4']*df_merged['k_tc'] + 
            p['b6']*df_merged['engerer2_hourly']
        )) + p['b5']*df_merged['k_de']
        
        return pd.Series(kd, index=df.index, name='yang4')


class StarkeModel(SeparationModel):
    """Base class for Starke models with dual logistic functions."""
    
    PARAMS = {}
    
    @classmethod
    def calculate(cls, df: pd.DataFrame) -> pd.Series:
        p = cls.PARAMS
        
        fd1 = 1 / (1 + np.exp(
            p['b0'] + p['b1']*df['Clearness_index'] + p['b2']*df['AST'] +
            p['b3']*df['Solar_altitud'] + p['b4']*df['Daily_KT'] + 
            p['b5']*df['Persistence'] + p['b6']*df['Global_clear_sky_rad']/277.78
        ))
        
        fd2 = 1 / (1 + np.exp(
            p['b7'] + p['b8']*df['Clearness_index'] + p['b9']*df['AST'] +
            p['b10']*df['Solar_altitud'] + p['b11']*df['Daily_KT'] + 
            p['b12']*df['Persistence'] + p['b13']*df['Global_clear_sky_rad']/277.78
        ))
        
        kd = np.where(
            (df['K_csi'] >= 1.05) & (df['Clearness_index'] > 0.65),
            fd2, fd1
        )
        
        return kd


class Starke1Model(StarkeModel):
    """Starke1 model fitted for Australia.
    
    Reference: https://doi.org/10.1016/j.renene.2018.02.107
    """
    
    category = ModelCategory.REGIONAL
    region = "Australia"
    
    PARAMS = {
        'b0': -6.70407, 'b1': 6.99137, 'b2': -0.00048, 'b3': 0.03839,
        'b4': 3.36003, 'b5': 1.97891, 'b6': -0.96758,
        'b7': 0.15623, 'b8': -4.21938, 'b9': -0.00207, 'b10': -0.06604,
        'b11': 2.12613, 'b12': 2.56515, 'b13': 1.62075
    }


class Starke2Model(StarkeModel):
    """Starke2 model fitted for Brazil.
    
    Reference: https://doi.org/10.1016/j.renene.2018.02.107
    """
    
    category = ModelCategory.REGIONAL
    region = "Brazil"
    
    PARAMS = {
        'b0': -6.37505, 'b1': 6.68399, 'b2': 0.01667, 'b3': 0.02552,
        'b4': 3.32837, 'b5': 1.97935, 'b6': -0.74116,
        'b7': 0.19486, 'b8': -3.52376, 'b9': -0.00325, 'b10': -0.03737,
        'b11': 2.68761, 'b12': 1.60666, 'b13': 1.07129
    }


class Starke3Model(SeparationModel):
    """Base class for Starke3 climate-specific models.
    
    Reference: https://doi.org/10.1016/j.renene.2021.05.108
    """
    
    category = ModelCategory.CLIMATE_SPECIFIC
    PARAMS = {}
    
    @classmethod
    def calculate(cls, df: pd.DataFrame) -> pd.Series:
        p = cls.PARAMS
        
        fd1 = 1 / (1 + np.exp(
            p['b0'] + p['b1']*df['Clearness_index'] + p['b2']*df['AST'] +
            p['b3']*df['Solar_altitud'] + p['b4']*df['Daily_KT'] + 
            p['b5']*df['Persistence'] + p['b6']*df['Global_clear_sky_rad'] +
            p['b7']*df['Hourly_kT']
        ))
        
        fd2 = 1 / (1 + np.exp(
            p['b8'] + p['b9']*df['Clearness_index'] + p['b10']*df['AST'] +
            p['b11']*df['Solar_altitud'] + p['b12']*df['Daily_KT'] + 
            p['b13']*df['Persistence'] + p['b14']*df['Global_clear_sky_rad'] +
            p['b15']*df['Hourly_kT']
        ))
        
        kd = np.where(
            (df['K_csi'] >= 1.05) & (df['Clearness_index'] > 0.75),
            fd1, fd2
        )
        
        return kd


class Starke3TropicalModel(Starke3Model):
    """Starke3 for tropical climate (Köppen A)."""
    
    climate_zone = "A"
    
    PARAMS = {
        'b0': 0.29566, 'b1': -3.64571, 'b2': -0.00353, 'b3': -0.01721,
        'b4': 1.7119, 'b5': 0.79448, 'b6': 0.00271, 'b7': 1.38097,
        'b8': -7.00586, 'b9': 6.35348, 'b10': -0.00087, 'b11': 0.00308,
        'b12': 2.89595, 'b13': 1.13655, 'b14': -0.0013, 'b15': 2.75815
    }


class Starke3DryModel(Starke3Model):
    """Starke3 for dry climate (Köppen B)."""
    
    climate_zone = "B"
    
    PARAMS = {
        'b0': -1.7463, 'b1': -2.20055, 'b2': 0.01182, 'b3': -0.03489,
        'b4': 2.46116, 'b5': 0.70287, 'b6': 0.00329, 'b7': 2.30316,
        'b8': -6.53133, 'b9': 6.63995, 'b10': 0.01318, 'b11': -0.01043,
        'b12': 1.73562, 'b13': 0.85521, 'b14': -0.0003, 'b15': 2.63141
    }


class Starke3MildModel(Starke3Model):
    """Starke3 for mild temperate climate (Köppen C)."""
    
    climate_zone = "C"
    
    PARAMS = {
        'b0': -0.083, 'b1': -3.14711, 'b2': 0.00176, 'b3': -0.03354,
        'b4': 1.40264, 'b5': 0.81353, 'b6': 0.00343, 'b7': 1.95109,
        'b8': -7.28853, 'b9': 7.15225, 'b10': 0.00384, 'b11': 0.02535,
        'b12': 2.35926, 'b13': 0.83439, 'b14': -0.00327, 'b15': 3.19723
    }


class Starke3SnowModel(Starke3Model):
    """Starke3 for snow/continental climate (Köppen D)."""
    
    climate_zone = "D"
    
    PARAMS = {
        'b0': 0.67867, 'b1': -3.79515, 'b2': -0.00176, 'b3': -0.03487,
        'b4': 1.33611, 'b5': 0.76322, 'b6': 0.00353, 'b7': 1.82346,
        'b8': -7.90856, 'b9': 7.63779, 'b10': 0.00145, 'b11': 0.10784,
        'b12': 2.00908, 'b13': 1.12723, 'b14': -0.00889, 'b15': 3.72947
    }


class Starke3PolarModel(Starke3Model):
    """Starke3 for polar climate (Köppen E)."""
    
    climate_zone = "E"
    
    PARAMS = {
        'b0': 0.51643, 'b1': -5.32887, 'b2': -0.00196, 'b3': -0.07346,
        'b4': 1.6064, 'b5': 0.74681, 'b6': 0.00543, 'b7': 3.53205,
        'b8': -11.70755, 'b9': 10.8476, 'b10': 0.00759, 'b11': 0.53397,
        'b12': 1.76082, 'b13': 0.41495, 'b14': -0.03513, 'b15': 6.04835
    }


class AbreuModel(SeparationModel):
    """Base class for Abreu polynomial models.
    
    Reference: https://doi.org/10.1016/j.rser.2019.04.055
    """
    
    category = ModelCategory.CLIMATE_SPECIFIC
    PARAMS = {}
    
    @classmethod
    def calculate(cls, df: pd.DataFrame) -> pd.Series:
        p = cls.PARAMS
        kt_shifted = df['Clearness_index'] - 0.5
        
        kd = (1 + (p['A'] * kt_shifted**2 + p['B'] * kt_shifted + 1)**(-p['n']))**(-1/p['n'])
        
        return kd


class AbreuTropicalModel(AbreuModel):
    """Abreu model for tropical climate (Köppen A)."""
    
    climate_zone = "A"
    PARAMS = {'A': 11.59, 'B': -6.14, 'n': 1.87}


class AbreuDryModel(AbreuModel):
    """Abreu model for dry/arid climate (Köppen B)."""
    
    climate_zone = "B"
    PARAMS = {'A': 11.39, 'B': -6.25, 'n': 1.86}


class AbreuMildModel(AbreuModel):
    """Abreu model for mild temperate climate (Köppen C)."""
    
    climate_zone = "C"
    PARAMS = {'A': 10.79, 'B': -5.87, 'n': 2.24}


class AbreuHighAlbedoModel(AbreuModel):
    """Abreu model for snow and polar climates (Köppen D and E)."""
    
    climate_zone = "D/E"
    PARAMS = {'A': 7.83, 'B': -4.59, 'n': 3.25}


class EveryModel(SeparationModel):
    """Base class for Every logistic models."""
    
    PARAMS = {}
    
    @classmethod
    def calculate(cls, df: pd.DataFrame) -> pd.Series:
        p = cls.PARAMS
        kd = 1 / (1 + np.exp(
            p['b0'] + p['b1']*df['Clearness_index'] + p['b2']*df['AST'] +
            p['b3']*df['Solar_altitud'] + p['b4']*df['Daily_KT'] + 
            p['b5']*df['Persistence']
        ))
        
        return kd


class Every1Model(EveryModel):
    """Every1 world version (universal).
    
    Reference: https://doi.org/10.1016/j.renene.2019.09.114
    """
    
    category = ModelCategory.UNIVERSAL
    
    PARAMS = {
        'b0': -6.862, 'b1': 9.068, 'b2': 0.01468,
        'b3': -0.00472, 'b4': 1.703, 'b5': 1.084
    }


class Every2AmModel(EveryModel):
    """Every2 for tropical monsoon climate (Köppen Am)."""
    
    category = ModelCategory.CLIMATE_SPECIFIC
    climate_zone = "Am"
    
    PARAMS = {
        'b0': -6.433, 'b1': 8.774, 'b2': -0.00044,
        'b3': -0.00578, 'b4': 2.096, 'b5': 0.684
    }


class Every2AwModel(EveryModel):
    """Every2 for tropical savanna climate (Köppen Aw)."""
    
    category = ModelCategory.CLIMATE_SPECIFIC
    climate_zone = "Aw"
    
    PARAMS = {
        'b0': -6.047, 'b1': 7.540, 'b2': 0.00624,
        'b3': -0.00299, 'b4': 2.077, 'b5': 1.208
    }


class Every2BShModel(EveryModel):
    """Every2 for hot semi-arid climate (Köppen BSh)."""
    
    category = ModelCategory.CLIMATE_SPECIFIC
    climate_zone = "BSh"
    
    PARAMS = {
        'b0': -6.734, 'b1': 8.853, 'b2': 0.02454,
        'b3': -0.00495, 'b4': 1.874, 'b5': 0.939
    }


class Every2BSkModel(EveryModel):
    """Every2 for cold semi-arid climate (Köppen BSk)."""
    
    category = ModelCategory.CLIMATE_SPECIFIC
    climate_zone = "BSk"
    
    PARAMS = {
        'b0': -7.310, 'b1': 10.089, 'b2': 0.01852,
        'b3': -0.00693, 'b4': 1.296, 'b5': 1.114
    }


class Every2BWhModel(EveryModel):
    """Every2 for hot desert climate (Köppen BWh)."""
    
    category = ModelCategory.CLIMATE_SPECIFIC
    climate_zone = "BWh"
    
    PARAMS = {
        'b0': -7.097, 'b1': 9.416, 'b2': 0.01254,
        'b3': -0.00416, 'b4': 1.661, 'b5': 1.130
    }


class Every2CfaModel(EveryModel):
    """Every2 for humid subtropical climate (Köppen Cfa)."""
    
    category = ModelCategory.CLIMATE_SPECIFIC
    climate_zone = "Cfa"
    
    PARAMS = {
        'b0': -6.484, 'b1': 8.301, 'b2': 0.01577,
        'b3': -0.00338, 'b4': 1.607, 'b5': 1.307
    }


class Every2CfbModel(EveryModel):
    """Every2 for oceanic climate (Köppen Cfb)."""
    
    category = ModelCategory.CLIMATE_SPECIFIC
    climate_zone = "Cfb"
    
    PARAMS = {
        'b0': -6.764, 'b1': 9.958, 'b2': 0.01271,
        'b3': -0.01249, 'b4': 0.928, 'b5': 1.142
    }


class Every2CsaModel(EveryModel):
    """Every2 for hot-summer Mediterranean climate (Köppen Csa)."""
    
    category = ModelCategory.CLIMATE_SPECIFIC
    climate_zone = "Csa"
    
    PARAMS = {
        'b0': -7.099, 'b1': 10.152, 'b2': -0.00026,
        'b3': -0.00744, 'b4': 1.147, 'b5': 1.184
    }


class Every2CsbModel(EveryModel):
    """Every2 for warm-summer Mediterranean climate (Köppen Csb)."""
    
    category = ModelCategory.CLIMATE_SPECIFIC
    climate_zone = "Csb"
    
    PARAMS = {
        'b0': -7.080, 'b1': 10.460, 'b2': 0.00964,
        'b3': -0.01420, 'b4': 1.134, 'b5': 1.017
    }


class Every2OtherModel(EveryModel):
    """Every2 for other/unspecified climates."""
    
    category = ModelCategory.CLIMATE_SPECIFIC
    climate_zone = "Other"
    
    PARAMS = {
        'b0': -5.38, 'b1': 6.63, 'b2': 0.006,
        'b3': -0.007, 'b4': 1.75, 'b5': 1.31
    }


class PaulescuModel(SeparationModel):
    """Paulescu piecewise linear model."""
    
    category = ModelCategory.UNIVERSAL
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.Series:
        kt = df['Clearness_index']
        daily_kt = df['Daily_KT']
        
        kd = (1.0119 - 0.0316*kt - 0.0294*daily_kt
              - 1.6567*(kt - 0.367)*np.where(kt >= 0.367, 1, 0)
              + 1.8982*(kt - 0.734)*np.where(kt >= 0.734, 1, 0)
              - 0.8548*(daily_kt - 0.462)*np.where(daily_kt >= 0.462, 1, 0))
        
        return kd


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY: Dict[str, type[SeparationModel]] = {
    'boland': BolandModel,
    'engerer2': Engerer2Model,
    'engerer4': Engerer4Model,
    'yang4': Yang4Model,
    'starke1': Starke1Model,
    'starke2': Starke2Model,
    'starke3_tropical': Starke3TropicalModel,
    'starke3_dry': Starke3DryModel,
    'starke3_mild': Starke3MildModel,
    'starke3_snow': Starke3SnowModel,
    'starke3_polar': Starke3PolarModel,
    'abreu_tropical': AbreuTropicalModel,
    'abreu_dry': AbreuDryModel,
    'abreu_mild': AbreuMildModel,
    'abreu_high_albedo': AbreuHighAlbedoModel,
    'every1': Every1Model,
    'every2_am': Every2AmModel,
    'every2_aw': Every2AwModel,
    'every2_bsh': Every2BShModel,
    'every2_bsk': Every2BSkModel,
    'every2_bwh': Every2BWhModel,
    'every2_cfa': Every2CfaModel,
    'every2_cfb': Every2CfbModel,
    'every2_csa': Every2CsaModel,
    'every2_csb': Every2CsbModel,
    'every2_other': Every2OtherModel,
    'paulescu': PaulescuModel,
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def calculate_diffuse_fraction(df: pd.DataFrame, model_name: str) -> pd.Series:
    """
    Calculate diffuse fraction using a single specified model.
    
    Args:
        df: DataFrame with required columns (see module docstring)
        model_name: Name of the model to use (see list_models())
    
    Returns:
        Series containing calculated diffuse fraction (kd)
    
    Raises:
        ValueError: If model_name is not recognized
        
    Example:
        >>> kd = calculate_diffuse_fraction(df, 'engerer2')
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    result = model_class.calculate(df)
    
    # Ensure result is always a pandas Series
    if isinstance(result, np.ndarray):
        result = pd.Series(result, index=df.index, name=f'kd_{model_name}')
    elif isinstance(result, pd.Series):
        result.name = f'kd_{model_name}'
    
    return result


def calculate_multiple_models(df: pd.DataFrame, 
                              model_names: List[str],
                              prefix: str = 'kd_') -> pd.DataFrame:
    """
    Calculate diffuse fraction using multiple specified models.
    
    Args:
        df: DataFrame with required columns
        model_names: List of model names to calculate
        prefix: Prefix for column names in output DataFrame
    
    Returns:
        DataFrame with one column per model containing kd values
        
    Example:
        >>> models = ['engerer2', 'starke1', 'every1']
        >>> kd_df = calculate_multiple_models(df, models)
    """
    results = {}
    
    for model_name in model_names:
        try:
            results[f"{prefix}{model_name}"] = calculate_diffuse_fraction(df, model_name)
        except Exception as e:
            print(f"Warning: Model '{model_name}' failed: {e}")
            results[f"{prefix}{model_name}"] = np.nan
    
    return pd.DataFrame(results, index=df.index)


def calculate_all_models(df: pd.DataFrame,
                        climate_zone: Optional[str] = None,
                        include_universal: bool = True,
                        include_regional: bool = False,
                        prefix: str = 'kd_') -> pd.DataFrame:
    """
    Calculate diffuse fraction using all applicable models.
    
    Args:
        df: DataFrame with required columns
        climate_zone: Köppen-Geiger climate code (e.g., 'A', 'BWh', 'Cfa')
                     If None, only universal models are used
        include_universal: Include universal models (default: True)
        include_regional: Include regional models (default: False)
        prefix: Prefix for column names in output DataFrame
    
    Returns:
        DataFrame with one column per applicable model
        
    Example:
        >>> # All universal models only
        >>> kd_df = calculate_all_models(df)
        
        >>> # All models for tropical climate
        >>> kd_df = calculate_all_models(df, climate_zone='A')
        
        >>> # Climate-specific only (no universal models)
        >>> kd_df = calculate_all_models(df, climate_zone='Cfa', 
        ...                               include_universal=False)
    """
    models_to_run = []
    
    # Add universal models
    if include_universal:
        models_to_run.extend(get_models_by_category('universal'))
    
    # Add regional models
    if include_regional:
        models_to_run.extend(get_models_by_category('regional'))
    
    # Add climate-specific models
    if climate_zone:
        climate_models = get_models_by_climate(climate_zone)
        models_to_run.extend(climate_models)
    
    # Remove duplicates while preserving order
    models_to_run = list(dict.fromkeys(models_to_run))
    
    return calculate_multiple_models(df, models_to_run, prefix=prefix)


def get_models_by_category(category: str) -> List[str]:
    """
    Get list of model names by category.
    
    Args:
        category: 'universal', 'regional', or 'climate_specific'
    
    Returns:
        List of model names in the specified category
        
    Example:
        >>> universal_models = get_models_by_category('universal')
        >>> print(universal_models)
        ['engerer2', 'engerer4', 'every1', 'paulescu']
    """
    category_enum = ModelCategory(category.lower())
    models = []
    
    for name, model_class in MODEL_REGISTRY.items():
        if model_class.category == category_enum:
            models.append(name)
    
    return sorted(models)


def get_models_by_climate(climate_zone: str) -> List[str]:
    """
    Get list of model names applicable to a specific climate zone.
    
    Args:
        climate_zone: Köppen-Geiger climate code (e.g., 'A', 'BWh', 'Cfa')
    
    Returns:
        List of applicable model names for the climate zone
        
    Example:
        >>> models = get_models_by_climate('A')
        >>> print(models)
        ['starke3_tropical', 'abreu_tropical', 'every2_am', 'every2_aw']
    """
    models = []
    main_climate = climate_zone[0] if climate_zone else None
    
    for name, model_class in MODEL_REGISTRY.items():
        if model_class.category != ModelCategory.CLIMATE_SPECIFIC:
            continue
        
        zone = model_class.climate_zone
        if not zone:
            continue
        
        # Exact match
        if zone == climate_zone:
            models.append(name)
        # Main climate group match (e.g., 'A' matches for climate 'Am')
        elif zone == main_climate:
            models.append(name)
        # Special cases
        elif zone == "D/E" and main_climate in ["D", "E"]:
            models.append(name)
        elif zone == "Other" and climate_zone not in [
            'Am', 'Aw', 'BSh', 'BSk', 'BWh', 'Cfa', 'Cfb', 'Csa', 'Csb'
        ]:
            models.append(name)
    
    return sorted(models)


def get_climate_specific_model(base_model: str, climate_zone: str) -> str:
    """
    Get the appropriate climate-specific model name for a base model family.
    
    Args:
        base_model: Base model family ('starke3', 'abreu', 'every2')
        climate_zone: Köppen-Geiger climate code
    
    Returns:
        Full model name with climate suffix
        
    Raises:
        ValueError: If base_model doesn't have climate variants or 
                   climate_zone is not recognized
    
    Example:
        >>> model = get_climate_specific_model('starke3', 'A')
        >>> print(model)
        'starke3_tropical'
    """
    climate_mappings = {
        'starke3': {
            'A': 'starke3_tropical',
            'B': 'starke3_dry',
            'C': 'starke3_mild',
            'D': 'starke3_snow',
            'E': 'starke3_polar',
        },
        'abreu': {
            'A': 'abreu_tropical',
            'B': 'abreu_dry',
            'C': 'abreu_mild',
            'D': 'abreu_high_albedo',
            'E': 'abreu_high_albedo',
        },
        'every2': {
            'Am': 'every2_am',
            'Aw': 'every2_aw',
            'BSh': 'every2_bsh',
            'BSk': 'every2_bsk',
            'BWh': 'every2_bwh',
            'Cfa': 'every2_cfa',
            'Cfb': 'every2_cfb',
            'Csa': 'every2_csa',
            'Csb': 'every2_csb',
        }
    }
    
    if base_model not in climate_mappings:
        raise ValueError(
            f"No climate variants for model '{base_model}'. "
            f"Available: {', '.join(climate_mappings.keys())}"
        )
    
    mapping = climate_mappings[base_model]
    
    if climate_zone not in mapping:
        if base_model == 'every2':
            return 'every2_other'
        raise ValueError(
            f"Unknown climate zone '{climate_zone}' for {base_model}. "
            f"Available: {', '.join(mapping.keys())}"
        )
    
    return mapping[climate_zone]


def list_models(category: Optional[str] = None,
               climate_zone: Optional[str] = None,
               detailed: bool = False) -> Union[List[str], pd.DataFrame]:
    """
    List available models with optional filtering.
    
    Args:
        category: Filter by category ('universal', 'regional', 'climate_specific')
        climate_zone: Filter by Köppen-Geiger climate zone
        detailed: If True, return DataFrame with full model information
    
    Returns:
        List of model names, or DataFrame if detailed=True
        
    Example:
        >>> # Simple list
        >>> print(list_models())
        
        >>> # Detailed information
        >>> df = list_models(detailed=True)
        >>> print(df)
        
        >>> # Filter by category
        >>> universal = list_models(category='universal')
        
        >>> # Filter by climate
        >>> tropical = list_models(climate_zone='A')
    """
    models = []
    
    for name, model_class in MODEL_REGISTRY.items():
        # Apply filters
        if category and model_class.category.value != category:
            continue
        
        if climate_zone:
            zone = model_class.climate_zone
            main_climate = climate_zone[0] if climate_zone else None
            
            # Skip if it's climate-specific but doesn't match
            if model_class.category == ModelCategory.CLIMATE_SPECIFIC:
                if zone and zone != climate_zone and zone != main_climate:
                    if not (zone == "D/E" and main_climate in ["D", "E"]):
                        if not (zone == "Other" and climate_zone not in [
                            'Am', 'Aw', 'BSh', 'BSk', 'BWh', 'Cfa', 'Cfb', 'Csa', 'Csb'
                        ]):
                            continue
        
        if detailed:
            info = model_class.get_info()
            info['model_name'] = name
            models.append(info)
        else:
            models.append(name)
    
    if detailed:
        df = pd.DataFrame(models)
        if not df.empty:
            df = df[['model_name', 'category', 'climate_zone', 'region', 'description']]
        return df
    else:
        return sorted(models)


def validate_dataframe(df: pd.DataFrame, 
                      required_columns: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Validate that DataFrame has required columns for separation models.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns. If None, checks basic columns.
    
    Returns:
        Dictionary with validation results for each column
        
    Example:
        >>> validation = validate_dataframe(df)
        >>> if not all(validation.values()):
        ...     print("Missing columns:", [k for k, v in validation.items() if not v])
    """
    if required_columns is None:
        required_columns = [
            'Clearness_index',
            'Daily_KT',
            'AST',
            'Solar_altitud',
            'Persistence',
        ]
    
    validation = {}
    for col in required_columns:
        validation[col] = col in df.columns
    
    return validation


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Demonstration of the separation models package.
    Shows different usage patterns for researchers.
    """
    print("=" * 70)
    print("SOLAR RADIATION SEPARATION MODELS")
    print("=" * 70)
    print()
    
    # Show all available models by category
    print("AVAILABLE MODELS BY CATEGORY:")
    print("-" * 70)
    
    for cat in ['universal', 'regional', 'climate_specific']:
        models = get_models_by_category(cat)
        print(f"\n{cat.upper()} ({len(models)} models):")
        for model in models:
            print(f"  - {model}")
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES:")
    print("-" * 70)
    
    print("""
# Example 1: Single model calculation
# ----------------------------------
kd = calculate_diffuse_fraction(df, 'engerer2')

# Example 2: Multiple specific models
# ----------------------------------
models = ['engerer2', 'starke1', 'every1']
kd_df = calculate_multiple_models(df, models)

# Example 3: All universal models
# ----------------------------------
kd_df = calculate_all_models(df, include_universal=True)

# Example 4: Climate-specific models (tropical)
# ----------------------------------
kd_df = calculate_all_models(df, climate_zone='A')

# Example 5: Everything for a specific climate
# ----------------------------------
kd_df = calculate_all_models(df, 
                             climate_zone='Cfa',
                             include_universal=True,
                             include_regional=True)

# Example 6: Get the right model for your climate
# ----------------------------------
model_name = get_climate_specific_model('starke3', 'A')
kd = calculate_diffuse_fraction(df, model_name)

# Example 7: List models for your climate
# ----------------------------------
my_models = list_models(climate_zone='BWh')
print(f"Applicable models: {my_models}")

# Example 8: Detailed model information
# ----------------------------------
model_info = list_models(detailed=True)
print(model_info)

# Example 9: Validate your DataFrame
# ----------------------------------
validation = validate_dataframe(df)
if not all(validation.values()):
    missing = [col for col, valid in validation.items() if not valid]
    print(f"Missing columns: {missing}")
""")
    
    print("\n" + "=" * 70)
    print("CLIMATE ZONE GUIDE (Köppen-Geiger):")
    print("-" * 70)
    print("""
Main Groups:
  A - Tropical climates
  B - Dry (arid and semi-arid) climates
  C - Temperate climates
  D - Continental/Snow climates
  E - Polar climates

Specific Zones (for Every2 models):
  Am  - Tropical monsoon
  Aw  - Tropical savanna
  BSh - Hot semi-arid
  BSk - Cold semi-arid
  BWh - Hot desert
  Cfa - Humid subtropical
  Cfb - Oceanic
  Csa - Hot-summer Mediterranean
  Csb - Warm-summer Mediterranean
""")
    
    print("\n" + "=" * 70)
    print("For more information, see module docstring and function help().")
    print("=" * 70)


if __name__ == '__main__':
    main()