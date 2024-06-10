# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 18:37:08 2022

@author: edrod
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

"""
Separation models where the input data is a dataframe called df and must have the columns 
defined as follow,

df['Clearness_index'] = Clearness index
df['Daily_KT'] = Daily clearness index 
df['AST'] = AST 
df['Solar_altitud'] = Solar altitud
df['Persistence'] = Persistence
df['k_tc'] = Difference between clearness index of clear-sky GHI and clearness index
df['k_de'] = Portion of the diffuse fraction that is attributable to cloud enhancement events
df['Global_clear_sky_rad'] = Global clear sky rad

The output of the separation models is the diffuse fraction kd
"""

#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2009.07.018
def boland(df):
    
    def coef(x, b0, b1, b2, b3, b4, b5):
        return 1/(1+np.exp( b0+b1*x['Clearness_index']+b2*x['AST']+b3*x['Solar_altitud']+
                           b4*x['Daily_KT']+b5*x['Persistence'] ) )
    
    val, pcov = curve_fit(coef, df, df['Diffuse_Fraction'], method='trf')
    
    kd = 1/( 1+np.exp( val[0]+val[1]*df['Clearness_index']+val[2]*df['AST']+
                                    val[3]*df['Solar_altitud']+val[4]*df['Daily_KT']+
                                    val[5]*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.solener.2015.04.012
def engerer2(df):
    # Engerer2 coefficients for quasi-universal model
    c = 0.042336 
    b0 = -3.7912
    b1 = 7.5479
    b2 = -0.010036
    b3 = 0.003148
    b4 = -5.3146
    b5 = 1.7073
    
    kd = c + (1-c)/( 1+np.exp( b0+b1*df['Clearness_index']+b2*df['AST']+
                              b3*df['Solar_altitud']+b4*df['k_tc'] ) ) + b5*df['k_de']
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1063/1.5097014
def engerer4(df):
    # Engerer4 coefficients
    c = 0.10562 
    b0 = -4.1332
    b1 = 8.2578
    b2 = 0.010087
    b3 = 0.00088801
    b4 = -4.9302
    b5 = 0.44378
    
    kd = c + (1-c)/( 1+np.exp( b0+b1*df['Clearness_index']+b2*df['AST']+
                              b3*df['Solar_altitud']+b4*df['k_tc'] ) ) + b5*df['k_de']
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.rser.2022.112195
def yang4(df):
    # Yang4 coefficient for universal model
    c2 = 0.0361
    b0 = -0.5744
    b1 = 4.3184
    b2 = -0.0011
    b3 = 0.0004
    b4 = -4.7952
    b5 = 1.4414
    b6 = -2.8396
    
    x = pd.DataFrame(df['engerer2'].resample('H').mean()).rename(columns = {'engerer2' : 'engerer2_hourly'} )

    df = pd.merge_asof(df, x, right_index=True, left_index=True, direction='backward').fillna(method='ffill')
    
    kd = c2 + (1-c2)/( 1+np.exp( b0+b1*df['Clearness_index']+b2*df['AST']+
                                b3*df['Solar_altitud']+b4*df['k_tc']+b6*df['engerer2_hourly'] ) ) + b5*df['k_de']
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2018.02.107
def starke1(df):
    # Starke Coefficients for starke1 fitted in Australia
    b0 = -6.70407
    b1 = 6.99137
    b2 = -0.00048
    b3 = 0.03839
    b4 = 3.36003
    b5 = 1.97891
    b6 = -0.96758
    b7 = 0.15623
    b8 = -4.21938
    b9 = -0.00207
    b10 = -0.06604
    b11 = 2.12613
    b12 = 2.56515
    b13 = 1.62075
    
    Fd_starke1 = 1/(1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                     +b4*df['Daily_KT']+b5*df['Persistence']+b6*df['Global_clear_sky_rad']/277.78 ) )

    Fd_starke2 = 1/(1+np.exp(b7+b8*df['Clearness_index']+b9*df['AST']+b10*df['Solar_altitud']
                     +b11*df['Daily_KT']+b12*df['Persistence']+b13*df['Global_clear_sky_rad']/277.78 ) )
    
    kd = np.where((df['K_csi'] >= 1.05) & (df['Clearness_index'] > 0.65), 
                                Fd_starke2, Fd_starke1 )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2018.02.107
def starke2(df):
    # Starke Coefficients for starke2 fitted in Brasil
    b0 = -6.37505
    b1 = 6.68399
    b2 = 0.01667
    b3 = 0.02552
    b4 = 3.32837
    b5 = 1.97935
    b6 = -0.74116
    b7 = 0.19486
    b8 = -3.52376
    b9 = -0.00325
    b10 = -0.03737
    b11 = 2.68761
    b12 = 1.60666
    b13 = 1.07129
    
    Fd_starke1 = 1/(1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                     +b4*df['Daily_KT']+b5*df['Persistence']+b6*df['Global_clear_sky_rad']/277.78 ) )

    Fd_starke2 = 1/(1+np.exp(b7+b8*df['Clearness_index']+b9*df['AST']+b10*df['Solar_altitud']
                     +b11*df['Daily_KT']+b12*df['Persistence']+b13*df['Global_clear_sky_rad']/277.78 ) )
    
    kd = np.where((df['K_csi'] >= 1.05) & (df['Clearness_index'] > 0.65), 
                                Fd_starke2, Fd_starke1 )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2021.05.108
def starke3_A(df):
    # Starke Coefficients for tropical climate
    b0 = 0.29566
    b1 = -3.64571
    b2 = -0.00353
    b3 = -0.01721 
    b4 =  1.7119
    b5 = 0.79448
    b6 = 0.00271
    b7 = 1.38097
    b8 = -7.00586
    b9 = 6.35348
    b10 = -0.00087
    b11 = 0.00308
    b12 = 2.89595
    b13 = 1.13655
    b14 = -0.0013
    b15 =  2.75815
    
    Fd_starke1 = 1/(1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                     +b4*df['Daily_KT']+b5*df['Persistence']+b6*df['Global_clear_sky_rad']
                     +b7*df['Hourly_kT'] ) )

    Fd_starke2 = 1/(1+np.exp(b8+b9*df['Clearness_index']+b10*df['AST']+b11*df['Solar_altitud']
                     +b12*df['Daily_KT']+b13*df['Persistence']+b14*df['Global_clear_sky_rad']
                     +b15*df['Hourly_kT'] ) )


    kd = np.where((df['K_csi'] >= 1.05) & (df['Clearness_index'] > 0.75), 
                                Fd_starke1, Fd_starke2 )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2021.05.108
def starke3_B(df):
    # Starke Coefficients for dry climate
    b0 = -1.7463
    b1 = -2.20055
    b2 = 0.01182
    b3 = -0.03489 
    b4 =  2.46116
    b5 = 0.70287
    b6 = 0.00329
    b7 = 2.30316
    b8 = -6.53133
    b9 = 6.63995
    b10 = 0.01318
    b11 = -0.01043
    b12 = 1.73562
    b13 = 0.85521
    b14 = -0.0003
    b15 =  2.63141
    
    Fd_starke1 = 1/(1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                     +b4*df['Daily_KT']+b5*df['Persistence']+b6*df['Global_clear_sky_rad']
                     +b7*df['Hourly_kT'] ) )

    Fd_starke2 = 1/(1+np.exp(b8+b9*df['Clearness_index']+b10*df['AST']+b11*df['Solar_altitud']
                     +b12*df['Daily_KT']+b13*df['Persistence']+b14*df['Global_clear_sky_rad']
                     +b15*df['Hourly_kT'] ) )


    kd = np.where((df['K_csi'] >= 1.05) & (df['Clearness_index'] > 0.75), 
                                Fd_starke1, Fd_starke2 )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2021.05.108
def starke3_C(df):
    # Starke Coefficients for mild temperate climate
    b0 = -0.083
    b1 = -3.14711
    b2 = 0.00176
    b3 = -0.03354 
    b4 =  1.40264
    b5 = 0.81353
    b6 = 0.00343
    b7 = 1.95109
    b8 = -7.28853
    b9 = 7.15225
    b10 = 0.00384
    b11 = 0.02535
    b12 = 2.35926
    b13 = 0.83439
    b14 = -0.00327
    b15 =  3.19723
    
    Fd_starke1 = 1/(1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                     +b4*df['Daily_KT']+b5*df['Persistence']+b6*df['Global_clear_sky_rad']
                     +b7*df['Hourly_kT'] ) )

    Fd_starke2 = 1/(1+np.exp(b8+b9*df['Clearness_index']+b10*df['AST']+b11*df['Solar_altitud']
                     +b12*df['Daily_KT']+b13*df['Persistence']+b14*df['Global_clear_sky_rad']
                     +b15*df['Hourly_kT'] ) )


    kd = np.where((df['K_csi'] >= 1.05) & (df['Clearness_index'] > 0.75), 
                                Fd_starke1, Fd_starke2 )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2021.05.108
def starke3_D(df):
    # Starke Coefficients for snow climate
    b0 = 0.67867
    b1 = -3.79515
    b2 = -0.00176
    b3 = -0.03487 
    b4 =  1.33611
    b5 = 0.76322
    b6 = 0.00353
    b7 = 1.82346
    b8 = -7.90856
    b9 = 7.63779
    b10 = 0.00145
    b11 = 0.10784
    b12 = 2.00908
    b13 = 1.12723
    b14 = -0.00889
    b15 =  3.72947
    
    Fd_starke1 = 1/(1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                     +b4*df['Daily_KT']+b5*df['Persistence']+b6*df['Global_clear_sky_rad']
                     +b7*df['Hourly_kT'] ) )

    Fd_starke2 = 1/(1+np.exp(b8+b9*df['Clearness_index']+b10*df['AST']+b11*df['Solar_altitud']
                     +b12*df['Daily_KT']+b13*df['Persistence']+b14*df['Global_clear_sky_rad']
                     +b15*df['Hourly_kT'] ) )


    kd = np.where((df['K_csi'] >= 1.05) & (df['Clearness_index'] > 0.75), 
                                Fd_starke1, Fd_starke2 )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2021.05.108
def starke3_E(df):
    # Starke Coefficients for polar climate
    b0 = 0.51643
    b1 = -5.32887
    b2 = -0.00196
    b3 = -0.07346 
    b4 =  1.6064
    b5 = 0.74681
    b6 = 0.00543
    b7 = 3.53205
    b8 = -11.70755
    b9 = 10.8476
    b10 = 0.00759
    b11 = 0.53397
    b12 = 1.76082
    b13 = 0.41495
    b14 = -0.03513
    b15 =  6.04835
    
    Fd_starke1 = 1/(1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                     +b4*df['Daily_KT']+b5*df['Persistence']+b6*df['Global_clear_sky_rad']
                     +b7*df['Hourly_kT'] ) )

    Fd_starke2 = 1/(1+np.exp(b8+b9*df['Clearness_index']+b10*df['AST']+b11*df['Solar_altitud']
                     +b12*df['Daily_KT']+b13*df['Persistence']+b14*df['Global_clear_sky_rad']
                     +b15*df['Hourly_kT'] ) )


    kd = np.where((df['K_csi'] >= 1.05) & (df['Clearness_index'] > 0.75), 
                                Fd_starke1, Fd_starke2 )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.rser.2019.04.055
def abreu_A(df):
    # Abreu Coefficients for tropical climate
    A = 11.59
    B = -6.14
    n = 1.87
    
    kd = ( 1+(A*(df['Clearness_index']-0.5)**2+B*(df['Clearness_index']-0.5)+1)**(-n) )**(-1/n)
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.rser.2019.04.055
def abreu_B(df):
    # Abreu Coefficients for dry (or arid) climate
    A = 11.39
    B = -6.25
    n = 1.86
    
    kd = ( 1+(A*(df['Clearness_index']-0.5)**2+B*(df['Clearness_index']-0.5)+1)**(-n) )**(-1/n)
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.rser.2019.04.055
def abreu_C(df):
    # Abreu Coefficients for mild temperate climate
    A = 10.79
    B = -5.87
    n = 2.24
    
    kd = ( 1+(A*(df['Clearness_index']-0.5)**2+B*(df['Clearness_index']-0.5)+1)**(-n) )**(-1/n)
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.rser.2019.04.055
def abreu_HighAlbedo(df):
    # Abreu Coefficients for snow and polar climates
    A = 7.83
    B = -4.59
    n = 3.25
    
    kd = ( 1+(A*(df['Clearness_index']-0.5)**2+B*(df['Clearness_index']-0.5)+1)**(-n) )**(-1/n)
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every1(df):
    # Every Coefficients for the world version
    b0 = -6.862
    b1 = 9.068
    b2 = 0.01468
    b3 = -0.00472
    b4 = 1.703
    b5 = 1.084
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every2_Am(df):
    # Every Coefficients for Am climate
    b0 = -6.433
    b1 = 8.774
    b2 = -0.00044
    b3 = -0.00578
    b4 = 2.096
    b5 = 0.684
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every2_Aw(df):
    # Every Coefficients for Aw climate
    b0 = -6.047
    b1 = 7.540
    b2 = 0.00624
    b3 = -0.00299
    b4 = 2.077
    b5 = 1.208
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every2_BSh(df):
    # Every Coefficients for BSh climate
    b0 = -6.734
    b1 = 8.853
    b2 = 0.02454
    b3 = -0.00495
    b4 = 1.874
    b5 = 0.939
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every2_BSk(df):
    # Every Coefficients for BSk climate
    b0 = -7.310
    b1 = 10.089
    b2 = 0.01852
    b3 = -0.00693
    b4 = 1.296
    b5 = 1.114
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every2_BWh(df):
    # Every Coefficients for BWh climate
    b0 = -7.097
    b1 = 9.416
    b2 = 0.01254
    b3 = -0.00416
    b4 = 1.661
    b5 = 1.130
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every2_Cfa(df):
    # Every Coefficients for Cfa climate
    b0 = -6.484
    b1 = 8.301
    b2 = 0.01577
    b3 = -0.00338
    b4 = 1.607
    b5 = 1.307
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every2_Cfb(df):
    # Every Coefficients for Cfb climate
    b0 = -6.764
    b1 = 9.958
    b2 = 0.01271
    b3 = -0.01249
    b4 = 0.928
    b5 = 1.142
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every2_Csa(df):
    # Every Coefficients for Csa climate
    b0 = -7.099
    b1 = 10.152
    b2 = -0.00026
    b3 = -0.00744
    b4 = 1.147
    b5 = 1.184
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every2_Csb(df):
    # Every Coefficients for Csb climate
    b0 = -7.080
    b1 = 10.460
    b2 = 0.00964
    b3 = -0.01420
    b4 = 1.134
    b5 = 1.017
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.renene.2019.09.114
def every2_Other(df):
    # Every Coefficients for other climates
    b0 = -5.38
    b1 = 6.63
    b2 = 0.006
    b3 = -0.007
    b4 = 1.75
    b5 = 1.31
    
    kd = 1/( 1+np.exp(b0+b1*df['Clearness_index']+b2*df['AST']+b3*df['Solar_altitud']
                      +b4*df['Daily_KT']+b5*df['Persistence'] ) )
    
    return kd
#------------------------------------------------------------------------------
# url: https://doi.org/10.1016/j.solener.2019.01.029
def paulescu(df):
    b0 = 1.0119
    b1 = -0.0316
    b2 = -0.0294
    b3 = -1.6567
    b4 = 0.367
    b5 = 1.8982
    b6 = 0.734
    b7 = -0.8548
    b8 = 0.462
    
    kd = b0+b1*df['Clearness_index']+b2*df['Daily_KT']+b3*(df['Clearness_index']
                    -b4)*np.where(df['Clearness_index']>=b4,1,0)
    +b5*(df['Clearness_index']-b6)*np.where(df['Clearness_index']>=b6,1,0)
    +b7*(df['Daily_KT']-b8)*np.where(df['Daily_KT']>=b8,1,0)
    
    return kd










































