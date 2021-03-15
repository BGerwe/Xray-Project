import numpy as np

from impedance.models.circuits import CustomCircuit
from impedance import preprocessing

def read_EIS_data(direc, f_name, z_format=preprocessing.readZPlot):
    """
    Import EIS data with specified impedance.py function and enforce sorting high to low."""
    f, Z = z_format(direc + f_name)
    # enforce frequencies from high to low
    Z = Z[np.argsort(f)[::-1]]
    f = f[np.argsort(f)[::-1]]

    f, Z = preprocessing.ignoreBelowX(f, Z)
    return f, Z


def c_chem(delta):
    """
    :param delta:
    """
    d = delta
    F = 96485
    R = 8.314
    T = 973
    Vm = 33.7
    cfit = 228000
    
    return 4 * F**2 / (Vm * (cfit + R * T / d))

def ASR(ao, xvo, tg):
    """
    :param ao:
    :param xvo:
    :param tg:
    """

    F = 96485
    R = 8.314
    T = 973
    co = 3 / 33.7
    L = 630e-7
    return R * T / (4 * F**2) * ao * tg / (co * xvo * L)

def Rg(ao, xvo, tg, ld, w):
    """
    :param ao:
    :param xvo:
    :param tg:
    :param ld:
    :param w:
    """
    F = 96485
    R = 8.314
    T = 973
    co = 3 / 33.7
    L = 630e-7
    
    return R * T / (4 * F**2) * 1 / (2 * w) * (tg / ld) * ao / (co * L * xvo)


def predict_branchB(params_df, gas, R0, freqs):
    """
    :param params_df: pd.DataFrame containing fit FR-XAS parameters and estimate circuit model values
    :param gas: int, index of gas measurement
    :param R0: float, series resistance attributed to current constriction
    """
    circ = 'R0-p(G1,p(R1,C1))'

    inits = [R0, params_df.Rg[gas], params_df.tg[gas], params_df.R_gate[gas], params_df.C_gate[gas]]
    branch_b_1 = CustomCircuit(circuit=circ, initial_guess = inits)
    Z_branch_b = branch_b_1.predict(freqs)
    return Z_branch_b


def predict_branchB_scaled_tg_Rg(params_df, tg_scale, rg_scale, gas, r_con, freqs):
    """
    :param params_df: pd.DataFrame containing fit FR-XAS parameters and estimate circuit model values
    :param tg_scale: float, multiplying factor on tg
    :param rg_scale: float, mupltiplying factor on rg
    :param gas: int, index of gas measurement
    :param r_con: float, series resistance attributed to current constriction
    """
    circ = 'R0-p(G1,p(R1,C1))'

    inits = [r_con, params_df.Rg[gas] * rg_scale, params_df.tg[gas] * tg_scale, params_df.R_gate[gas], params_df.C_gate[gas]]
    branch_b_1 = CustomCircuit(circuit=circ, initial_guess = inits)
    Z_branch_b = branch_b_1.predict(freqs)
    return Z_branch_b


def branchB_and_nh_parallel(Z_branch_b, Z_nh, geom_factor):
    """
    :param Z_branch_b: np.array, Impedance predicted as a resistor in series with Gerisher in parallel with RC
    :param Z_nh: np.array, Measured No Hole sample impedance
    :param geom_factor: float, to correct No Hole sample geometry to the patterned electrode
    """
    Y_nh = 1 / (Z_nh * geom_factor)
    Y_b = 1 / Z_branch_b
    
    Y_tot = Y_nh + Y_b
    Z_model = 1 / Y_tot
    return Z_model

def nh_and_b_para_2param_scaling(b_scale, nh_scale, Z_branch_b, Z_nh, area_ratio):
    """
    :param b_scale: float, scaling factor on branch B impedance
    :param nh_scale: float, scaling factor on NoHole sample impedance
    :param Z_branch_b: np.array, Impedance predicted as a resistor in series with Gerisher in parallel with RC
    :param Z_nh: np.array, Measured No Hole sample impedance
    :param area_ratio: float, ratio of NoHole sample to patterned electrode superficial areas

    :returns: np.array, predicted impedance of proposed circuit model
    """
    Y_nh = 1 / (Z_nh * area_ratio * nh_scale)
    Y_b = 1 / (Z_branch_b * b_scale)
    
    Y_tot = Y_nh + Y_b
    Z_model = 1 / Y_tot
    return Z_model  


def minimize_geom_factor(geom_factor, Z_measured, Z_branch_b, Z_nh):
    """
    Least squares regression to find geometry correction factor for No Hole sample EIS
    that minimizes residuals between predicted model impedance and measurements.
    """
    Z_model = branchB_and_nh_parallel(Z_branch_b, Z_nh, geom_factor)
    
    residuals_real = np.abs((Z_model.real - Z_measured.real)) / np.abs(Z_measured)
    residuals_imag = np.abs((Z_model.imag - Z_measured.imag)) / np.abs(Z_measured)
    
    sum_res = np.sum(residuals_real + residuals_imag)
    return sum_res


def minimize_geom_factor_all(geom_factor, Zs_measured, Zs_branch_b, Zs_nh):
    """
    Same as minimize_geom_factor, but enforces same geometry correction factor across all datasets.
    """
    Zs_model = []
    res_reals, res_imags = 0, 0
    for Z_branch_b, Z_nh, Z_measured in zip(Zs_branch_b, Zs_nh, Zs_measured):
        Z_model = branchB_and_nh_parallel(Z_branch_b, Z_nh, geom_factor)
        Zs_model.append(Z_model)
    
        res_reals += np.sum(np.abs((Z_model.real - Z_measured.real)) / np.abs(Z_measured))
        res_imags += np.sum(np.abs((Z_model.imag - Z_measured.imag)) / np.abs(Z_measured))
    
    sum_res = np.sum(res_reals + res_imags)
    return sum_res


def minimize_2param_scaling(scaling_params, Zs_measured, Zs_nh, nh_freqs, area_ratio,
                            r_con, params_df):
    """
    Adjust scaling factors on NoHole sample and predicted branch B impedance to fit patterned
    electrode measurements.
    :param scaling_params: list, first element is nh_scale, second element is  b_scale
    :param Zs_measured: list of np.array, measured patterned electrode EIS data
    :param Zs_nh: list of np.array, measured NoHole sample EIS data
    :param nh_freqs: list of np.array, measured NoHole sample EIS data frequency lists
    :param area_ratio: float, ratio of NoHole sample to patterned electrode superficial areas
    :param r_con: float, estimated current constriction resistance
    :param params_df: pd.DataFrame containing fit FR-XAS parameters and estimate circuit model values
    """
    nh_scale = scaling_params[0]
    b_scale = scaling_params[1]

    Zs_model = []
    res_reals, res_imags = 0, 0
    for gas, (Z_nh, Z_measured, freqs) in enumerate(zip(Zs_nh, Zs_measured, nh_freqs)):
        Z_branch_b = predict_branchB(params_df, gas, r_con, freqs)
        Z_model = nh_and_b_para_2param_scaling(b_scale, nh_scale, Z_branch_b, Z_nh, area_ratio)
        Zs_model.append(Z_model)
    
        res_reals += np.sum(np.abs((Z_model.real - Z_measured.real)) / np.abs(Z_measured))
        res_imags += np.sum(np.abs((Z_model.imag - Z_measured.imag)) / np.abs(Z_measured))

    sum_res = np.sum(res_reals + res_imags)
    return sum_res


def minimize_2param_scaling_Rcon(scaling_params, Zs_measured, Zs_nh, nh_freqs,
                                 area_ratio, params_df):
    """
    Adjust scaling factors on NoHole sample, predicted branch B impedance, and r_con
    to fit patterned electrode measurements.
    :param scaling_params: list, first element is nh_scale, second element is b_scale
                           third element is r_con
    :param Zs_measured: list of np.array, measured patterned electrode EIS data
    :param Zs_nh: list of np.array, measured NoHole sample EIS data
    :param nh_freqs: list of np.array, measured NoHole sample EIS data frequency lists
    :param area_ratio: float, ratio of NoHole sample to patterned electrode superficial areas
    :param params_df: pd.DataFrame containing fit FR-XAS parameters and estimate circuit model values
    """
    nh_scale = scaling_params[0]
    b_scale = scaling_params[1]
    r_con = scaling_params[2]

    Zs_model = []
    res_reals, res_imags = 0, 0
    for gas, (Z_nh, Z_measured, freqs) in enumerate(zip(Zs_nh, Zs_measured, nh_freqs)):
        Z_branch_b = predict_branchB(params_df, gas, r_con, freqs)
        Z_model = nh_and_b_para_2param_scaling(b_scale, nh_scale, Z_branch_b, Z_nh, area_ratio)
        Zs_model.append(Z_model)

        res_reals += np.sum(np.abs((Z_model.real - Z_measured.real)) / np.abs(Z_measured))
        res_imags += np.sum(np.abs((Z_model.imag - Z_measured.imag)) / np.abs(Z_measured))

    sum_res = np.sum(res_reals + res_imags)
    return sum_res

def minimize_3param_scaling(scaling_params, Zs_measured, Zs_nh, nh_freqs, area_ratio, R0, params_df):
    """
    Same as minimize_geom_factor, but enforces same geometry correction factor across all datasets.
    
    :param scaling_params: list, first element is nh_scale, second is tg_scale, third is rg_scale
    :param Zs_measured: list of np.array, measured impedance of patterned electrode
    :param Zs_nh: list of np.array, measured impedance of NoHole sample
    :param nh_freqs: list of np.array, frequencies of NoHole sample measurements
    :param area_ratio: float, ratio of NoHole sample to patterned electrode superficial areas
    :param R0: float, estimated current constriction resistance
    :param params_df: pd.DataFrame, contains relevant FR-XAS fitting parameters, and predicted Rg
                      from 1-D model

    :returns: float, sum of squared residuals for both real and imaginary parts, and for all data sets
    """
    nh_scale = scaling_params[0]
    tg_scale = scaling_params[1]
    rg_scale = scaling_params[2]
    
    Zs_model = []
    res_reals, res_imags = 0, 0
    for gas, (Z_nh, Z_measured, freqs) in enumerate(zip(Zs_nh, Zs_measured, nh_freqs)):
        Z_branch_b = predict_branchB_scaled_tg_Rg(params_df, tg_scale, rg_scale, gas, R0, freqs)
        Z_model =  nh_and_b_para_2param_scaling(1, nh_scale, Z_branch_b, Z_nh, area_ratio)
        Zs_model.append(Z_model)
    
        res_reals += np.sum(np.abs((Z_model.real - Z_measured.real)) / np.abs(Z_measured))
        res_imags += np.sum(np.abs((Z_model.imag - Z_measured.imag)) / np.abs(Z_measured))
    
    sum_res = np.sum(res_reals + res_imags)
    return sum_res


def minimize_3param_scaling_Rcon(scaling_params, Zs_measured, Zs_nh, nh_freqs, area_ratio, params_df):
    """
    Same as minimize_geom_factor, but enforces same geometry correction factor across all datasets.
    
    :param scaling_params: list, first element is nh_scale, second is tg_scale, third is rg_scale
                           fourth is r_con
    :param Zs_measured: list of np.array, measured impedance of patterned electrode
    :param Zs_nh: list of np.array, measured impedance of NoHole sample
    :param nh_freqs: list of np.array, frequencies of NoHole sample measurements
    :param area_ratio: float, ratio of NoHole sample to patterned electrode superficial areas
    :param params_df: pd.DataFrame, contains relevant FR-XAS fitting parameters, and predicted Rg
                      from 1-D model

    :returns: float, sum of squared residuals for both real and imaginary parts, and for all data sets
    """
    nh_scale = scaling_params[0]
    tg_scale = scaling_params[1]
    rg_scale = scaling_params[2]
    r_con = scaling_params[3]
    
    Zs_model = []
    res_reals, res_imags = 0, 0
    for gas, (Z_nh, Z_measured, freqs) in enumerate(zip(Zs_nh, Zs_measured, nh_freqs)):
        Z_branch_b = predict_branchB_scaled_tg_Rg(params_df, tg_scale, rg_scale, gas, r_con, freqs)
        Z_model =  nh_and_b_para_2param_scaling(1, nh_scale, Z_branch_b, Z_nh, area_ratio)
        Zs_model.append(Z_model)
    
        res_reals += np.sum(np.abs((Z_model.real - Z_measured.real)) / np.abs(Z_measured))
        res_imags += np.sum(np.abs((Z_model.imag - Z_measured.imag)) / np.abs(Z_measured))
    
    sum_res = np.sum(res_reals + res_imags)
    return sum_res
