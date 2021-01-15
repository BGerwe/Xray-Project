import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from impedance import preprocessing
from impedance.models.circuits.circuits import CustomCircuit
from impedance.visualization import plot_nyquist

def get_all_files(file_dir, *args):#**kwargs): 
    
    sep ='*'
    end = "*.csv"

    match_str = file_dir + sep.join(args) + end
    all_files = glob.glob(match_str)

    return all_files
    
    
def extract_plot_Z(file, ax, label, scale=False):
    dat = pd.read_csv(file, skiprows=4)
    
    f, Z = extract_Z(dat)
    
    if scale:
        plot_nyquist(ax, Z/Z[-1], label=label)
    else:
        plot_nyquist(ax, Z, label=label)
    
    return f, Z


def extract_Z(dat):
    freq_key, Z_key = [], []
    for col in dat.columns:
        if (col == 'f') or (col == '% f'):
            freq_key.append(col)
        if (col == 'impedance from LT edge (s^3/(kg*m^2))'):
            #'impedance from LT gate (s^3/(kg*m^2))') or (col == 'impedance from LT holes (s^3/(kg*m^2))') or (col == 'Z_mc (s^3/(kg*m^2))'):
            Z_key.append(col)

    if (len(freq_key) == 0) or (len(freq_key) > 1):
        raise Exception(f'Unable to identify frequency header. Found:{freq_key}')
        
    if (len(Z_key) == 0) or (len(Z_key) > 1):
        raise Exception(f'Unable to identify frequency header. Found:{Z_key}')
        
    f = np.array(dat[freq_key[0]])
    
    Z_raw = dat[Z_key[0]]
    Z = np.array([complex(s.replace('i','j')) for s in Z_raw])
    
    init_len = len(f)
    f, Z = preprocessing.ignoreBelowX(f, Z)
    fin_len = len(f)
    print(f'Filtered {init_len-fin_len} data points with positive Im[Z]')
    
    f, Z = f[::-1], Z[::-1]

    return f, Z


def get_inits_R_RCPE_RC(f, Z, a_init):
    peak_inds = signal.find_peaks(-Z.imag)[0]

    R1_ind = peak_inds[0] + np.argmax(Z[peak_inds[0]:peak_inds[1]].imag)
    R1_init = Z[R1_ind].real - Z[0].real

    f_1 = f[peak_inds[0]]
    C1_init = 1 / (2 * np.pi * f_1 * R1_init)

    R2_init = Z[-1].real - Z[R1_ind].real

    f_2 = f[peak_inds[1]]
    C2_init = 1 / (2 * np.pi * f_2 * R2_init)

    inits = [Z[0].real, R1_init, C1_init, a_init, R2_init, C2_init]

    return inits


def get_inits_R_RCPE_RCPE_G(f, Z, tg, a_init):
    peak_inds = signal.find_peaks(-Z.imag)[0]
    

    R1_ind = peak_inds[0] + np.argmax(Z[peak_inds[0]:peak_inds[1]].imag)
    R1_init = Z[R1_ind].real - Z[0].real
    
    f_1 = f[peak_inds[0]]
    CPE1_init = 1 / (2 * np.pi * f_1 * R1_init)

    R2_init = (Z[-1].real - Z[R1_ind].real) / 2

    f_2 = f[peak_inds[1]]
    CPE2_init = 1 / (2 * np.pi * f_2 * R2_init)

    inits = [Z[0].real, R1_init, CPE1_init, a_init, R2_init, CPE2_init,
             a_init, R2_init, tg]
    
    return inits
    
    
def fit_R_RCPE_RC(file_name, ax, ind=None, return_Z=True, return_ax=True,
                  scale=False):
    """Returns: ax, f, Z, circ"""
#     print(f'Processing {file_name}\n')
    dat = pd.read_csv(file_name, skiprows=4)
    
    f, Z = extract_Z(dat)

    inits = get_inits_R_RCPE_RC(f, Z)
    
    circ_str = "R0-p(R1,CPE1)-p(R2,C2)"
    circ = CustomCircuit(circuit=circ_str, initial_guess=inits)
    circ.fit(f, Z)
 
    Z_sim = circ.predict(f)

    if scale:
        plot_nyquist(ax, Z / np.max(Z.real), fmt='x')
        plot_nyquist(ax, Z_sim / np.max(Z_sim.real), label=f'{ind}', fmt='-')
    else:
        plot_nyquist(ax, Z, fmt='x')
        plot_nyquist(ax, Z_sim, label=f'{ind}', fmt='-')

    returns = []
    if return_ax:
        returns.append(ax)
    else:
        plt.show()
        
    if return_Z:
        returns.append(f)
        returns.append(Z)
        returns.append(circ)

    return tuple(returns)


def fit_R_RCPE_RCPE_G(file_name, ax, tg, ind=None, return_Z=True,
                      return_ax=True, scale=False, a_init=0.8):
#     print(f'Processing {file_name}\n')
    """Returns: ax, f, Z, circ"""
    dat = pd.read_csv(file_name, skiprows=4)
    
    f, Z = extract_Z(dat)

    inits = get_inits_R_RCPE_RCPE_G(f, Z, tg, a_init)
    
    circ_str = "R0-p(R1,CPE1)-p(R2,CPE2)-G3"
    circ = CustomCircuit(circuit=circ_str, initial_guess=inits)
    circ.fit(f, Z)
 
    Z_sim = circ.predict(f)

    if scale:
        plot_nyquist(ax, Z / np.max(Z.real), fmt='x')
        plot_nyquist(ax, Z_sim / np.max(Z_sim.real), label=f'{ind}', fmt='-')
    else:
        plot_nyquist(ax, Z, fmt='x')
        plot_nyquist(ax, Z_sim, label=f'{ind}', fmt='-')

    returns = []
    if return_ax:
        returns.append(ax)
    else:
        plt.show()
        
    if return_Z:
        returns.append(f)
        returns.append(Z)
        returns.append(circ)

    return tuple(returns)


def process_n0(file_name, ax, return_Z=True, return_ax=True):
    print(f'Processing {file_name}\n')
    dat_n0 = pd.read_csv(file_name, skiprows=4)

    freq_key = []
    for col in dat_n0.columns:
        if (col == 'f') or (col == '% f'):
            freq_key.append(col)
    if (len(freq_key) == 0) or (len(freq_key) > 1):
        raise Exception(f'Unable to identify frequency header. Found:{freq_key}')

    f_n0 = np.array(dat_n0[freq_key[0]])
    Z_raw = dat_n0['impedance from LT holes_mask (s^3/(kg*m^2))']
    Z_n0 = np.array([complex(s.replace('i','j')) for s in Z_raw])

    init_len = len(f_n0)
    f_n0, Z_n0 = preprocessing.ignoreBelowX(f_n0, Z_n0)
    fin_len = len(f_n0)
    print(f'Filtered {init_len-fin_len} data points with positive Im[Z]')

    f_n0, Z_n0 = f_n0[::-1], Z_n0[::-1]
    
    circ_str = "R0-p(R1,C1)"

    R0_init = Z_n0[0].real
    
    f1 = f_n0[np.argmin(Z_n0.imag)]
    R1_init = (Z_n0[-1] - Z_n0[0]).real
    C1_init = 1 / (2 * np.pi * f1 * R1_init)
    
    inits = [R0_init, R1_init, C1_init]
    print(inits, '\n', f1)
    circ = CustomCircuit(circuit=circ_str, initial_guess=inits)
    circ.fit(f_n0, Z_n0)
    print(circ)

    plot_nyquist(ax, Z_n0)
    plot_nyquist(ax, circ.predict(f_n0))

    returns = []
    if return_ax:
        returns.append(ax)
    else:
        plt.show()
        
    if return_Z:
        returns.append(f_n0)
        returns.append(Z_n0)
        returns.append(circ)

    return tuple(returns)
