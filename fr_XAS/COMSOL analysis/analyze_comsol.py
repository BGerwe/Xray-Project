import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from impedance import preprocessing
from impedance.models.circuits.circuits import CustomCircuit
from impedance.visualization import plot_nyquist


def extract_Z(dat):
    f = np.array(dat['% f'])
    Z_raw = dat['impedance from LT gate (s^3/(kg*m^2))']
    Z = np.array([complex(s.replace('i','j')) for s in Z_raw])
    
    init_len = len(f)
    f, Z = preprocessing.ignoreBelowX(f, Z)
    fin_len = len(f)
    print(f'Filtered {init_len-fin_len} data points with positive Im[Z]')
    
    f, Z = f[::-1], Z[::-1]

    return f, Z


def get_inits(f, Z):
    peak_inds = signal.find_peaks(-Z.imag)[0]

    R1_ind = peak_inds[0]+np.argmax(Z[peak_inds[0]:peak_inds[1]].imag)
    R1_init = Z[R1_ind].real - Z[0].real

    f_1 = f[peak_inds[0]]
    C1_init = 1 / (2 * np.pi * f_1 * R1_init)

    R2_init = Z[-1].real - Z[R1_ind].real

    f_2 = f[peak_inds[1]]
    C2_init = 1 / (2 * np.pi * f_2 * R2_init)

    inits = [Z[0].real, R1_init, C1_init, 0.5, R2_init, C2_init]

    return inits


def process_n(file_name, ax, ind=None, return_Z=True, return_ax=True):
    print(f'Processing {file_name}\n')
    dat = pd.read_csv(file_name, skiprows=4)
    
    f, Z = extract_Z(dat)

    plot_nyquist(ax, Z, label=f'{ind}')

    circ_str = "R0-p(R1,CPE1)-p(R2,C2)"
    inits = get_inits(f, Z)
    circ = CustomCircuit(circuit=circ_str, initial_guess=inits)
    circ.fit(f, Z)

    plot_nyquist(ax, circ.predict(f), fmt='x')

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

    f_n0 = np.array(dat_n0['% f'])
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
