import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import gif
from larch import io, xafs


def colorfun(v, v_max=1, v_min=-1):
#     print(V, Max)
    if v>0:
        R = v / v_max
        G = 0
        B = 0
    else:
        R = 0
        G = 0
        B = v / v_min
    return (R,G,B)

def format_XANES(energies, Irs, etas, step=1, start_ind=0, stop_ind=None, marker='.',
                 inset_dict={}, startE=7705, stopE=7730, size=(9, 6),
                 plot_data=True):

    def format_func(value, tick_number):
        return f'{value:.1f}'

    if not inset_dict:
        inset_dict = {'start_energy':7718, 'stop_energy':7720, 'x1_adj': 0, 'x2_adj': 0, 'x_interval':0.1, 'y1_adj': 0,
                      'y2_adj': 0, 'y_interval': 0.02, 'x_lateral': 0,
                      'y_vertical':0}

    x1_adj = inset_dict['x1_adj']
    x2_adj = inset_dict['x2_adj']
    x_interval = inset_dict['x_interval']
    y1_adj = inset_dict['y1_adj']
    y2_adj = inset_dict['y2_adj']
    y_interval = inset_dict['y_interval']
    x_lat = inset_dict['x_lateral']
    y_vert = inset_dict['y_vertical']

    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal'}
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['axes.labelsize'] = 16


    # Finding max and min of Ir across all data sets for setting plot limits
    Ir_abs_max = np.max(Irs)
    Ir_abs_min = np.round(np.min(Irs[np.argwhere(energies>=startE)[0][0]]), 2)

    # Finding max and min of eta (or voltage) for setting color bar limits
    max_eta = np.max(etas)
    min_eta = np.min(etas)

    ## Plotting full XANES spectra
    fig = plt.figure(constrained_layout=False, figsize=size)
    gs = fig.add_gridspec(7, 25)
    f_ax1 = fig.add_subplot(gs[:, :-1])
    f_ax2=fig.add_subplot(gs[0:4,3:13])

    ## Choosing plot limits for inset
    xind1=np.argwhere(energies>=inset_dict['start_energy'])[0]
    xind2=np.argwhere(energies<=inset_dict['stop_energy'])[-1]

    x1=round(energies[xind1][0])
    x2=round(energies[xind2][0])

    y1=round(Irs[xind1,0][0]*.95,2)
    y2=round(Irs[xind2,0][0]*1.05,2)
    
    if stop_ind is None:
        stop_ind = Irs.shape[-1]

        
    if plot_data:
        for n in range(start_ind, stop_ind, step):
            color = colorfun(etas[n], max_eta, min_eta)
            fig = plot_XANES_data(fig, energies, Irs[:, n], color=color)

    f_ax1.set(xlim=[startE,stopE], ylim=[.97 * Ir_abs_min, 1.05 * Ir_abs_max])
    f_ax1.set_ylabel(r'normalized $\mu$(E)  / a.u.', **title_font)
    f_ax1.set_xlabel(r'E  /  eV', **title_font)

    f_ax2.set(xlim=[x1+x_lat+x1_adj, x2+x_lat+x2_adj], 
          xticks=np.arange(x1+x1_adj, x2+x2_adj, x_interval),
          ylim=[y1+y_vert+y1_adj, y2+y_vert+y2_adj], 
          yticks=np.arange(y1+y1_adj, y2+y2_adj,y_interval))

    f_ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    f_ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    f_ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func))


    # Plotting colorbar
    fig = add_colorbar(fig, gs, max_eta, min_eta)
    
    return fig


def plot_XANES_data(fig, energies, Irs, color=(0,0,0)):
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]
    
    # Main Spectra
    ax1.plot(energies, Irs, color=color,
         linestyle='-',linewidth=.3, marker='', markersize=3)
    # Plot inset
    ax2.plot(energies, Irs, color=color,
             linestyle='-',linewidth=1, marker='', markersize=3)
    return fig


@gif.frame
def XANES_gif(fig, energies, Irs, color=(0,0,0)):
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]

    # Main Spectra
    ax1.plot(energies, Irs, color=color,
         linestyle='-',linewidth=1.5, marker='', markersize=3)
    # Plot inset
    ax2.plot(energies, Irs, color=color,
             linestyle='-',linewidth=1.5, marker='', markersize=3)


def format_diffXANES(energies, Irs, etas, start_ind=0, stop_ind=None, step=1, startE=7700, stopE=7735, size=(9,6), plot_data=True):
    fig = plt.figure(constrained_layout=False, figsize=size)
    gs = fig.add_gridspec(7, 25)
    ax = fig.add_subplot(gs[:, :-1])

    max_eta = np.max(etas-etas[start_ind])
    min_eta = np.min(etas-etas[start_ind])
#     color_max = np.max(etas-etas[start_ind])
#     color_min = np.min(etas-etas[start_ind])
#     color = colorfun(eta_diff, v_max=color_max, v_min=color_min)
    if stop_ind is None:
        stop_ind = Irs.shape[-1]

    for n in np.arange(start_ind, stop_ind, step):
        if plot_data:
            color = colorfun(etas[n] - etas[start_ind], v_max=max_eta, v_min=min_eta)
            plot_diffXANES(fig, energies, Irs, n, start_ind, color)

    Ir_diff = Irs - Irs[:, start_ind,None]
    ylim = np.round(1.05 * np.max(np.abs([Ir_diff.min(), Ir_diff.max()])), 3)
    
    ax.set_xlim(startE, stopE)
    ax.set_ylim(-ylim, ylim)
    n_ticks = 5
    ax.set_yticks(np.round(np.arange(-ylim, ylim*1.01, 2*ylim/(n_ticks-1)), 3))
    
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal'}
    ax.set_ylabel(r'$\Delta$ $\mu_{norm}$  / a.u.', labelpad=-10, **title_font)
    ax.set_xlabel(r'Energy  /  eV', **title_font)  

    # Plotting colorbar
    add_colorbar(fig, gs, max_eta, min_eta)

    return fig


def plot_diffXANES(fig, energies, Irs, n, start_ind, color=(0,0,0)):
    ax = fig.axes[0]

    Ir_diff = Irs[:, n] - Irs[:, start_ind]
    ax.plot(energies, Ir_diff, color=color)

    return fig


@gif.frame
def diffXANES_gif(fig, energies, Irs, n, start_ind, color=(0,0,0)):
    ax = fig.axes[0]

    Ir_diff = Irs[:, n] - Irs[:, start_ind]
    ax.plot(energies, Ir_diff, color=color)


def plot_lissajou(energies, Irs, etas, plot_e=7719.8, start_ind=0, stop_ind=None, step=1, fig=None, **kwargs):
    Ir_diffs, eta_diffs = [], []
    if stop_ind is None:
        stop_ind = Irs.shape[-1]

    for n in np.arange(start_ind, stop_ind, step):
        try:
            Ir_diff = Irs[:, n] - Irs[:, start_ind]
            Ir_diffs.append(Ir_diff)
            eta_diff = etas[n] - etas[start_ind]
            eta_diffs.append(eta_diff)
        except:
            pass
    print(n, Irs.shape[-1], etas[n])
    Ir_diffs = np.array(Ir_diffs)
    eta_diffs = np.array(eta_diffs) * 1000
    print(np.min(eta_diffs), np.max(eta_diffs))
    e_ind = np.argwhere(np.isclose(energies[:,0], plot_e))[0,0]
    if fig is None:
        fig, ax = plt.subplots()
    ax.plot(eta_diffs, Ir_diffs[:, e_ind], **kwargs)
    ax.set_xlabel('Overpotential \ mV')
    ax.set_ylabel(f'$\Delta\mu$(E={energies[e_ind,0]:.1f} eV)')

    return fig


def add_colorbar(fig, gs, max_eta, min_eta):

    f_ax3=fig.add_subplot(gs[:-1,-1])
    cdict = {'red':   [(0.0, 0.0, 0.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 1.0, 1.0)],

                     'green': [(0.0, 0.0, 0.0),
                               (1.0, 0.0, 0.0)],

                     'blue':  [(0.0, 1.0, 1.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 0.0, 0.0)],}

    cmap_name = 'my_list'
    cm = mpl.colors.LinearSegmentedColormap(cmap_name, cdict, N=256)

    min_eta = int(round(min_eta * 1000, 0))
    max_eta = int(round(max_eta * 1000, 0))

    norm = mpl.colors.Normalize(vmin=min_eta, vmax=max_eta)
    cb1=mpl.colorbar.ColorbarBase(f_ax3, cmap=cm, norm=norm, orientation='vertical')
    cb1.set_ticks([min_eta, 0, max_eta])
    cb1.set_label('$\eta$ / mV',rotation=0, labelpad=-3, y=.75)

    return fig


def group_name_to_eta(name):
    eta_str = name.split('Eta_')[-1].split('_mV')[0].replace('n', '-').replace('_', '.')
    return float(eta_str)


def transpose_list(list_a):
    return np.array(list_a).T


def larch_xafs_normalization(filepath, _larch=None):
    """
    """
    try:
        del group
    except:
        None
    group = io.read_ascii(filepath, labels='col1, col2')
    group.path = filepath
    group.is_frozen = False
    group.datatype = 'xas'
    group.plot_xlabel = 'col1'
    group.plot_ylabel = 'col2/1.0'
    group.xdat = group.data[0, : ]
    group.ydat = group.data[1, : ] / 1.0
    group.yerr = 1.000000
    group.energy = group.xdat
    group.mu = group.ydat
    xafs.sort_xafs(group, overwrite=True, fix_repeats=True)
    filename = filepath.split('\\')[-1].replace(' ', '_').replace('-', 'n')
    group.filename = filename
    group.filenumber = filename.split('_Eta')[0]
    group.eta = filename.split('Eta_')[-1].split(' mV')[0]
    xafs.pre_edge(group, pre1=-70.00, pre2=-25.00, nvict=0.00, nnorm=1.00, norm1=40.00, norm2=115.00)
    group.norm_poly = 1.0*group.norm
    group.dnormde=group.dmude/group.edge_step
    xafs.autobk(group, rbkg=0.85, kweight=2)

    return group


def format_wfm(freq, etas, t_fin=1):
    fig, ax =  plt.subplots(constrained_layout=True, sharex=True, sharey=True)
    amp = np.max(np.abs(etas))

    ax.set_xlim(0, t_fin)
    ax.set_ylim(-1.05 * amp, 1.05 * amp)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('V')
    
    return fig


@gif.frame
def plot_wfm(fig, t, wfm, Etas, n):
    ax = fig.axes[0]
    amp = np.max(np.abs(Etas))
    
    ax.plot(t, wfm, color=(.4, .4, .4))
    color = colorfun(wfm[n], v_max = amp, v_min = -amp)
    ax.plot(t[n], wfm[n],  color=color, marker='o', fillstyle='full')
