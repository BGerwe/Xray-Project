import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from larch import io, xafs


def colorfun(V,Max=1):
#     print(V, Max)
    if V>0:
        R=V/Max
        G=0
        B=0
    else:
        R=0
        G=0
        B=-V/Max
    return (R,G,B)

def plotXANES(energies, Irs, Etas, start_eta=0, stop_eta=0, marker='.', inset_dict={},
             startE=7705, stopE=7730, size=(9, 6)):
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
    Ir_abs_min = np.round(np.min(Irs[np.argwhere(energies>=startE)[0][0], :]), 2)

    # Finding max and min of eta (or voltage) for setting color bar limits
    max_eta = int(round(np.max(Etas) * 1000, 0))
    min_eta = int(round(np.min(Etas) * 1000, 0))

    ## Plotting full XANES spectra
    fig = plt.figure(constrained_layout=False, figsize=size)
    gs = fig.add_gridspec(7, 25)
    f_ax1 = fig.add_subplot(gs[:, :-1])

    for n in range(start_eta, Irs.shape[-1] - stop_eta):
        f_ax1.plot(energies, Irs[:, n], color=colorfun(Etas[n], np.max(np.abs(Etas))),
                 linestyle='-',linewidth=.3, marker='', markersize=3,
                   label=f'{Etas[n]} mV')

    f_ax1.set(xlim=[startE,stopE], ylim=[.97 * Ir_abs_min, 1.05 * Ir_abs_max])
    f_ax1.set_ylabel(r'normalized $\mu$(E)  / a.u.', **title_font)
    f_ax1.set_xlabel(r'E  /  eV', **title_font)

    ## Plotting XANES inset
    f_ax2=fig.add_subplot(gs[0:4,3:13])    
    for n in range(start_eta, Irs.shape[-1] - stop_eta):
        f_ax2.plot(energies, Irs[:, n], color=colorfun(Etas[n], np.max(np.abs(Etas))),
                 linestyle='-',linewidth=1, marker=marker, markersize=3,
                   label=f'{Etas[n]} mV')

    ## Choosing plot limits for inset
    xind1=np.argwhere(energies>=inset_dict['start_energy'])[0]
    xind2=np.argwhere(energies<=inset_dict['stop_energy'])[-1]

    x1=round(energies[xind1][0])
    x2=round(energies[xind2][0])

    y1=round(Irs[xind1,0][0]*.95,2)
    y2=round(Irs[xind2,0][0]*1.05,2)

    f_ax2.set(xlim=[x1+x_lat+x1_adj, x2+x_lat+x2_adj], 
          xticks=np.arange(x1+x1_adj, x2+x2_adj, x_interval),
          ylim=[y1+y_vert+y1_adj, y2+y_vert+y2_adj], 
          yticks=np.arange(y1+y1_adj, y2+y2_adj,y_interval))

    f_ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    f_ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    f_ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func))


    # Plotting colorbar
    f_ax3=fig.add_subplot(gs[:5,-1])
    cdict = {'red':   [(0.0, 0.0, 0.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 1.0, 1.0)],

                     'green': [(0.0, 0.0, 0.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 0.0, 0.0)],

                     'blue':  [(0.0, 1.0, 1.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 0.0, 0.0)],}

    cmap_name = 'my_list'
    cm = mpl.colors.LinearSegmentedColormap(cmap_name, cdict, N=100)
    norm = mpl.colors.Normalize(vmin=min_eta, vmax=max_eta)
    cb1=mpl.colorbar.ColorbarBase(f_ax3, cmap=cm, norm=norm, orientation='vertical')
    cb1.set_ticks([min_eta, 0, max_eta])
    cb1.set_label('mV',rotation=0, labelpad=-3, verticalalignment='center')
    
    return fig


def plot_diffXANES(energies, Irs, etas, start_eta=0, larch_step=100, startE=7700, stopE=7735, size=(9,6)):
    fig = plt.figure(constrained_layout=False, figsize=size)
    gs = fig.add_gridspec(7, 25)
    ax1 = fig.add_subplot(gs[:, :-1])

    max_eta = int(round(np.max(etas) * 1000, 0))
    min_eta = int(round(np.min(etas) * 1000, 0))

    Ir_diffs, eta_diffs = [], []
    for n in range(start_eta, int(Irs.shape[-1]/larch_step)):
        try:
            Ir_diff = Irs[:, n*larch_step] - Irs[:,np.argmin(np.abs(etas))]
            Ir_diffs.append(Ir_diff)
            eta_diff = etas[n*larch_step+start_eta]
            eta_diffs.append(eta_diff)
#             print(eta_diff, Ir_diff[80:90])
            ax1.plot(energies, Ir_diff, color=colorfun(etas[n*larch_step+start_eta], np.max(np.abs(etas))))

        except:
            pass
    ax1.set_xlim(startE, stopE)
    Ir_diffs = np.array(Ir_diffs)
    eta_diffs = np.array(eta_diffs)

    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal'}
    ax1.set_ylabel(r'$\Delta$ $\mu_{norm}$  / a.u.', **title_font)
    ax1.set_xlabel(r'Energy  /  eV', **title_font)  
    # Plotting colorbar
    f_ax3=fig.add_subplot(gs[:5,-1])
    cdict = {'red':   [(0.0, 0.0, 0.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 1.0, 1.0)],

                     'green': [(0.0, 0.0, 0.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 0.0, 0.0)],

                     'blue':  [(0.0, 1.0, 1.0),
                               (0.5, 0.0, 0.0),
                               (1.0, 0.0, 0.0)],}

    cmap_name = 'my_list'
    cm = mpl.colors.LinearSegmentedColormap(cmap_name, cdict, N=100)
    norm = mpl.colors.Normalize(vmin=min_eta, vmax=max_eta)
    cb1=mpl.colorbar.ColorbarBase(f_ax3, cmap=cm, norm=norm, orientation='vertical')
    cb1.set_ticks([min_eta, 0, max_eta])
    cb1.set_label('mV',rotation=0, labelpad=-3, verticalalignment='center')

    return fig


def plot_lissajou(energies, Irs, etas, plot_e=7719.8, start_eta=0, larch_step=100, fig=None, **kwargs):
    Ir_diffs, eta_diffs = [], []
    for n in range(start_eta, int(Irs.shape[-1]/larch_step)):
        try:
            Ir_diff = Irs[:, n*larch_step] - Irs[:,np.argmin(np.abs(etas))]
            Ir_diffs.append(Ir_diff)
            eta_diff = etas[n*larch_step+start_eta]
            eta_diffs.append(eta_diff)
        except:
            pass
    
    Ir_diffs = np.array(Ir_diffs)
    eta_diffs = np.array(eta_diffs)

    e_ind = np.argwhere(np.isclose(energies[:,0], plot_e))[0,0]
    if fig is None:
        fig, ax = plt.subplots()
    ax.plot(eta_diffs, Ir_diffs[:, e_ind], **kwargs)
    ax.set_xlabel('Overpotential \ mV')
    ax.set_ylabel(f'$\mu$(E={energies[e_ind,0]:.1f} eV)')
    
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