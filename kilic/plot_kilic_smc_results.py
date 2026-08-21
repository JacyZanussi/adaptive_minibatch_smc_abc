"""
Plot posterior distributions and HDPR regions for
adaptive minibatch (FVC) and constant minibatch SMC-ABC
applied to the Kilic model.
"""

# =========================
# Imports
# =========================

import pickle as pkl
import numpy as np
#import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.colors as mcolors


FONT_SIZE_AXES_LABEL = 16
FONT_SIZE_AXES_TITLE = 16
FONT_SIZE_XTICK = 12
FONT_SIZE_YTICK = 12
FONT_SIZE_LEGEND = 10
FONT_SIZE_LEGEND_TITLE = 10

#plt.rcParams
mpl.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['CMU Serif'],
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'CMU Serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    'axes.labelsize': FONT_SIZE_AXES_LABEL,
    'axes.titlesize': FONT_SIZE_AXES_TITLE,
    'xtick.labelsize': FONT_SIZE_XTICK,
    'ytick.labelsize': FONT_SIZE_YTICK,
    'legend.fontsize': FONT_SIZE_LEGEND,
    'legend.title_fontsize': FONT_SIZE_LEGEND_TITLE,
    
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'lines.linewidth': 2.2,
    'errorbar.capsize': 2.5,
    'svg.fonttype' : 'none'
})



# =========================
# Paths
# =========================

FVC_PATH = "kilic/ecoli_slowgrowth_fvc.pkl"
CONST_PATH = "kilic/ecoli_slowgrowth.pkl"
OUT_PATH = "kilic/posterior_hdpr_ecoli.pdf"

# =========================
# Load results
# =========================

with open(FVC_PATH, "rb") as f:
    df_fvc = pkl.load(f)
#    tracked_fvc, posterior_fvc, weights_fvc = pkl.load(f)

with open(CONST_PATH, "rb") as f:
    df_const = pkl.load(f)
#    tracked_const, posterior_const, weights_const = pkl.load(f)



#Auxiliary function. It's often a nice flare and good contrast to mix in a little of another color.
def mix_colors(color1, color2, w=0.5):
    """
    Return a convex combination of two colors.
    w = weight for color1 (between 0 and 1)
    """
    c1 = np.array(mcolors.to_rgb(color1))
    c2 = np.array(mcolors.to_rgb(color2))
    mixed = (1 - w) * c1 + w * c2
    return mcolors.to_hex(mixed)


var_names = ["$K_{1,2}$","$K_{2,1}$","$B_1$","$B_2$"]


## get estimate and HDPR information
#wang_ests = data_Wang['ground']['rates']
wang_ests = np.array([0.00533,0.03,0.0,0.166,0.00533]) # From Wang

#NOTE: kilic_ests are manually read off a published figure (not machine-readable/extracted data);
# treat as an approximate visual reference only, not a precise ground truth.
kilic_ests = np.array([0.00255,0.012,0.00015,0.1666])

last_gen_const = np.max([x for x in df_const[0]])
const_ests = df_const[0][last_gen_const]['hdpr'][0]
const_hdpr = df_const[0][last_gen_const]['hdpr'][1]
posterior_const = df_const[1]
weights_const = df_const[2]
#data_const = pd.DataFrame(posterior_const,columns = var_names) 

last_gen_fvc = np.max([x for x in df_fvc[0]])
fvc_ests = df_fvc[0][last_gen_fvc]['hdpr'][0]
fvc_hdpr = df_fvc[0][last_gen_fvc]['hdpr'][1]
posterior_fvc = df_fvc[1]
weights_fvc = df_fvc[2]
#data_fvc = pd.DataFrame(posterior_fvc,columns = var_names)

fvc_color = 'darkmagenta'
const_color = 'crimson'
wang_color = 'blue'
kilic_color = 'black'
hist1_color = mix_colors('steelblue','white',0.1)
hist2_color = mix_colors('goldenrod','white',0.1)
hdpr1_color = mix_colors('orange','white',0.8)
hdpr2_color = mix_colors('blue','white',0.8)


last_gen_const = np.max([x for x in df_const[0]])
const_ests = df_const[0][last_gen_const]['hdpr'][0]
const_hdpr = df_const[0][last_gen_const]['hdpr'][1]
posterior_const = df_const[1]
weights_const = df_const[2]

last_gen_fvc = np.max([x for x in df_fvc[0]])
fvc_ests = df_fvc[0][last_gen_fvc]['hdpr'][0]
fvc_hdpr = df_fvc[0][last_gen_fvc]['hdpr'][1]
posterior_fvc = df_fvc[1]
weights_fvc = df_fvc[2]

# Compute combined bins for each variable
bin_edges_list = []
for i, column in enumerate(var_names):
    combined_data = np.concatenate([
        posterior_fvc[:, i],
        posterior_const[:, i]
    ])
    # Here we choose 20 bins as an example
    bins = np.histogram_bin_edges(combined_data, bins=25)
    bin_edges_list.append(bins)


fig, axes = plt.subplots(4, 4, figsize=(9, 9), sharex='col')
#axes = axes.flatten()

diag_xlims = [None] * 4

marker_size = 15
marker_size_ests = 30
marker_type = '.'
marker_type_ests = '*'


for i in range(4):
    for j in range(4):
        if i == j:
            axes[i,i].hist(
                posterior_fvc[:, i], 
                bins=bin_edges_list[i], 
                weights=weights_fvc, 
                density=True, 
                alpha=0.7, 
                color=hist1_color,
                edgecolor = None,
                linewidth=0.4,
                zorder=1
            )
            # Constant histogram
            axes[i,i].hist(
                posterior_const[:, i],
                bins=bin_edges_list[i],
                weights=weights_const,
                density=True,
                alpha=0.7,
                color=hist2_color,
                edgecolor = None,
                linewidth=0.4,
                zorder=0
            )
            ylims = axes[i,i].get_ylim()
            linealpha = 0.8
            axes[i,i].plot([wang_ests[i],wang_ests[i]],ylims, color = wang_color,zorder = 10,alpha = linealpha, ls='solid')
            axes[i,i].plot([fvc_ests[i],fvc_ests[i]],ylims, color = fvc_color,zorder = 10,alpha = linealpha, ls='dotted')
            axes[i,i].plot([const_ests[i],const_ests[i]], ylims, color = const_color,zorder = 10,alpha = linealpha, ls='dashed')
            axes[i,i].plot([kilic_ests[i],kilic_ests[i]], ylims, color = kilic_color,zorder = 10,alpha = linealpha, ls='dashdot')
            
            diag_xlims[i] = axes[i,i].get_xlim()
            bnds = const_hdpr[i]
            if bnds.ndim == 1:
                axes[i,i].axvspan(bnds[0],bnds[1],color = hdpr1_color,alpha = 1,zorder = -1)
            else:
                for bnd in bnds:
                    axes[i,i].axvspan(bnd[0],bnd[1],color = hdpr1_color,alpha = 1,zorder = -1)
            bnds = fvc_hdpr[i]
            if bnds.ndim == 1:
                axes[i,i].axvspan(bnds[0],bnds[1],ylims[0],ylims[1],color = hdpr2_color,alpha = 1,zorder = -1)
            else:
                for bnd in bnds:
                    axes[i,i].axvspan(bnd[0],bnd[1],ylims[0],ylims[1],color = hdpr2_color,alpha = 1,zorder = -1)
            axes[i,i].set_ylim(ylims)
            # if (i == 0) or (i == 2):
            #     axes[i,i].set_ylabel(rf"Posterior Density", fontsize=14)
        elif i < j:
            axes[i,j].scatter(posterior_fvc[:,j],posterior_fvc[:,i],color = hist1_color,alpha = 0.7,marker = marker_type,s=marker_size,rasterized = True,edgecolors='none')
            axes[i,j].scatter(wang_ests[j],wang_ests[i],color = wang_color,marker = 'd',s=marker_size_ests,rasterized = True,edgecolors='none')
            axes[i,j].scatter(fvc_ests[j],fvc_ests[i],color = fvc_color,marker = 'P',s=marker_size_ests,rasterized = True,edgecolors='none')
            axes[i,j].scatter(const_ests[j],const_ests[i],color = const_color,marker = 'X',s=marker_size_ests,rasterized = True,edgecolors='none')
            axes[i,j].scatter(kilic_ests[j],kilic_ests[i],color = kilic_color,marker = '*',s=marker_size_ests,rasterized = True,edgecolors='none')

            bndsi = fvc_hdpr[i]
            bndsj = fvc_hdpr[j]

            # Ensure 2D shape: (num_intervals, 2)
            if bndsi.ndim == 1:
                bndsi = bndsi[None, :]
            if bndsj.ndim == 1:
                bndsj = bndsj[None, :]

            for yi in bndsi:
                for xj in bndsj:
                    x0, x1 = xj
                    y0, y1 = yi
                    rect = Rectangle(
                        (float(x0), float(y0)),
                        float(x1 - x0),
                        float(y1 - y0),
                        facecolor=hdpr2_color,
                        alpha=1,
                        zorder=-1
                    )
                    axes[i,j].add_patch(rect)

            pass
        else:
            axes[i,j].scatter(posterior_const[:,j],posterior_const[:,i],color = hist2_color,alpha = 0.7,marker = marker_type,s=marker_size,rasterized = True,edgecolors='none')
            axes[i,j].scatter(wang_ests[j],wang_ests[i],color = wang_color,marker = 'd',s=marker_size_ests,rasterized = True,edgecolors='none')
            axes[i,j].scatter(fvc_ests[j],fvc_ests[i],color = fvc_color,marker = 'P',s=marker_size_ests,rasterized = True,edgecolors='none')
            axes[i,j].scatter(const_ests[j],const_ests[i],color = const_color,marker = 'X',s=marker_size_ests,rasterized = True,edgecolors='none')
            axes[i,j].scatter(kilic_ests[j],kilic_ests[i],color = kilic_color,marker = '*',s=marker_size_ests,rasterized = True,edgecolors='none')
            ylims = axes[i,j].get_ylim()
            xlims = axes[i,j].get_xlim()

            bndsi = const_hdpr[i]
            bndsj = const_hdpr[j]

            # Ensure 2D shape: (num_intervals, 2)
            if bndsi.ndim == 1:
                bndsi = bndsi[None, :]
            if bndsj.ndim == 1:
                bndsj = bndsj[None, :]

            for yi in bndsi:
                for xj in bndsj:
                    x0, x1 = xj
                    y0, y1 = yi
                    rect = Rectangle(
                        (float(x0), float(y0)),
                        float(x1 - x0),
                        float(y1 - y0),
                        facecolor=hdpr1_color,
                        alpha=1,
                        zorder=-1
                    )
                    axes[i,j].add_patch(rect)

            axes[i,j].set_xlim(diag_xlims[j])
            axes[i,j].set_ylim(diag_xlims[i])
        if i == 3:
            axes[i,j].set_xlabel(var_names[j], fontsize=FONT_SIZE_AXES_LABEL)
        if j == 0:
            axes[i,j].set_ylabel(var_names[i], fontsize=FONT_SIZE_AXES_LABEL)

for i in range(4): #row
    for j in range(4): #column
        if i == j:
            continue
        axes[i,j].set_xlim(diag_xlims[j])
        axes[i,j].set_ylim(diag_xlims[i])

for i in range(4):
    for j in range(4):
        ax = axes[i,j]
        ax.tick_params(
            labelbottom=(i == 3),
            labelleft=(j == 0),
            labeltop=False,
            labelright=False,
            bottom=(i == 3),
            left=(j == 0),
            top=False,
            right=False
        )

fig.subplots_adjust(left=0.04, right=0.99, bottom=0.04, top=0.99, wspace=0.0, hspace=0.0)

# Legend handles
wang_line = Line2D([], [], color=wang_color, ls='solid', lw=2, label="Wang et al")
wang_m = Line2D([], [], color=wang_color, marker='d', ls ='', lw=2, label="Wang et al")
fvc_line = Line2D([], [], color=fvc_color, ls='dotted',lw=2, label="FVC")
fvc_m = Line2D([], [], color=fvc_color, marker='P', ls ='', lw=2, label="FVC")
const_line = Line2D([], [], color=const_color, ls='dashed', lw=2, label="Constant mb")
const_m = Line2D([], [], color=const_color, marker='X', ls ='', lw=2, label="Constant mb")
kilic_line = Line2D([], [], color=kilic_color, ls='dashdot', lw=2, label="Kilic et al")
kilic_m = Line2D([], [], color=kilic_color, marker='*', ls ='', lw=2, label="Kilic et al")

hdpr_patch_fvc = Patch(facecolor=hdpr2_color,alpha=1,label='FVC hdpr')
patch_fvc = Patch(facecolor=hist1_color,alpha=0.7,label='FVC')
hdpr_patch_const = Patch(facecolor=hdpr1_color,alpha=1,label='Constant hdpr')
patch_const = Patch(facecolor=hist2_color,alpha=0.7,label='Constant')


from matplotlib.legend_handler import HandlerTuple

axes[0,0].legend(
    handles=[(fvc_line,fvc_m), (const_line,const_m), (wang_line,wang_m), (kilic_line,kilic_m)],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    labels=['FVC','Constant','Wang et al','Kilic et al'],
    loc='upper left',
    frameon=True,
    framealpha=0.9,
    facecolor='white',
    fontsize=FONT_SIZE_LEGEND,
    handlelength=6,
    ncol = 2,
    columnspacing = 1.5,
    handletextpad = 0.5,
    labelspacing = 0.6,
    bbox_to_anchor=(0.35, 1.3)
)
axes[0,3].legend(
    handles=[patch_fvc, hdpr_patch_fvc, patch_const, hdpr_patch_const],
    loc='upper right',
    frameon=True,
    framealpha=0.9,
    facecolor='white',
    fontsize=FONT_SIZE_LEGEND,
    ncol = 2,
    columnspacing = 1.5,
    handletextpad = 0.5,
    labelspacing = 0.6,
    bbox_to_anchor=(0.65, 1.3)
)


print(plt.rcParams['font.family'], plt.rcParams['font.serif'])

plt.savefig(OUT_PATH,
            format = 'pdf',
            dpi = 600,
            bbox_inches = 'tight',
            transparent = True)
