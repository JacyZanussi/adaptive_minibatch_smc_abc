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


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.colors as mcolors

# ── LaTeX + font setup ────────────────────────────────────────────────────
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'lines.linewidth': 1.4,
})

# =========================
# Paths
# =========================

FVC_PATH = "kilic/ecoli_slowgrowth_fvc.pkl"
CONST_PATH = "kilic/ecoli_slowgrowth.pkl"
OUT_PATH = "kilic/posterior_hdpr_ecoli.svg"

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

fvc_color = mix_colors('green','magenta',0.75)
const_color = mix_colors('crimson','gold',0.15)
wang_color = mix_colors('darkgreen','cyan',0.3)
kilic_color = 'black'
hist1_color = "#dbe4ff"
hist1_color = mix_colors('blue','white',0.35)
hist2_color = mix_colors(mix_colors('gold','black',0.1),'orange',0.5)


fvc_color = 'magenta'
const_color = 'crimson'
wang_color = 'blue'
kilic_color = 'black'
hist1_color = mix_colors('steelblue','white',0.1)
#hist1_color = 'darkcyan'
#hist1_color = "#738678"
hist2_color = mix_colors('goldenrod','white',0.1)
#hist2_color = 'darkgoldenrod'
hdpr1_color = mix_colors('orange','white',0.8)
hdpr2_color = mix_colors('blue','white',0.8)

## get estimate and HDPR information
#wang_ests = data_Wang['ground']['rates']
wang_ests = np.array([0.00533,0.03,0.0,0.166,0.00533]) # From Wang

#NOTE: Eyeballed.... Because They didn't get these... I literally took a piece of paper to my computer screen and zoomed in.
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


fig, axes = plt.subplots(4, 4, figsize=(14, 14))
#axes = axes.flatten()

marker_size = 15
marker_size_ests = 20
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
            axes[i,i].plot([wang_ests[i],wang_ests[i]],ylims,linewidth = 2.2, color = wang_color,zorder = 10,alpha = linealpha)
            axes[i,i].plot([fvc_ests[i],fvc_ests[i]],ylims,linewidth = 2.2, color = fvc_color,zorder = 10,alpha = linealpha)
            axes[i,i].plot([const_ests[i],const_ests[i]],ylims,linewidth = 2.2, color = const_color,zorder = 10,alpha = linealpha)
            axes[i,i].plot([kilic_ests[i],kilic_ests[i]],ylims,linewidth = 2.2, color = kilic_color,zorder = 10,alpha = linealpha)
            
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
        elif i > j:
            axes[i,j].scatter(posterior_fvc[:,j],posterior_fvc[:,i],color = hist1_color,alpha = 0.7,marker = marker_type,s=marker_size)
            axes[i,j].scatter(wang_ests[j],wang_ests[i],color = wang_color,marker = marker_type_ests,s=marker_size_ests)
            axes[i,j].scatter(fvc_ests[j],fvc_ests[i],color = fvc_color,marker = marker_type_ests,s=marker_size_ests)
            axes[i,j].scatter(const_ests[j],const_ests[i],color = const_color,marker = marker_type_ests,s=marker_size_ests)
            axes[i,j].scatter(kilic_ests[j],kilic_ests[i],color = kilic_color,marker = marker_type_ests,s=marker_size_ests)
            ylims = axes[i,j].get_ylim()
            xlims = axes[i,j].get_xlim()

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




            #bndsi = fvc_hdpr[i]
            #bndsj = fvc_hdpr[j]
            #x0 = bndsj[0]
            #x1 = bndsj[1]
            #y0 = bndsi[0]
            #y1 = bndsi[1]
            #rect = Rectangle(
            #    (x0,y0),
            #    x1 - x0,
            #    y1 - y0,
            #    facecolor=hdpr2_color,
            #    alpha=1,
            #    zorder = -1
            #)
            #axes[i,j].add_patch(rect)
            axes[i,j].set_ylim(ylims)
            axes[i,j].set_xlim(xlims)
        else:
            axes[i,j].scatter(posterior_const[:,j],posterior_const[:,i],color = hist2_color,alpha = 0.7,marker = marker_type,s=marker_size)
            axes[i,j].scatter(wang_ests[j],wang_ests[i],color = wang_color,marker = marker_type_ests,s=marker_size_ests)
            axes[i,j].scatter(fvc_ests[j],fvc_ests[i],color = fvc_color,marker = marker_type_ests,s=marker_size_ests)
            axes[i,j].scatter(const_ests[j],const_ests[i],color = const_color,marker = marker_type_ests,s=marker_size_ests)
            axes[i,j].scatter(kilic_ests[j],kilic_ests[i],color = kilic_color,marker = marker_type_ests,s=marker_size_ests)
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



            #bndsi = const_hdpr[i]
            #bndsj = const_hdpr[j]
            #x0 = bndsj[0]
            #x1 = bndsj[1]
            #y0 = bndsi[0]
            #y1 = bndsi[1]
            #rect = Rectangle(
            #    (x0,y0),
            #    x1 - x0,
            #    y1 - y0,
            #    facecolor=hdpr1_color,
            #    alpha=1,
            #    zorder = -1
            #)
            #axes[i,j].add_patch(rect)
            axes[i,j].set_ylim(ylims)
            axes[i,j].set_xlim(xlims)
        if i == 3:
            axes[i,j].set_xlabel(var_names[j], fontsize=11)
        if j == 0:
            axes[i,j].set_ylabel(var_names[i], fontsize=11)


# Legend handles
fvc_line = Line2D([], [], color=fvc_color, lw=2, label="FVC")
const_line = Line2D([], [], color=const_color, lw=2, label="Constant mb")
wang_line = Line2D([], [], color=wang_color, lw=2, label="Wang et al")
kilic_line = Line2D([], [], color=kilic_color, lw=2, label="Kilic et al")

hdpr_patch_fvc = Patch(facecolor=hdpr2_color,alpha=1,label='FVC hdpr')
patch_fvc = Patch(facecolor=hist1_color,alpha=0.7,label='FVC')
hdpr_patch_const = Patch(facecolor=hdpr1_color,alpha=1,label='Constant hdpr')
patch_const = Patch(facecolor=hist2_color,alpha=0.7,label='Constant')

axes[1,1].legend(
    handles=[fvc_line, const_line, wang_line, kilic_line],
    loc='upper right',
    frameon=True,
    framealpha=0.9,
    facecolor='white',
    fontsize=8
)
axes[3,3].legend(
    handles=[patch_fvc, hdpr_patch_fvc, patch_const, hdpr_patch_const],
    loc='upper right',
    frameon=True,
    framealpha=0.9,
    facecolor='white',
    fontsize=8
)
plt.tight_layout()

plt.savefig(OUT_PATH,
            format = 'svg',
            dpi = 600,
            bbox_inches = 'tight',
            transparent = True)
