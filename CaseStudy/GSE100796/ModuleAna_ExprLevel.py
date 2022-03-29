import pandas as pd
import numpy as np
import MATTE
import dill as pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open('./HumanChimp_v101.pkl','rb') as f:
    R = pickle.load(f)

sf = R.SampleFeature()
modulesnr = R.ModuleSNR(sf)

sns.set_theme(style='white',context='talk')
f,axes = plt.subplots(1,2,figsize=(8,15),dpi=300)
plt.subplots_adjust(
    left=None, bottom=None, right=None, top=None, 
    wspace=0.3, hspace=None)

barax = axes[0]
heatax = axes[1]

df_bar = modulesnr.reset_index()
df_bar.columns = ['Module','SNR']
sns.barplot(
    x = 'SNR',
    y = 'Module',
    data=df_bar,
    ax=barax
)
barax.invert_xaxis()
#barax.invert_yaxis()
_ = barax.set_yticks([])

sns.heatmap(
    sf.T.loc[modulesnr.index,:],
    cmap='Blues',
    ax=heatax,
    cbar=False,
)
_ = heatax.set_yticks(
    ticks=[i+0.5 for i in np.arange(len(modulesnr))],
    labels=[i.split('_')[0] for i in modulesnr.index])
_ = heatax.set_xticks([])
ny = len(modulesnr)
nx = R.pheno.value_counts()[1]
an1 = heatax.annotate(
    '', (0, ny+0.5), (nx, ny+0.5),
    xycoords='data',
    horizontalalignment='left', 
    verticalalignment='top',
    annotation_clip=False,
    arrowprops=dict(arrowstyle='-', color="#3274a1",lw=4,
    )
)
an2 = heatax.annotate(
    '', (nx, ny+0.5), (sf.shape[0], ny+0.5),
    xycoords='data',
    horizontalalignment='left', 
    verticalalignment='top',
    annotation_clip=False,
    arrowprops=dict(arrowstyle='-', color="#3c6e3b",lw=4,
    )
)
an1_text = heatax.annotate(
    'Human', (0, ny+1),
    xycoords='data',
    horizontalalignment='left',
    verticalalignment='top',
    fontsize=16,
    annotation_clip=False,)
an2_text = heatax.annotate(
    'Chimpanzee', (nx, ny+1),
    xycoords='data',
    horizontalalignment='left',
    verticalalignment='top',
    fontsize=16,
    annotation_clip=False,)
an_text = heatax.annotate(
    "Samples", (nx, ny+3),
    xycoords='data',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=18,
    annotation_clip=False,)
heatax.axvline(nx,color='white',linestyle='--',lw=2)

f.savefig('./Results/ModuleConf.pdf',bbox_inches='tight')

sns.set_theme(style='white',context='notebook')
f2 = R.Vis_Jmat()
plt.title("Number of genes")
f2.savefig('./Results/Jmat.pdf',bbox_inches='tight')

sns.set_theme(style='white',context='talk')
f3 = plt.figure(dpi=300,figsize=(8,6))
sns.heatmap(
    R.df_exp.loc[R.module_genes['M3.5'],:],
    cbar=False,
    cmap='Blues',
)
plt.xticks([])
plt.yticks([])
plt.xlabel('Human      Chimpanzee')
plt.ylabel("Genes in M3.5")
f3.savefig('./Results/M3.5_GeneExp.pdf',bbox_inches='tight')