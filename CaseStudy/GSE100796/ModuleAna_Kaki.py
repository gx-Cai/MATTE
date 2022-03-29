import pandas as pd
import numpy as np
import MATTE
import dill as pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency,kruskal
import matplotlib.cm as cm
import matplotlib.patches as mpatches
## --- preprocess --- ##

## WGCNA

def WGCNA_data():
    df_label = pd.read_csv('./WGCNAa/WGCNA_labels.csv',index_col=0)
    preservation = pd.read_csv('./WGCNAa/preserve_pvalue.csv',index_col=0)
    label_colors0 = preservation.index.values.tolist()
    label_colors1 = preservation.columns.values.tolist()
    x,y = np.where(preservation.values < 0.05) ## columns~chimp index~human

    for i in range(df_label.shape[0]):
        df_label.iloc[i,0] = label_colors1.index(df_label.iloc[i,0])
        df_label.iloc[i,1] = label_colors0.index(df_label.iloc[i,1])
        df_label.loc[df_label.index[i],'matched'] = (
        df_label.iloc[i,0],df_label.iloc[i,1]) in list(zip(y,x))
    JM = np.zeros(shape=(len(label_colors1),len(label_colors0)))
    for i in range(df_label.shape[0]):
        idx = df_label.iloc[i,0]
        idy = df_label.iloc[i,1]
        JM[idx,idy] += 1
    df_label['matched'] = df_label['matched'].astype(bool)
    return df_label, JM

## MATTE
def matte_data():
    with open('./HumanChimp_v101.pkl','rb') as f:
        R = pickle.load(f)
        df_label = R.res
        JM = R.JM
    return df_label,JM


## --- Module Analysis Functions --- ##
def process_evo_info(res):
    Evo = pd.read_csv('./Data/Human_Chimpanzee_Orthologue_Ka_Ks_Ki.csv',index_col=0).drop_duplicates()
    Evo = Evo[Evo.index.isin(res.index)]
    Evo['Ka/Ks'] = Evo['Ka/Ks'].fillna(0)
    return Evo

def box_samemodule(df_label, Evo,ax=None):
    sns.boxplot(
    x=df_label.loc[Evo.index,'matched'],
    y=Evo['Ka/Ki'],
    fliersize=0,
    palette=[cm.get_cmap('Paired')(i+1) for i in range(2)],
    ax = ax
    )

def bar_matched(ax, K_match):
    mp,mnp = [K_match[K_match.matched]['postive_select'].sum(),sum(~K_match[K_match.matched]['postive_select'])]
    nmp,nmnp = [K_match[~K_match.matched]['postive_select'].sum(),sum(~K_match[~K_match.matched]['postive_select'])]    

    ax.bar(
            0,100*mp/(mp+nmp),
            color = '#3274a1',
        )
    ax.bar(
            0, 100*nmp/(mp+nmp), 
            color='#b3d495', 
            bottom = 100*mp/(mp+nmp),
            )
    ax.bar(
            1,100*mnp/(mnp+nmnp),
            color = '#3274a1',            
        )
    ax.bar(
            1, 100*nmnp/(mnp+nmnp),
            color='#b3d495',
            bottom = 100*mnp/(mnp+nmnp),
            )

def chi2test_matched(K_match):
    ## chi2 test
    chi2,p_value = chi2_contingency(
    [
        [K_match[K_match.matched]['postive_select'].sum(),sum(~K_match[K_match.matched]['postive_select'])],
        [K_match[~K_match.matched]['postive_select'].sum(),sum(~K_match[~K_match.matched]['postive_select'])]
    ]
)[:2]
    # print("If diag/nodiag genes related to ks/ki",chi2,p_value)
    return p_value

def kruskal_matched(K_match):
    ## kruskal test
    k_pvalue = kruskal(
        K_match[K_match.matched]['Ka/Ki'].values,
        K_match[~K_match.matched]['Ka/Ki'].values
    )[1]
    return k_pvalue

## ----- Not used ----- ##
def matte_sigmodule_posratio(R, Evo, K_match):
    sf = R.SampleFeature()
    module_snr = R.ModuleSNR(sf)
    sig_module_rank = pd.Series([i.split('_')[0] for i in module_snr.index])

    for m in sig_module_rank:
        genes = R.module_genes[m]
        genes = genes[genes.isin(Evo.index)]
        print(
        m,
        K_match.loc[genes,'postive_select'].sum()/genes.size,
    )

def fig_heatmap_K(JM, K_match):
    poselect_JM = np.zeros(shape=JM.shape) # number of pos-select genes
    kratio_JM = np.zeros(shape=JM.shape) # sum of ka/ki in a module
    for i in range(K_match.shape[0]):
        locs = K_match.iloc[i,1:3].values
        kratio_JM[locs[0],locs[1]] += K_match.iloc[i,0]
        if not K_match.iloc[i,-1]:
            continue
        poselect_JM[locs[0],locs[1]] += 1

    sns.heatmap(
    poselect_JM/JM,
    cmap='Reds',
    annot=JM,
    fmt = '.0f',
    )
    plt.title('Ratio of Positive Select Genes')
    plt.show()

    sns.heatmap(
    kratio_JM/JM,
    cmap='Reds',
    annot=JM,
    fmt = '.0f',

    )
    plt.title('Average of Ka/Ki')
    plt.show()

## ----- Main ----- ##
if __name__ == '__main__':


    # F1: Box Plot
    sns.set_theme(style='white')
    f,axes = plt.subplots(1,2,figsize=(6,3.5),dpi=300)

    for i,ax in enumerate(axes):
        if i == 0:
            df_label,_ = WGCNA_data()
            name = 'WGCNA'
        elif i == 1:
            df_label,_ = matte_data()
            name = 'MATTE'

        Evo = process_evo_info(df_label)
        K_match = pd.concat([Evo['Ka/Ki'],df_label.loc[Evo.index,:]],axis=1)
        K_match['postive_select'] = K_match['Ka/Ki']>1
        K_match['matched'] = K_match.matched.astype(bool)

        p = kruskal_matched(K_match)
        box_samemodule(df_label, Evo,ax=ax)

        ax.set_title(name,fontsize=12)
        ax.set_ylim(0,1)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.text(
            0.1,0.9,
            f'p-value: {p:.2e}',
            transform=ax.transAxes,fontsize=10)
            
    # add x and y label for complete figure
    plt.text(-0.28,0.5,'Ka/Ki',fontsize=16,rotation=90,transform=axes[0].transAxes)
    plt.text(
        -0.85,-0.2,
        'Genes in Matched Module',
        fontsize=16,
        transform=axes[1].transAxes,
        ha = 'left',
        )
    
    plt.savefig('./Results/Ka_Ki_Kruskal_Boxplot.pdf',bbox_inches='tight')


    # F2: Bar plot
    sns.set_theme(style='white')
    f,axes = plt.subplots(1,2,figsize=(6,4),dpi=300)

    for i,ax in enumerate(axes):
        if i == 0:
            df_label,_ = WGCNA_data()
            name = 'WGCNA'
        elif i == 1:
            df_label,_ = matte_data()
            name = 'MATTE'

        Evo = process_evo_info(df_label)
        K_match = pd.concat([Evo['Ka/Ki'],df_label.loc[Evo.index,:]],axis=1)
        K_match['postive_select'] = K_match['Ka/Ki']>1
        K_match['matched'] = K_match.matched.astype(bool)

        p = chi2test_matched(K_match)
        
        bar_matched(ax, K_match)

        ax.set_title(f"{name} p-value:{p:.2f}",fontsize=12)
        ax.set_xticks([0,1],['>1','<1'])
        if i == 1:
            ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        # ax.text(
        #     0.1,0.9,
        #     f'p-value: {p:.2f}',
        #     transform=ax.transAxes,fontsize=12)

    # add x and y label for complete figure
    plt.text(-0.28,0.4,'Ratio(%)',fontsize=16,rotation=90,transform=axes[0].transAxes)
    plt.text(
        -0.25,-0.18,
        'Ka/Ki',
        fontsize=16,
        transform=axes[1].transAxes
        )
    handles = [
        mpatches.Patch(color='#3274a1',label='Matched'),
        mpatches.Patch(color='#b3d495',label='Unmatched'),
    ]
    plt.legend(
        handles=handles,
        bbox_to_anchor=(1.05,0.65),
        loc='upper left',
        fontsize=12,
        edgecolor='white',
        )
    
    f.savefig('./Results/Ka_Ki_chi2_Barplot.pdf',bbox_inches='tight')