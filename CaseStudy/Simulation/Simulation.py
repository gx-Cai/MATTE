import pandas as pd
import numpy as np

import MATTE
print(MATTE.__version__)

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

import sklearnex
sklearnex.patch_sklearn()

# ----- network methods ------#
def dist_rthres(data:np.array,pheno,thres=0.8):
    data = np.array(data)
    dmat = []
    for i in pheno.unique():
        subdata = data[pheno==i].T
        distance_matrix = pairwise_distances(subdata,metric='correlation')
        distance_matrix[distance_matrix < thres] = 0
        dmat.append(distance_matrix)
    
    return np.subtract(*dmat)

def dist_rsf(data:np.array,pheno=None,thres=0.9):
    data = np.array(data)
    distance_matrix = pairwise_distances(data.T,metric='correlation')
    lr = LinearRegression()
    best_power = 6
    for power in range(4,13):
        y,x = np.histogram((distance_matrix**power).sum(axis=1),bins=100)
        lr.fit(np.log10(x[1:]+1).reshape(-1,1),np.log10(y+1))
        score = lr.score(np.log10(x[1:]+1).reshape(-1,1),np.log10(y+1))
        if score > thres:
            best_power = power

    dmat = []
    for i in pheno.unique():
        subdata = data[pheno==i].T
        distance_matrix = pairwise_distances(subdata,metric='correlation')
        distance_matrix = distance_matrix ** best_power
        dmat.append(distance_matrix)

    return np.subtract(*dmat)

def connect_intra(distmat,cluster = None):
    if cluster is None:
        cluster = AgglomerativeClustering(
            n_clusters=4,affinity='precomputed',linkage='average')
    
    cluster.fit(distmat)
    label = cluster.labels_
    return np.abs(np.array([distmat[i,np.where(label==label[i])[0]].sum()
        for i in range(len(label))
    ]))

def diff_kin(data,pheno,method):
    l1,l2 = pheno.unique()
    data1 = data[pheno==l1].values.T
    data2 = data[pheno==l2].values.T

    aff1 = method(data1)
    aff2 = method(data2)

    k1 = connect_intra(aff1)
    k2 = connect_intra(aff2)
    
    k = np.abs(np.log2(k1/k2))
    k[np.isinf(k)] = 100
    k[np.isnan(k)] = 0
    return pd.Series(k,index=data.columns)
    
def dist_ecf(data,pheno):
    from diff_stat import ECF
    dist_mat = ECF(data.values,pheno.values)
    return dist_mat

def dist_zscore(data,pheno):
    from diff_stat import zscore
    return zscore(data.values,pheno.values)

def dist_diffcoex(data,pheno):
    from diff_stat import DCE
    dist_mat = DCE(data.values,pheno.values)
    return dist_mat

def dist_ent(data,pheno):
    from diff_stat import entropy
    return entropy(data.values,pheno.values)

def diff_k(data,pheno,method):
    return method(data,pheno).sum(axis=1)
# ----- process data -----#

def add_noise(data,type:str,level:int):
    if type == 'loc':
        loc = level
        scale = 0.1
    elif type == 'scale':
        loc = 0
        scale = level
    elif type == 'mix':
        loc = level
        scale = level
    else:
        raise ValueError('type must be loc,scale,mix')
    return pd.DataFrame(
        data= data.values+np.random.normal(loc=loc,scale=scale,size=data.shape),
        columns = data.columns,
        index = data.index
    )


# ----- methods -----#
def matte_rank(data, pheno):
    pipe = MATTE.AlignPipe()
    pipe.funcs = pipe.funcs[4:] ## delete preprocess steps
    MED = MATTE.ModuleEmbedder(pipe)
    MED.gene_rank(data,pheno,verbose=False)
    return MED.gene_ranking

def cross_cluster(data, pheno):
    pipe = MATTE.AlignPipe()
    pipe.funcs = pipe.funcs[4:] ## delete preprocess steps
    R = pipe.calculate(df_exp=data.T,df_pheno = pheno,verbose=False)
    return [f"{i[0]}{i[1]}" for i in R.res.iloc[:,0:2].values.tolist()]

    
# ------  main ----- #
def RunMethods_eva(data,pheno,truelabel):

    def evaluate(func,**kwargs):
        return roc_auc_score(truelabel,func(data,pheno,**kwargs))

    ret = {
        'matte': evaluate(matte_rank),
        'coexp-thres': evaluate(diff_k,method=dist_rthres),
        'coexp-sf': evaluate(diff_k,method=dist_rsf),
        'ecf':evaluate(diff_k,method=dist_ecf),
        'zscore':evaluate(diff_k,method=dist_zscore),
        'diffcoex':evaluate(diff_k,method=dist_diffcoex),
        'ent':evaluate(diff_k,method=dist_ent),
    }
    return ret

def RunMethods_noise(data,pheno,noise_level):
    def evaluate(method,):
        cluster = AgglomerativeClustering(
            n_clusters=8,affinity='precomputed',linkage='average')

        def clustering(d,p):
            aff = method(d,p)
            return cluster.fit_predict(aff)

        basic_label = clustering(data,pheno,)
        label = clustering(add_noise(data,'scale',noise_level),pheno,)
        return adjusted_rand_score(basic_label,label)
    
    def evalueate_matte():
        basic_label = cross_cluster(data,pheno)
        label = cross_cluster(add_noise(data,'scale',noise_level),pheno)
        return adjusted_rand_score(basic_label,label)

    ret = {
        'matte': evalueate_matte(),
        'coexp-thres': evaluate(method=dist_rthres),
        'coexp-sf': evaluate(method=dist_rsf),
        'ecf':evaluate(method=dist_ecf),
        'zscore':evaluate(method=dist_zscore),
        'diffcoex':evaluate(method=dist_diffcoex),
        'ent':evaluate(method=dist_ent),
    }
    return ret

def main_accuracy(n_iters=10):
    from genSimDat import gendat_mix
    from tqdm import trange
    all_results = {}
    for _ in trange(n_iters):
        gen,truelabel,negtive_gen = gendat_mix()
        data = pd.DataFrame(gen) # row samples, col genes
        genes = pd.Series([f'G{i}' for i in range(data.shape[1])])
        samples= pd.Series([f'S{i}' for i in range(data.shape[0])] + \
            [f'NS{i}' for i in range(negtive_gen.shape[0])])
        pheno = pd.Series([1]*data.shape[0] + [0]*negtive_gen.shape[0],index=samples)
        data = pd.concat([data,pd.DataFrame(negtive_gen)],axis=0).reset_index(drop=True)
        data = pd.DataFrame(data.values,index=samples.values,columns=genes)
        truelabel = pd.Series(truelabel,index=genes)

        results = {
            i: RunMethods_eva(add_noise(data,type='scale',level=i),pheno,truelabel)
            for i in np.linspace(0,3,10)
        }
        all_results[f"mix_{_}"] = pd.DataFrame(results)
    return all_results

def main_noise(n_iters=10,subsample = 100):
    from tqdm import trange
    from random import sample
    all_results = {}
    
    data = pd.read_table('A:/Data/Co_Exp_Evolution/GSE100796/GSE100796_TPM_fixed.txt',index_col=0)
    data = pd.concat([data.filter(like='Human'),data.filter(like='Chimpanzee')],axis=1)
    data = data.T
    pheno = pd.Series([i.split('_')[0] for i in data.index],index=data.index)

    for _ in trange(n_iters):
        genes = sample(data.columns.values.tolist(),k=subsample)
        subdata = data.loc[:,genes]

        results = {
            i: RunMethods_noise(subdata,pheno,i)
            for i in np.linspace(0.1,3,10)
        }
        all_results[f"mix_{_}"] = pd.DataFrame(results)
    return all_results

def main_batch(n_iters=10):
    from genSimDat import gendat_mix
    from tqdm import trange
    allres_auc = {}
    for _ in trange(n_iters):
        gen,truelabel,negtive_gen = gendat_mix(n_genes=10)
        data = pd.DataFrame(gen) # row samples, col genes
        genes = pd.Series([f'G{i}' for i in range(data.shape[1])])
        samples= pd.Series([f'S{i}' for i in range(data.shape[0])] + \
            [f'NS{i}' for i in range(negtive_gen.shape[0])])
        pheno = pd.Series([1]*data.shape[0] + [0]*negtive_gen.shape[0],index=samples)
        data = pd.concat([data,pd.DataFrame(negtive_gen)],axis=0).reset_index(drop=True)
        data = pd.DataFrame(data.values,index=samples.values,columns=genes)
        truelabel = pd.Series(truelabel,index=genes)

        results = { 
            i: RunMethods_batch(
                add_noise_biased(data,type='loc',levels=[i]+[0]*4),
                pheno,truelabel)
            for i in np.linspace(0.1,1,10)
        }
        allres_auc[f"mix_{_}"] = pd.DataFrame(results)
    return allres_auc

def main_confuse(n_iters=10,ranker = relative_rank,**kwargs):
    from genSimDat import gendat_mix
    from tqdm import trange

    n_genes = 10
    
    confuse_mat = pd.DataFrame(index=['S','W','N'],columns=['S','W','N'],data=0)
    diff_patten = pd.DataFrame(
        data="N",
        index = [f"G{i}" for i in range(n_genes*8*(10+1))],columns = ['DE','DC'])
    diff_patten.loc[[f"G{i}" for i in range(n_genes)],:] = ["S","S"]
    diff_patten.loc[[f"G{i}" for i in range(n_genes,2*n_genes)],:] = ["S","W"]
    diff_patten.loc[[f"G{i}" for i in range(2*n_genes,3*n_genes)],:] = ["W","S"]
    diff_patten.loc[[f"G{i}" for i in range(3*n_genes,4*n_genes)],:] = ["W","W"]
    diff_patten.loc[[f"G{i}" for i in range(4*n_genes,5*n_genes)],:] = ["S","N"]
    diff_patten.loc[[f"G{i}" for i in range(5*n_genes,6*n_genes)],:] = ["W","N"]
    diff_patten.loc[[f"G{i}" for i in range(6*n_genes,7*n_genes)],:] = ["N","S"]
    diff_patten.loc[[f"G{i}" for i in range(7*n_genes,8*n_genes)],:] = ["N","W"]
    
    for _ in trange(n_iters):
        gen,truelabel,negtive_gen = gendat_mix(n_genes=n_genes)
        data = pd.DataFrame(gen) # row samples, col genes
        genes = pd.Series([f'G{i}' for i in range(data.shape[1])])
        samples= pd.Series([f'S{i}' for i in range(data.shape[0])] + \
            [f'NS{i}' for i in range(negtive_gen.shape[0])])
        pheno = pd.Series([1]*data.shape[0] + [0]*negtive_gen.shape[0],index=samples)
        data = pd.concat([data,pd.DataFrame(negtive_gen)],axis=0).reset_index(drop=True)
        data = pd.DataFrame(data.values,index=samples.values,columns=genes)
        truelabel = pd.Series(truelabel,index=genes)

        rank = pd.Series(ranker(data,pheno,**kwargs),index=data.columns)
        # rank_order = pd.Series(index=rank.sort_values(ascending=False).index, data=list(range(rank.shape[0])))

        # mcc_rank = pd.Series(
        #     [mcc(truelabel,(rank_order<=i).astype(int)[truelabel.index]) for i in range(len(rank))])
        # pred = (rank >= rank[rank_order.index[mcc_rank.argmax()]]).astype(int)
        thres = rank[diff_patten[(diff_patten==['N','N']).all(axis=1)].index].quantile(0.95)
        pred = (rank >= thres).astype(int)
        
        for depat in ['S','W','N']:
            for  dcpat in ['S','W','N']:
                pat_genes = diff_patten.index[(diff_patten == [depat,dcpat]).all(axis=1)]
                confuse_mat.loc[depat,dcpat] += pred[pat_genes].mean()
    confuse_mat /= n_iters
    return confuse_mat

def main_test(n_iters=10,type='mix'):
    from genSimDat import gendat_mix,gendat_dc,gendat_de
    from tqdm import trange
    allres_auc = []
    allres_line = []

    if type=='mix':
        gendat_func = gendat_mix
    elif type=='de':
        gendat_func = gendat_de
    elif type=='dc':
        gendat_func = gendat_dc

    for _ in trange(n_iters):
        gen,truelabel,negtive_gen = gendat_func()
        data = pd.DataFrame(gen) # row samples, col genes
        genes = pd.Series([f'G{i}' for i in range(data.shape[1])])
        samples= pd.Series([f'S{i}' for i in range(data.shape[0])] + \
            [f'NS{i}' for i in range(negtive_gen.shape[0])])
        pheno = pd.Series([1]*data.shape[0] + [0]*negtive_gen.shape[0],index=samples)
        data = pd.concat([data,pd.DataFrame(negtive_gen)],axis=0).reset_index(drop=True)
        data = pd.DataFrame(data.values,index=samples.values,columns=genes)
        truelabel = pd.Series(truelabel,index=genes)

        result_auc,result_line = RunMethods_eva(data,pheno,truelabel)
        allres_auc.append(result_auc)
        allres_line.append(result_line)

    return pd.concat(allres_auc),pd.concat(allres_line)


# ------ visualization ----- #
def dict2df_flatten(d):
    df = pd.concat(d.values(),axis=1)
    ret = pd.DataFrame(columns=['method','metric','noise'])
    n = 0
    for i in range(df.shape[1]):
        noise = df.columns[i]
        for j in range(df.shape[0]):
            method = df.index[j]
            ret.loc[n] = [method,df.iloc[j,i],noise]
            n += 1
    return ret

def vis_noise(results,ylabel):
    sns.set_theme(style="whitegrid")
    f = sns.catplot(data = results,
                kind='box',
                x='noise',hue='method',y='metric',
                aspect=1.7
                )
    sns.despine(left=True)
    _ = plt.xticks(ticks = range(10),labels=[f"{i:.2f}" for i in results.noise.unique()])
    plt.xlabel('Noise')
    plt.ylabel(ylabel)
    return f

def acc_line(all_results,ylabel):
    
    sns.set_theme(style="white")
    ## hide vertical gride line
    f = sns.relplot(
        x = 'noise', y = 'metric',
        hue='method',
        data=all_results,
        # ci=False,
        kind='line',
        aspect=1.7,
    )
    plt.ylabel(ylabel)
    plt.xlabel('Noise')
    return f

def confuse_mat_heat_seq(confuse_mat,method_name,ax,row=False,col=False):
    confuse_mat.loc['N','N'] = 1- confuse_mat.loc['N','N']
    sns.heatmap(
        confuse_mat,
        vmin=0,vmax=1,
        annot=True,cbar=False,ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    if col:
        ax.set_xticks(ticks=[0.5+i for i in range(3)],labels=['Strong','Weak','None'])
    if row:
        ax.set_yticks(ticks=[0.5+i for i in range(3)],labels=['Strong','Weak','None'])
    ax.set_title(f'{method_name}')
    return ax


if __name__ == '__main__':

    all_results = main_accuracy(n_iters=100)
    all_results = dict2df_flatten(all_results)

    fa1 = acc_line(all_results,ylabel='AUC')
    fa2 = vis_noise(all_results,ylabel='AUC')
    fa1.savefig('./Results/ACCTest_line.pdf')
    fa2.savefig('./Results/ACCTest_box.pdf')
    all_results.to_csv('./Results/ACCTest.csv')

    noise_result = main_noise(n_iters=100)
    noise_result = dict2df_flatten(noise_result)

    fb1 = acc_line(noise_result,ylabel='ARI')
    fb2 = vis_noise(noise_result,ylabel='ARI')
    fb1.savefig('./Results/NoiseTest_line.pdf')
    fb2.savefig('./Results/NoiseTest_box.pdf')
    noise_result.to_csv('./Results/NoiseTest.csv')
    
    confuse_mat_rdm = main_confuse(n_iters=100,ranker=relative_rank,type='RDM')
    confuse_mat_rde = main_confuse(n_iters=100,ranker=relative_rank,type='RDE')
    confuse_mat_rdc = main_confuse(n_iters=100,ranker=relative_rank,type='RDC')
    confuse_mat_ent = main_confuse(n_iters=100,ranker=diff_k,method=dist_ent)
    confuse_mat_ecf = main_confuse(n_iters=100,ranker=diff_k,method=dist_ecf)
    confuse_mat_z = main_confuse(n_iters=100,ranker=diff_k,method=dist_zscore)
    confuse_mat_diffcoex = main_confuse(n_iters=100,ranker=diff_k,method=dist_diffcoex)
    confuse_mat_f = main_confuse(n_iters=100,ranker=lambda x,y:f_classif(x,y)[0])
    confuse_mat_mi = main_confuse(n_iters=100,ranker=mutual_info_classif)
    confuse_mat_snr = main_confuse(n_iters=100,ranker=snr)

    f,ax = plt.subplots(2,5,figsize=(12,6),dpi=300)
    confuse_mat_heat_seq(confuse_mat_rdm,'RDM',ax[0,0],row=True)
    confuse_mat_heat_seq(confuse_mat_rde,'RDE',ax[0,1])
    confuse_mat_heat_seq(confuse_mat_rdc,'RDC',ax[0,2])
    confuse_mat_heat_seq(confuse_mat_ent,'ENT',ax[0,3],)
    confuse_mat_heat_seq(confuse_mat_mi,'MI',ax[0,4],)
    confuse_mat_heat_seq(confuse_mat_ecf,'ECF',ax[1,0],row=True,col=True)
    confuse_mat_heat_seq(confuse_mat_z,'ZSCORE',ax[1,1],col=True)
    confuse_mat_heat_seq(confuse_mat_diffcoex,'DIFF-COEX',ax[1,2],col=True)
    confuse_mat_heat_seq(confuse_mat_f,'F',ax[1,3],col=True)
    confuse_mat_heat_seq(confuse_mat_snr,'SNR',ax[1,4],col=True)

    f.text(0.5, 0.06, 'Differenctial Co-Expression', ha='center', va='center',fontweight='bold')
    f.text(0.08, 0.5, 'Differenctial Expression', ha='center', va='center', rotation='vertical',fontweight='bold')
    f.text(0.5, 0.96, 'Accuracy Matrix in simulations(FP=0.05)', ha='center',fontsize=18)
    f.savefig('./Results/ConfuseMat.pdf',bbox_inches='tight')
