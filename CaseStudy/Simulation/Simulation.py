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
    
