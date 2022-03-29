import pandas as pd
import numpy as np
import MATTE
print(MATTE.__version__)
import dill as pickle
import os

import gtfparse
gencode = gtfparse.read_gtf('A:/Data/Annotation/gencode.v38.annotation.gtf')
gencode = gencode[['gene_name','gene_id']].drop_duplicates()
gencode.set_index('gene_id',inplace=True)
gencode.index = [i.split('.')[0] for i in gencode.index]
idmap = gencode.to_dict()['gene_name']
del gencode


## WGCNA
def WGCNA_data():
    df_label = pd.read_csv('./WGCNAa/WGCNA_labels.csv',index_col=0)
    preservation = pd.read_csv('./WGCNAa/preserve_pvalue.csv',index_col=0)
    kiw = pd.read_csv('./WGCNAa/DC.csv',index_col=0).iloc[:,0:2]
    kiw = kiw.fillna(0)
    dkiw = np.abs(np.log(kiw.iloc[:,0] / (1e-5+kiw.iloc[:,1])))
    
    label_colors = preservation.index.values.tolist()
    x,y = np.where(preservation.values < 0.05) ## columns~chimp index~human
    JM = np.zeros(shape=(len(label_colors),len(label_colors)))
    module_genes = {}
    
    for i in range(df_label.shape[0]):
        df_label.iloc[i,0] = label_colors.index(df_label.iloc[i,0])
        df_label.iloc[i,1] = label_colors.index(df_label.iloc[i,1])
        df_label.loc[df_label.index[i],'matched'] = (
        df_label.iloc[i,0],df_label.iloc[i,1]) in list(zip(y,x))

        idx = df_label.iloc[i,0]
        idy = df_label.iloc[i,1]
        
        module_key = f"C{idx}H{idy}"
        module_genes[module_key] = module_genes.get(module_key,[]) + [df_label.index[i]] 
        
        JM[idx,idy] += 1
        
    df_label['matched'] = df_label['matched'].astype(bool)
    df_label['score'] = dkiw

    return df_label, module_genes

## MATTE
with open('./HumanChimp_v101.pkl','rb') as f:
    R = pickle.load(f)

def gene_rank_matte(R):
    gene_rank = pd.Series(index=R.df_exp.index,data=0)
    sf = R.SampleFeature()
    sf.columns = [i.split('_')[0] for i in sf.columns]
    modulesnr = R.ModuleSNR(sf)
    samples = sf.index
    for module,genes in R.module_genes.items():
        for g in genes:
            gene_rank[g] = modulesnr[module] * np.abs(np.corrcoef(
                x = R.df_exp.loc[g,samples].values,
                y = sf.loc[samples,module].values
            )[0,1])
    gene_rank = gene_rank.sort_values(ascending=False)
    return gene_rank


## ----------

import gseapy as gp
from tqdm import tqdm
print(gp.__version__)
defined_genesets = gp.parser.gsea_gmt_parser('A:/Data/Annotation/Brain_gmt/merged.gmt')

def enrichr(module_genes:dict):
    enriched = {}
    module_genes = {k:pd.Series(v) for k,v in module_genes.items()}
    for module,genes in tqdm(module_genes.items()):
        transfer_genes = [idmap[i] for i in genes[genes.isin(pd.Series(idmap.keys()))]]
        enr_res = gp.enrichr(
            gene_list=transfer_genes, 
            gene_sets=defined_genesets,
            outdir=None,verbose=False)
        res = enr_res.res2d
        if res is not None:
            res['module'] = module
            enriched[module] = res
    enriched_all = pd.concat(list(enriched.values()))
    enriched_all = enriched_all.set_index('Gene_set')

    return enriched_all

def prerank(gene_score):
    gene_score = gene_score[gene_score.index.isin(idmap.keys())]
    gene_score.index = [idmap[i] for i in gene_score.index]
    gene_score = pd.DataFrame(gene_score).reset_index()

    pre_res = gp.prerank(
        gene_score,min_size=5,
        gene_sets=defined_genesets,
        outdir=None,verbose=False)
    res = pre_res.res2d
    res[res['fdr'] >1]['fdr'] = 0.0 ## BUGS when all items result to 0.0
    return res


df_label,module_genes = WGCNA_data()
enrichr_w = enrichr(
    {
        'matched':df_label[df_label['matched']].index,
        'unmatched':df_label[~df_label['matched']].index,
    }
)
# gsea_w1 = prerank(df_label[~df_label['matched']]['score'])
# gsea_w2 = prerank(df_label[df_label['matched']]['score'])


#matte_gene_rank = gene_rank_matte(R)
enrichr_m = enrichr(
    {
        'matched':R.res[R.res['matched']].index,
        'unmatched':R.res[~R.res['matched']].index,
    }
)
# gsea_m1 = prerank(matte_gene_rank[R.res[~R.res['matched']].index])
# gsea_m2 = prerank(matte_gene_rank[R.res[R.res['matched']].index])


from MATTE.analysis import Fig_Fuction
category_map = {
    'matched':'Conserved',
    'unmatched':'Divergent',
}
df_enriched = enrichr_m[enrichr_m['Adjusted P-value']<=0.05]
df_enriched = df_enriched[['Term','Adjusted P-value','Overlap','module']]
df_enriched['Overlap'] = [eval(i) for i in df_enriched['Overlap']]
df_enriched.columns = ['Term','Adjusted P-value','gene_ratio','Category']
df_enriched['Category'] = [category_map[i] for i in df_enriched['Category']]
df_enriched = df_enriched.groupby(by='Category').apply(lambda x: x.iloc[0:10,:]).reset_index(drop=True).set_index('Term')
f = Fig_Fuction(df_enriched,color_columns='Adjusted P-value',dpi=300,figsize=(8,5))
f.savefig('./Results/FuncEnrich_matte.pdf',bbox_inches='tight')


from MATTE.analysis import Fig_Fuction
df_enriched = enrichr_w[enrichr_w['Adjusted P-value']<=0.05]
df_enriched = df_enriched[['Term','Adjusted P-value','Overlap','module']]
df_enriched['Overlap'] = [eval(i) for i in df_enriched['Overlap']]
df_enriched.columns = ['Term','Adjusted P-value','gene_ratio','Category']
# df_enriched.set_index('Term',inplace=True)
df_enriched = df_enriched.groupby(by='Category').apply(lambda x: x.iloc[0:10,:]).reset_index(drop=True).set_index('Term')
f = Fig_Fuction(df_enriched,color_columns='Adjusted P-value',dpi=300,figsize=(15,5))





