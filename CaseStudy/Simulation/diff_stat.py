# In this file, serveral functions are defined to calculate the degree of differential expression or coexpression

# Python Implement is converted from following's R code.
# Ma H, Schadt EE, Kaplan LM, et al. COSINE: COndition-SpecIfic sub-NEtwork identification using a global optimization method. Bioinformatics 2011; 27:1290–1298
# Bhuva DD, Cursons J, Smyth GK, et al. Differential co-expression-based detection of conditional relationships in transcriptional data: comparative analysis and application to breast cancer. Genome Biology 2019; 20:236

# Methods raise by:
# ECF(expected conditional f): Lai Y, Wu B, Chen L, et al. A statistical method for identifying differential gene-gene co-expression patterns. Bioinformatics 2004; 20:3146–3155
# zscore: Zhang J, Ji Y, Zhang L. Extracting three-way gene interactions from microarray data. Bioinformatics. 2007;23:2903–9.
# DCE(DiffCoExp): Tesson BM, Breitling R, Jansen RC. DiffCoEx: a simple and sensitive method to find differentially coexpressed gene modules. BMC Bioinformatics 2010; 11:497
# entropy: Ho YY, Cope L, Dettling M, Parmigiani G. Statistical methods for identifying differentially expressed gene combinations. Methods Mol Biol. 2007;408:171–91.

import numpy as np
from numpy import sqrt
import numba
import warnings
warnings.filterwarnings("ignore")

def ECF(data,pheno):

    n_samples,n_genes = data.shape
    assert len(pheno) == n_samples

    l1,l2 = np.unique(pheno)

    n1 = (pheno==l1).sum()
    n2 = (pheno==l2).sum()
    mu1 = data[pheno==l1].mean(axis=0)
    mu2 = data[pheno==l2].mean(axis=0)
    var1 = np.var(data[pheno==l1],axis=0,ddof=1)
    var2 = np.var(data[pheno==l2],axis=0,ddof=1)
    pho1_ = np.corrcoef(
        data[pheno==l1],rowvar=False)
    pho2_ = np.corrcoef(
        data[pheno==l2],rowvar=False)

    @numba.jit(nopython=True)
    def ecf(i,j):
        mux1 = mu1[i]
        mux2 = mu2[i]
        muy1 = mu1[j]
        muy2 = mu2[j]

        varx1 = var1[i]
        varx2 = var2[i]
        vary1 = var1[j]
        vary2 = var2[j]

        pho1 = pho1_[i,j]
        pho2 = pho2_[i,j]

        x = ((muy1 - muy2) - (mux1 - mux2) * pho2 * sqrt(vary2)/sqrt(varx2))**2
        x = x + varx1 * (pho1 * sqrt(vary1)/sqrt(varx1) - pho2 * 
            sqrt(vary2)/sqrt(varx2))**2
        x = x * n1/(n1 + n2)
        y = ((muy1 - muy2) - (mux1 - mux2) * pho1 * sqrt(vary1)/sqrt(varx1))**2
        y = y + varx2 * (pho1 * sqrt(vary1)/sqrt(varx1) - pho2 * 
            sqrt(vary2)/sqrt(varx2))**2
        y = y * n2/(n1 + n2)
        z = (x + y)/((n1 + n2) * (vary1 * (1 - pho1 * pho1)/n2 + 
            vary2 * (1 - pho2 * pho2)/n1))
        z = 0 if z is None else z
        return z/2

    res = np.zeros((n_genes,n_genes))
    for i in range(n_genes):
        for j in range(i):
            res[i,j] = ecf(i,j)

    res = res + res.T
    return res

def zscore(data,pheno,):
    l1,l2 = np.unique(pheno)
    data1 = data[pheno==l1]## data1 is a n_samples * n_genes matrix
    data2 = data[pheno==l2]

    r1 = np.corrcoef(data1,rowvar=False)
    r2 = np.corrcoef(data2,rowvar=False)

    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    z = (z1-z2)/sqrt(1/(sum(pheno==l1)-3)) + 1/(sum(pheno==l2)-3)
    z[np.isnan(z)] = 0
    z[np.isinf(z)] = 0
    return z

def DCE(data,pheno,beta=6):
    l1,l2 = np.unique(pheno)
    data1 = data[pheno==l1]## data1 is a n_samples * n_genes matrix
    data2 = data[pheno==l2]

    r1 = np.corrcoef(data1,rowvar=False)
    r2 = np.corrcoef(data2,rowvar=False)

    D = sqrt( 0.5* np.abs(np.sign(r1)* r1**2 - np.sign(r2)* r2**2) )
    D = D**beta

    T = D @ D + D.shape[1] * D

    mins1 = np.array([[i]*D.shape[1] for i in D.sum(axis=1)])
    mins0 = np.array([[i]*D.shape[0] for i in D.sum(axis=0)]).T

    mins = np.minimum(mins1,mins0)

    T = 1 - T/(mins+1-D)
    T[np.diag_indices_from(T)] = 1

    return 1-T

def entropy(data,pheno):
    l1,l2 = np.unique(pheno)
    data1 = data[pheno==l1]## data1 is a n_samples * n_genes matrix
    data2 = data[pheno==l2]

    r1 = np.corrcoef(data1,rowvar=False)
    r2 = np.corrcoef(data2,rowvar=False)
    rall = np.corrcoef(data,rowvar=False)

    I1 = -((1 + abs(r1))/2 * np.log((1 + abs(r1))/2) + (1 - abs(r1))/2 * 
        np.log((1 - abs(r1))/2))
    
    I2 = -((1 + abs(r2))/2 * np.log((1 + abs(r2))/2) + (1 - abs(r2))/2 *
        np.log((1 - abs(r2))/2))
    
    Iall = -((1 + abs(rall))/2 * np.log((1 + abs(rall))/2) + (1 - abs(rall))/2 *
        np.log((1 - abs(rall))/2))

    ent = (I1 + I2)/2 - Iall
    ent[np.diag_indices_from(ent)] = 0

    return ent
