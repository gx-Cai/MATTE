from .utils import printv, kw_decorator
import pandas as pd
import numpy as np
from sklearn.preprocessing import KernelCenterer
from sklearn.feature_selection import f_classif
from joblib.parallel import Parallel, delayed
from sklearn.preprocessing import normalize,StandardScaler


__all__ = ["RPKM2TPM", "log2transform", "exp_filter", "Kernel_transform", "inputs_check",'normalization']

## --- Preprocess --- ##

@kw_decorator('Checking_status')
def inputs_check(df_exp, df_pheno):
    """Checking inputs if they are correct. decorated by :func:`MATTE.utils.kw_decorator`

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param df_pheno: Phenotype dataframe.
    :type df_pheno: pandas.Series
    :raises ValueError: If the input dataframe is not correct.
    :return: checking status
    :rtype: dict {`Checking_status`:'OK'}
    """    
    genes = df_exp.index
    samples = df_exp.columns
    phenoes = df_pheno.unique()

    # check gene names and pheno names should not contains '@'
    for i in genes:
        if '@' in i:
            raise ValueError(f"Gene name {i} contains '@'")
    for i in phenoes:
        if '@' in str(i):
            raise ValueError(f"Pheno name {i} contains '@'")
    
    # check samples
    if len(samples) != len(set(samples)):
        raise ValueError("Samples should be unique.")
    if set(list(df_pheno.index)) != set(samples):
        raise ValueError("Samples in df_pheno and df_exp are not consistent.")

    # check genes
    if len(genes) != len(set(genes)):
        raise ValueError("Genes should be unique.")

    assert (df_exp >=0).all().all(), "Expression contains negative values."
    return 'OK'

@kw_decorator(kw='df_exp')
def RPKM2TPM(df_exp):
    """Convert RPKM to TPM. decorated by :func:`MATTE.utils.kw_decorator`

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :return: TPM dataframe.
    :rtype: dict
    """
    df = df_exp.values
    oneM = 1e6
    df_sum = df.sum(axis=0)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            df[i, j] = (df[i, j]/df_sum[j]) * oneM
    return pd.DataFrame(data=df, columns=df_exp.columns, index=df_exp.index)


@kw_decorator(kw="df_exp")
def log2transform(df_exp):
    """Log2 transform the expression dataframe. decorated by :func:`MATTE.utils.kw_decorator`

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :return: Log2 transformed dataframe.
    :rtype: dict
    """    
    if type(df_exp) == pd.DataFrame:
        return pd.DataFrame(
            data=np.log2(df_exp.values+1),
            index=df_exp.index,
            columns=df_exp.columns
        )
    else:
        return np.log2(df_exp+1)

@kw_decorator(kw="df_exp")
def normalization(df_exp,norm='l1'):
    """Normalize the expression dataframe. decorated by :func:`MATTE.utils.kw_decorator`

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param norm: normalization type,should be one of `l1`,`l2` and `standard`, defaults to 'l1'
    :type norm: str, optional
    :return: Normalized dataframe.
    :rtype: dict
    """    
    cols = df_exp.columns
    rows = df_exp.index      
    if norm in ['l1','l2']:
        return pd.DataFrame(normalize(df_exp,norm=norm),index=rows,columns=cols)
    elif norm == 'standard':
        SD = StandardScaler()
        return pd.DataFrame(SD.fit_transform(df_exp),index=rows,columns=cols)


@kw_decorator(kw='df_exp')
def exp_filter(df_exp,df_pheno,gene_filter=None,filter_args:dict={}):
    """Filter the genes by rule. decorated by :func:`MATTE.utils.kw_decorator`

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param df_pheno: Phenotype dataframe.
    :type df_pheno: pandas.Series
    :param gene_filter: gene filter rule. `None` will delete some genes with extreme low expression. `f` will filter genes by Anova f value(p=0.05 by default). `function` is also allowed.defaults to None
    :type gene_filter: str or function, optional
    :param filter_args: other args sent to gene filter function, defaults to {}
    :type filter_args: dict, optional
    :return:  filtered dataframe.
    :rtype: dict
    """    
    if gene_filter is None:
        df_exp = df_exp[(df_exp >= 1).any(axis=1)]
    elif gene_filter == 'f':
        thres = filter_args.get('thres',0.05)
        f,p = f_classif(df_exp.T.values,df_pheno.values)
        df_exp = df_exp.iloc[np.where(p<thres)[0],:]
    else:
        df_exp = df_exp[gene_filter(df_exp,df_pheno,**filter_args)]
    return df_exp

## --- Variable Generate  --- ##

def generate_df_exp_mixed(df_exp, df_pheno):
    """Generate expression dataframe with mixed phenotypes. That's the first step of cross clustering. In this step, genes from differnet phenotypes are treated as different genes.

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param df_pheno: Phenotype dataframe.
    :type df_pheno: pandas.Series
    :return: Expression dataframe with mixed phenotypes, whose index is mixed genes(using `'@'`as a seperate to label genes from phenotypes).
    :rtype: pandas.DataFrame
    """    
    phenoes = df_pheno.unique()
    n_select_samples = df_pheno.value_counts().min()
    df_set = []
    for p in phenoes:
        df_sub = df_exp.loc[:, df_pheno[df_pheno ==
                                        p].index].iloc[:, 0:n_select_samples]
        df_sub.index = [f'{i}@{p}' for i in df_sub.index]
        df_sub.columns = range(n_select_samples)
        df_set.append(df_sub)
    return pd.concat(df_set)


## --- Kernel Transformation --- ##

def outer_subtract(x, absolute=False):
    """Outer subtract the mean of each gene.

    :param x: Expression data
    :type x: numpy.array
    :param absolute: whether after substract to take absolute, defaults to False
    :type absolute: bool, optional
    :return: cross substract mean of each gene.
    :rtype: numpy.array
    """    
    if absolute:
        return np.abs(x[:, np.newaxis] - x[np.newaxis, :])
    else:
        return x[:, np.newaxis] - x[np.newaxis, :]


def variant_rbf_kernel(x, gamma: float = 1.0):
    """Variant RBF kernel.

    :param x: Expression data
    :type x: numpy.array
    :param gamma: rbf kernel parameters, defaults to 1.0
    :type gamma: float, optional
    :return: RBF kernel matrix.
    :rtype: numpy.array
    """    
    assert x.shape[0] == x.shape[1]
    return np.exp(-gamma * x ** 2)


def mean_Kernel_transform(df_exp, df_pheno, genes_mixed, absolute=True,):
    """Mean Kernel transform.

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param df_pheno: Phenotype dataframe.
    :type df_pheno: pandas.Series
    :param genes_mixed: mixed genes.
    :type genes_mixed: array like
    :param absolute: whether to take absolute, defaults to True
    :type absolute: bool, optional
    :return: Mean Kernel transform matrix.
    :rtype: numpy.array
    """    
    mean_by_pheno = df_exp.groupby(df_pheno, axis=1).mean()
    mixed_genes_mean = mean_by_pheno.stack().reset_index()
    mixed_genes_mean.index = [
        f"{mixed_genes_mean.iloc[i,0]}@{mixed_genes_mean.iloc[i,1]}" for i in range(mixed_genes_mean.shape[0])]
    mixed_genes_mean = mixed_genes_mean.iloc[:, -1].loc[genes_mixed]
    return outer_subtract(mixed_genes_mean.values, absolute)


def meanrbf_Kernel_transform(df_exp, df_pheno, genes_mixed, gamma, absolute=True,):
    """Mean RBF Kernel transform. Is a variant of rbf kernel. 

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param df_pheno: Phenotype dataframe.
    :type df_pheno: pandas.Series
    :param genes_mixed: mixed genes.
    :type genes_mixed: array like
    :param gamma: rbf kernel parameters, defaults to 1.0
    :type gamma: float, optional
    :param absolute: whether to take absolute in outer substract, defaults to True
    :type absolute: bool, optional
    :return: Mean RBF Kernel transform matrix.
    :rtype: numpy.array
    """    
    mean_by_pheno = df_exp.groupby(df_pheno, axis=1).mean()
    mixed_genes_mean = mean_by_pheno.stack().reset_index()
    mixed_genes_mean.index = [
        f"{mixed_genes_mean.iloc[i,0]}@{mixed_genes_mean.iloc[i,1]}" for i in range(mixed_genes_mean.shape[0])]
    mixed_genes_mean = mixed_genes_mean.iloc[:, -1].loc[genes_mixed]
    return variant_rbf_kernel(outer_subtract(mixed_genes_mean.values, absolute),gamma=gamma)

def localrbf_Kernel_transform(df_exp, df_pheno, genes_mixed,K=7, absolute=True):
    """Local RBF Kernel transform. each genes' gamma is set to be its' distance to K-th nearest neighbor. 

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param df_pheno: Phenotype dataframe.
    :type df_pheno: pandas.Series
    :param genes_mixed: mixed genes.
    :type genes_mixed: array like
    :param K: nearest neighbor, defaults to 7
    :type K: int, optional
    :param absolute: whether to take absolute or not, defaults to True
    :type absolute: bool, optional
    :return: Mean RBF Kernel transform matrix.
    :rtype: numpy.array
    """    
    mean_by_pheno = df_exp.groupby(df_pheno, axis=1).mean()
    mixed_genes_mean = mean_by_pheno.stack().reset_index()
    mixed_genes_mean.index = [
        f"{mixed_genes_mean.iloc[i,0]}@{mixed_genes_mean.iloc[i,1]}" for i in range(mixed_genes_mean.shape[0])]
    mixed_genes_mean = mixed_genes_mean.iloc[:, -1].loc[genes_mixed]
    mean_mat:np.array = outer_subtract(mixed_genes_mean.values,absolute)

    ## calculation K-th nearest neighbor's distance
    sigmas = pd.Series(index = genes_mixed,dtype=float)
    for i,g in enumerate(genes_mixed):
        ind = mean_mat[i].argsort()[K]
        sigmas[g] = mean_mat[i,ind]
    
    for i in range(mean_mat.shape[0]):
        for j in range(i+1,mean_mat.shape[1]):
            gamma = -1/(sigmas[genes_mixed[i]] * sigmas[genes_mixed[j]])
            mean_mat[i,j] = np.exp(gamma * mean_mat[i,j]**2)
            mean_mat[j,i] = mean_mat[i,j]
    return mean_mat

# Used for any function between two vector
def cdist_generic(dataset1, dataset2, dist_fun, n_jobs=-1, verbose=0,
                  compute_diagonal=True, *args, **kwargs):
    """Compute the distance matrix from two datasets.

    :param dataset1: First dataset.
    :type dataset1: numpy.array
    :param dataset2: Second dataset.
    :type dataset2: numpy.array
    :param dist_fun: Distance function.
    :type dist_fun: function
    :param n_jobs: number of cpus used, defaults to -1
    :type n_jobs: int, optional
    :param verbose: defaults to 0
    :type verbose: int, optional
    :param compute_diagonal: defaults to True
    :type compute_diagonal: bool, optional
    :return: using dist_fun to compute the distance matrix.
    :rtype: numpy.array
    """
    if dataset2 is None:
        matrix = np.zeros((len(dataset1), len(dataset1)))
        indices = np.triu_indices(len(dataset1),
                                  k=0 if compute_diagonal else 1,
                                  m=len(dataset1))
        matrix[indices] = Parallel(n_jobs=n_jobs,
                                   verbose=verbose)(
            delayed(dist_fun)(
                dataset1[i], dataset1[j],
                *args, **kwargs
            )
            for i in range(len(dataset1))
            for j in range(i if compute_diagonal else i + 1,
                           len(dataset1))
        )
        indices = np.tril_indices(len(dataset1), k=-1, m=len(dataset1))
        matrix[indices] = matrix.T[indices]
        return matrix
    else:
        matrix = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(dist_fun)(
                dataset1[i], dataset2[j],
                *args, **kwargs
            )
            for i in range(len(dataset1)) for j in range(len(dataset2))
        )
        return np.array(matrix).reshape((len(dataset1), -1))


## --- Kernel Matrix Process --- ##

def Double_Centering(x: np.array):
    """Double Centering to Kernel Matrix

    :param x: Kernel matrix.
    :type x: np.array
    :return: double centered matrix
    :rtype: numpy.array
    """    
    return x - np.median(x,axis=0) - np.median(x,axis=1) + np.median(x)

## --- Main of Kernel Transformation --- ##

@kw_decorator(kw=['before_cluster_df', 'mixed_genes'])
def Kernel_transform(
    df_exp, df_pheno, kernel_type,
    centering_kernel=True,
    outer_subtract_absolute=True,
    double_centering=True,
    verbose=True,
    **kwargs
):
    """Kernel Transformation. Important preprocess to cross clustering. In this step, genes from different phenotypes are regarded as different genes. and the distance between them is computed. this function is decorated by :func:`MATTE.utils.kw_decorator`.

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param df_pheno: Phenotype dataframe.
    :type df_pheno: pandas.Series
    :param kernel_type: Kernel type, 'rbf' or 'localrbf' or 'mean'. functions are also allowed.
    :type kernel_type: str or function
    :param centering_kernel: whether to centering kernel, defaults to True
    :type centering_kernel: bool, optional
    :param outer_subtract_absolute: in outer subtract, use absolute or not, defaults to True
    :type outer_subtract_absolute: bool, optional
    :param double_centering: whether double centering kernel matrix or not, defaults to True
    :type double_centering: bool, optional
    :param verbose: defaults to True
    :type verbose: bool, optional
    :return: kernel matrix and mixed genes
    :rtype: dict
    """
    if verbose:
        printv(f"Calculating the kernel matrix using {kernel_type}",verbose=verbose)
        # if kernel_type not in ['mean','meanrbf']:
        #     printv(f'kernel type: {kernel_type} are not recommended for long running time and low score',verbose=verbose)

    genes = df_exp.index
    samples = df_exp.columns
    phenoes = df_pheno.unique()
    genes_mixed = pd.Series(
        [f"{i}@{j}" for j in phenoes for i in genes]
    )

    if kernel_type == 'meanrbf':
        gamma = kwargs.get('gamma', 2 / len(samples))
        if type(gamma) == list:
            assert len(gamma) == len(phenoes), "gamma should be a list of length equal to pheno if it is a list"
        printv(f'Setting gamma to {gamma}',verbose=verbose)
        Mat = meanrbf_Kernel_transform(
            df_exp, df_pheno, genes_mixed, gamma, outer_subtract_absolute)

    elif kernel_type == 'mean':
        Mat = mean_Kernel_transform(
            df_exp, df_pheno, genes_mixed, outer_subtract_absolute, )
    
    elif kernel_type == "localrbf":
        K = kwargs.get('K',7)
        Mat =  localrbf_Kernel_transform(
            df_exp, df_pheno, genes_mixed,K, outer_subtract_absolute)

    elif type(kernel_type) == function:
        df_exp_mixed = generate_df_exp_mixed(
            df_exp, df_pheno).loc[genes_mixed, :]
        Mat = cdist_generic(df_exp_mixed.values, None, kernel_type)
    
    else:
        raise ValueError("The kernel type is not in the list.")
    
    if double_centering:
        Mat = Double_Centering(Mat)
    if centering_kernel:
        Mat = KernelCenterer().fit_transform(Mat)
    
    return Mat, genes_mixed

