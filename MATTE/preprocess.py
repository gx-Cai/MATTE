import itertools
from .utils import printv, kw_decorator
import pandas as pd
import numpy as np
from sklearn.preprocessing import KernelCenterer
from sklearn.feature_selection import f_classif
from joblib.parallel import Parallel, delayed
from sklearn.preprocessing import normalize, StandardScaler
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

__all__ = ["RPKM2TPM", "log2transform", "exp_filter",
           "relative_diffdist", "inputs_check", 'normalization','relative_diff']

## --- Preprocess --- ##


@kw_decorator('mixed_genes')
def inputs_check(df_exp, df_pheno):
    """Checking inputs if they are correct. decorated by :func:`MATTE.utils.kw_decorator`

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param df_pheno: Phenotype dataframe.
    :type df_pheno: pandas.Series
    :raises ValueError: If the input dataframe is not correct.
    :return: checking status
    :rtype: dict {`mixed_genes`:'OK'}
    """
    genes = df_exp.index
    samples = df_exp.columns
    phenoes = df_pheno.unique()
    if len(phenoes) >2:
        raise NotImplementedError('Phenotypes over two is not implemented yet.')
    # check samples
    if len(samples) != len(set(samples)):
        raise ValueError("Samples should be unique.")
    if set(list(df_pheno.index)) != set(samples):
        raise ValueError("Samples in df_pheno and df_exp are not consistent.")

    # check genes
    if len(genes) != len(set(genes)):
        raise ValueError("Genes should be unique.")

    assert (df_exp >= 0).all().all(), "Expression contains negative values."

    return  pd.Series([f"{i}@{j}" for j in phenoes for i in genes])



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
    for i, j in itertools.product(range(df.shape[0]), range(df.shape[1])):
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
def normalization(df_exp, norm='l1'):
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
    if norm in ['l1', 'l2']:
        return pd.DataFrame(normalize(df_exp, norm=norm), index=rows, columns=cols)
    elif norm == 'standard':
        SD = StandardScaler()
        return pd.DataFrame(SD.fit_transform(df_exp), index=rows, columns=cols)


@kw_decorator(kw='df_exp')
def exp_filter(df_exp, df_pheno, gene_filter=None, filter_args: dict = {}):
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
        thres = filter_args.get('thres', 0.05)
        f, p = f_classif(df_exp.T.values, df_pheno.values)
        df_exp = df_exp.iloc[np.where(p < thres)[0], :]
    else:
        df_exp = df_exp[gene_filter(df_exp, df_pheno, **filter_args)]
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

def outer_subtract(x, absolute=True):
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

def RDE_transform(data,pheno,absolute):
    """Transform the data by RDE(Relative Differentila Expression)

    :param data: inputs data with rows as samples and columns as genes.
    :type data: pd.DataFrame
    :param pheno: phenotype data corresponding to expression data.
    :type pheno: pd.Series
    :param absolute: whether after substract to take absolute, defaults to False
    :type absolute: bool
    :return: transformed data.
    :rtype: pd.DataFrame
    """
    data1 = data[pheno==pheno.unique()[0]]
    data2 = data[pheno==pheno.unique()[1]]

    mean_vect = np.array([data1.mean(axis=0),data2.mean(axis=0)]).reshape(-1)

    return outer_subtract(mean_vect,absolute)

def decompdist(kmat,n_components=16):
    """calculate the distance(after decomposition)

    :param kmat: kernel matrix 
    :type kmat: numpy.array
    :param n_components: number of components used in decomposition, defaults to 16
    :type n_components: int, optional
    :return: distance matrix
    :rtype: numpy.array
    """    """"""
    pca = PCA(n_components=n_components)
    tkmat = pca.fit_transform(kmat) 
    tkmat1 = tkmat[0:kmat.shape[0]//2,:]
    tkmat2 = tkmat[kmat.shape[0]//2:,:]
    dist_mat = cdist(
        tkmat1,tkmat2,
        w=pca.explained_variance_ratio_)
    return np.abs(dist_mat)

def RDC_transform(data,pheno,centering_kernel,double_centering):
    """Transform the data by RDC(Relative Distance Correlation)

    :param data: inputs data with rows as samples and columns as genes.
    :type data: pd.DataFrame
    :param pheno: phenotype data corresponding to expression data.
    :type pheno: pd.Series
    :param centering_kernel: Whether to center the kernel matrix 
    :type centering_kernel: bool
    :param double_centering: Whether to double center the kernel matrix 
    :type double_centering: bool
    :return: transformed data.
    :rtype: numpy.array
    """    """"""
    f_corr_mat = lambda data: KernelMat_transform(
        1-np.abs(1-pairwise_distances(data.T,metric='correlation')),
        centering_kernel,double_centering)
    data1 = data[pheno==pheno.unique()[0]]
    data2 = data[pheno==pheno.unique()[1]]

    kmat1 = f_corr_mat(data1.values)
    kmat2 = f_corr_mat(data2.values)

    kmat = np.concatenate([kmat1,kmat2])
    return kmat


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

def double_center(x: np.array):
    """Double Centering to Kernel Matrix

    :param x: Kernel matrix.
    :type x: np.array
    :return: double centered matrix
    :rtype: numpy.array
    """
    return x - np.median(x, axis=0) - np.median(x, axis=1) + np.median(x)


def KernelMat_transform(Mat,centering_kernel, double_centering):
    """Transform the kernel matrix

    :param Mat: Kernel matrix.
    :type Mat: np.array
    :param centering_kernel: Whether to center the kernel matrix
    :type centering_kernel: bool
    :param double_centering: Whether to double center the kernel matrix
    :type double_centering: bool
    :return: transformed kernel matrix
    :rtype: np.array
    """    """"""
    if double_centering:
        Mat = double_center(Mat)
    if centering_kernel:
        Mat = KernelCenterer().fit_transform(Mat)
    return Mat

## --- Main of Kernel Transformation --- ##


@kw_decorator(kw='dist_mat')
def relative_diffdist(
    df_exp, df_pheno, kernel_type,
    centering_kernel=True,
    double_centering=False,
    outer_subtract_absolute=True,
    n_components = 16,
    verbose=True,
):
    """Calculate the relative difference distance.
    In this step, genes from different phenotypes are regarded as different genes. and the distance between them is computed. this function is decorated by :func:`MATTE.utils.kw_decorator`.

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param df_pheno: Phenotype dataframe.
    :type df_pheno: pandas.Series
    :param kernel_type: Kernel type, one of the following: ['RDE','RDC','RDM']; functions are also allowed.
    :type kernel_type: str or function
    :param centering_kernel: whether to centering kernel, defaults to False
    :type centering_kernel: bool, optional
    :param outer_subtract_absolute: in outer subtract, use absolute or not, defaults to False
    :type outer_subtract_absolute: bool, optional
    :param double_centering: whether double centering kernel matrix or not, defaults to False
    :type double_centering: bool, optional
    :param verbose: defaults to True
    :type verbose: bool, optional
    :return: distance matrix (key is 'dist_mat')
    :rtype: dict
    """
    printv(
        f"Calculating the kernel matrix using {kernel_type}", verbose=verbose)

    if kernel_type == 'RDE':
        Mat = RDE_transform(df_exp.T,df_pheno,outer_subtract_absolute)
        printv("Finish kernel matrix calculation. Transfer the kernel matrix.", verbose=verbose)
        Mat = KernelMat_transform(Mat,centering_kernel,double_centering)
    
    elif kernel_type == 'RDC':
        Mat = RDC_transform(df_exp.T,df_pheno,centering_kernel,double_centering)
        printv("Finish kernel matrix calculation.", verbose=verbose)

    elif kernel_type == 'RDM':
        def RDM_dist(mat1,mat2):
            z = lambda d: (d - d.mean())/d.std()
            return z(mat1) + z(mat2)
        return RDM_dist(
            relative_diffdist(
                df_exp = df_exp,df_pheno = df_pheno,
                kernel_type='RDE',
                centering_kernel = centering_kernel,
                double_centering= double_centering,
                outer_subtract_absolute= outer_subtract_absolute,
                n_components= n_components,
                verbose = verbose)['dist_mat'],
            relative_diffdist(
                df_exp = df_exp,df_pheno = df_pheno,
                kernel_type='RDC',
                centering_kernel = centering_kernel,
                double_centering= double_centering,
                outer_subtract_absolute= outer_subtract_absolute,
                n_components= n_components,
                verbose = verbose)['dist_mat'],
            )

    else:
        raise TypeError(f"kernel_type should be a string or a function, get {type(kernel_type)}")

    printv("Calculation Distance..", verbose=verbose)
    return decompdist(Mat,n_components)

@kw_decorator(kw=['before_cluster_df','weight'])
def relative_diff(
    df_exp, df_pheno, kernel_type,
    centering_kernel=False,
    double_centering=False,
    outer_subtract_absolute=False,
    n_components = 16,
    verbose=True,
):
    """Kernel Transformation. Important preprocess to cross clustering. In this step, genes from different phenotypes are regarded as different genes. and the distance between them is computed. this function is decorated by :func:`MATTE.utils.kw_decorator`.

    :param df_exp: Expression dataframe.
    :type df_exp: pandas.DataFrame
    :param df_pheno: Phenotype dataframe.
    :type df_pheno: pandas.Series
    :param kernel_type: Kernel type, one of the following: ['RDE','RDC','RDM']; functions are also allowed.
    :type kernel_type: str or function
    :param centering_kernel: whether to centering kernel, defaults to False
    :type centering_kernel: bool, optional
    :param outer_subtract_absolute: in outer subtract, use absolute or not, defaults to False
    :type outer_subtract_absolute: bool, optional
    :param double_centering: whether double centering kernel matrix or not, defaults to False
    :type double_centering: bool, optional
    :param verbose: defaults to True
    :type verbose: bool, optional
    :return: kernel matrix and weight
    :rtype: dict
    """
    if verbose:
        printv(
            f"Calculating the kernel matrix using {kernel_type}", verbose=verbose)

    if kernel_type == 'RDE':
        Mat = RDE_transform(df_exp.T,df_pheno,outer_subtract_absolute)
        Mat = KernelMat_transform(Mat,centering_kernel,double_centering)
    
    elif kernel_type == 'RDC':
        Mat = RDC_transform(df_exp.T,df_pheno,centering_kernel,double_centering)

    elif kernel_type == 'RDM':
        z = lambda d: (d - d.mean())/d.std()
        Mat1,W1 = relative_diff(
                df_exp = df_exp,df_pheno = df_pheno,
                kernel_type='RDE',
                centering_kernel = centering_kernel,
                double_centering= double_centering,
                outer_subtract_absolute= outer_subtract_absolute,
                n_components= n_components,
                verbose = verbose).values()
        Mat2,W2 = relative_diff(
                df_exp = df_exp,df_pheno = df_pheno,
                kernel_type='RDC',
                centering_kernel = centering_kernel,
                double_centering= double_centering,
                outer_subtract_absolute= outer_subtract_absolute,
                n_components= n_components,
                verbose = verbose).values()
        return np.concatenate([z(Mat1),z(Mat2)],axis=1),np.array(W1.tolist()+W2.tolist())

    else:
        raise TypeError(f"kernel_type should be a string or a function, get {type(kernel_type)}")
    pca = PCA(n_components=n_components)


    return pca.fit_transform(Mat),pca.explained_variance_ratio_

