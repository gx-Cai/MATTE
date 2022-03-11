from .utils import printv, kw_decorator
from .preprocess import *
from sklearn.decomposition import PCA
import dill as pickle
import os
import pandas as pd
import numpy as np
import warnings
from functools import wraps
from .cluster import CrossCluster,build_results
from itertools import combinations
from tqdm import tqdm
warnings.filterwarnings('ignore')

__version__ = "1.0"
__all__ = ["PipeFunc", "AlignPipe", 'preprocess_funcs_generate','ModuleEmbedder']

class PipeFunc():
    """`PipeFunc` is a wrapper of a function with storing arguments and kwargs **but not run it** until calling it self. And is used in :class:`AlignPipe` as a step.
    """
    def __init__(self, func, name=None, *args, **kwargs) -> None:
        """Pipe function used in :class:`AlignPipe`

        :param func: base function
        :type func: function
        :param name: names show in `str` func, defaults to None
        :type name: str, optional

        """

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.__name__ = func.__name__ if name is None else name
        self.__doc__ = func.__doc__

    def __str_generate(self) -> str:
        argtext = " ".join([self.__argstext(i) for i in self.args])
        kwargtext = ",".join(
            [f"{k}={self.__argstext(v)}" for k, v in self.kwargs.items()])
        return f"""<PipeFunc> {self.func.__name__}({argtext}{kwargtext})"""

    def __str__(self) -> str:
        return self._strtext if hasattr(self,"_strtext") else self.__str_generate()

    def __update_str(func):
        """update the :code:str(self)

        :param func: _description_
        :type func: _type_
        """
        @wraps(func) 
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._strtext = self.__str_generate()
            return result
        return wrapper

    def __call__(self, *args, **kwargs):
        args = self.args + args
        kwargs = {**self.kwargs, **kwargs}
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return self.__str__()

    @__update_str
    def add_params(self, *args, **kwargs):
        """Add params to PipeFunc, and refresh `str` text
        """        
        self.args += args
        self.kwargs.update(kwargs)

    def __argstext(self, arg):
        if type(arg) in [pd.DataFrame, pd.Series, np.ndarray]:
            return f"{type(arg)}({arg.shape})"
        elif type(arg) == str:
            return '\''+str(arg) + '\''
        else:
            return str(arg)


class AlignPipe():
    """Basic class in MATTE. Stores a list of functions or transformers.
    One can see the pipeline in :func:`__str__`

    The funcstions in :attr:`funcs` are called in order, and the results are passed to the next function. All returns of the functions should be `dict`,(using :func:`MATTE.utils.kw_decorator` are recommended). Then :attr:`cluster_func` will be called to cluster the results.

    .. note:: 
        Use :func:`MATTE.AlignPipe.add_step` to add a function to :attr:`funcs`, but can change order or delete some functions like `list`.
    
    """    
    def __init__(self, init=True, **config) -> None:
        """Initialize a AlignPipe object.

        :param init: weather to set up to default pipeline or not, defaults to True
        :type init: bool, optional
        """        
        for k, v in config.items():
            setattr(self, k, v)
        self.funcs = []
        self.cluster_func = []
        if init:
            self.default_pipeline()
    def __str__(self) -> str:
        """str is a string that shows the pipeline.

        :return: _summary_str (containing functions and parameters)
        :rtype: str
        """        
        strtext = "MATTE calculation pipeline\n"
        for n, f in enumerate(self.funcs):
            strtext += f"## STEP {n} \t"
            strtext += str(f)+"\n"
        for n, f in enumerate(self.cluster_func):
            strtext += f"## CLUSTER STEP {n} \t"
            strtext += str(f)+"\n"
        return strtext

    def __repr__(self) -> str:
        return self.__str__()

    def add_param(self, **kwargs):
        """add a function that return parameters to :attr:`funcs`.
        Multiple parameters can be added at the same time.
        """        
        for name, parm in kwargs.items():
            @kw_decorator(kw=name)
            def add_param():
                f"""adding params {name}."""
                return parm

            new_func = PipeFunc(func=add_param, name=f"add_{name}")
            self.funcs.append(new_func)

    def add_step(self, func, *setting, **kwsetting):
        """Add a function to :attr:`funcs`

        :param func: function(should be decarated by :func:`kw_decorator`) to add
        :type func: function
        """        
        new_func = PipeFunc(func, *setting, **kwsetting)
        self.funcs.append(new_func)

    def add_transformer(self, transformer):
        """Add a transformer to :attr:`funcs`

        :param transformer: transformer to add
        :type transformer: objects that has `fit` and `transform` methods
        :raises TypeError: if transformer is not a transformer
        """        
        if (not hasattr(transformer, "fit")) or (not hasattr(transformer, "transform")):

            raise TypeError(
                "The object is not a transformer (containing fit and transform function).")

        self.funcs.append(transformer)

    def set_cluster_method(self, func, *setting, **kwsetting):
        """Add a function to :attr:`cluster_func`

        :param func: function(should be decarated by :func:`kw_decorator`) to add
        :type func: function
        """        
        self.cluster_func.append(PipeFunc(func=func, *setting, **kwsetting))

    def __generate_transform_function(self, calling, kw="before_cluster_df"):
        if calling == "fit":

            @kw_decorator(kw=kw)
            def fit(transformer, before_cluster_df):
                return transformer.fit_transform(before_cluster_df)

            return fit

        else:

            @kw_decorator(kw=kw)
            def transform(transformer, before_cluster_df):
                return transformer.transform(before_cluster_df)

            return transform

    def sub_pipe(self, funcs_index, cluster_methods_index):

        new_pipe = AlignPipe(init=False)
        new_pipe.funcs = self.funcs[funcs_index]
        new_pipe.cluster_func = self.cluster_func[cluster_methods_index]
        return new_pipe

    def transform(self, df_exp, df_pheno, saving_temp=False, verbose=True):
        """Transform the dataframe using the pipeline.
        All parameters are same to :func:`AlignPipe.fit_transform` except use `transform` instead of `calculate`.
        """        
        tmpt_result = {"df_exp": df_exp, "df_pheno": df_pheno}
        for f in self.funcs:
            if type(f) == PipeFunc:
                printv(f"Running function {f.__name__}", verbose=verbose)
                tmpt_result.update(f(**tmpt_result))
            else:
                printv(f"Tranforming using model {f}", verbose=verbose)
                _f = self.__generate_transform_function(calling="transform")
                tmpt_result.update(_f(transformer=f, **tmpt_result))

        if saving_temp:
            if not os.path.exists(saving_temp):
                os.mkdir(saving_temp)

            with open(os.path.join(saving_temp, 'tempt.result'), "wb") as tempf:
                pickle.dump(tmpt_result, tempf)

    def fit_transform(self, df_exp, df_pheno, saving_temp=False, verbose=True):
        """Fit the data using the pipeline. If first performed, use :func:`AlignPipe.calculate` instead.

        :param df_exp: expression data whose index are genes and columns are samples
        :type df_exp: `pandas.DataFrame`
        :param df_pheno: phenotype data whose index are samples
        :type df_pheno: `pandas.Series`
        :param saving_temp: whether to save temp or not.if not False,set the parameter to be saving dir, defaults to False
        :type saving_temp: bool or str, optional
        :param verbose: defaults to True
        :type verbose: bool, optional
        :return: final results containing all results of the pipeline functions
        :rtype: dict
        """        

        tmpt_result = {"df_exp": df_exp,
                       "df_pheno": df_pheno, "verbose": verbose}
        for f in self.funcs:
            if type(f) == PipeFunc:
                printv(f"Running function {f.__name__}", verbose=verbose)
                f_res = f(**tmpt_result)
                tmpt_result.update(f_res)
            else:
                printv(f"Tranforming using model {f}", verbose=verbose)
                _f = self.__generate_transform_function(calling="fit")
                tmpt_result.update(_f(transformer=f, **tmpt_result))

        if saving_temp:
            printv(f"Saving tmpt result in {saving_temp}", verbose=verbose)
            if not os.path.exists(saving_temp):
                os.mkdir(saving_temp)
            with open(os.path.join(saving_temp, 'tempt.result'), "wb") as tempf:
                pickle.dump(tmpt_result, tempf)

        return tmpt_result

    def calculate(self, df_exp, df_pheno, verbose=True):
        """Calculate the data using the pipeline.

        :param df_exp: expression data whose index are genes and columns are samples
        :type df_exp: `pandas.DataFrame`
        :param df_pheno: phenotype data whose index are samples
        :type df_pheno: `pandas.Series`
        :param verbose: defaults to True
        :type verbose: bool, optional
        :return: clustering results
        :rtype: :class:`MATTE.analysis.ClusterResult`
        """        

        tmpt_result = self.fit_transform(
            df_exp, df_pheno, verbose=verbose, saving_temp=False)

        return self.calculate_from_temp(tmpt_result, verbose)

    def calculate_from_temp(self, tmpt_result, verbose=True):
        """Calculate the data using the pipeline but from temp file.

        :param tmpt_result: temp result saved by :func:`MATTE.AlignPipe.fit_transform`
        :type tmpt_result: dict
        :param verbose: defaults to True
        :type verbose: bool, optional
        :return: clustering results
        :rtype: :class:`MATTE.analysis.ClusterResult`
        """        
        for f in self.cluster_func:
            printv(f"Running function {f.__name__}", verbose=verbose)
            f_res = f(**tmpt_result)
            tmpt_result.update(f_res)

        result = tmpt_result["Result"]
        result.cluster_properties.update({'Process': self.__str__()})
        return result

    def get_attribute_from_transformer(self, attribute):
        """Get the attribute from the transformer.

        :param attribute: attribute name
        :type attribute: str
        :return: attribute value
        :rtype: any
        """        
        return next(
            (
                getattr(i, attribute)
                for i in self.funcs
                if type(i) != PipeFunc and hasattr(i, attribute)
            ),
            None,
        )

    def default_pipeline(self):
        """Get the default pipeline.

        :return: default pipeline
        :rtype: :class:`AlignPipe`

        .. note:: default pipeline contrains **preprocessing**, and if data is preprocessed, delete it.
        
        see default pipeline, run code below to see the result. 

        .. code-block:: python

            import MATTE
            print(MATTE.AlignPipe())

        """        

        self.add_step(inputs_check)
        self.add_step(RPKM2TPM)
        self.add_step(log2transform)
        self.add_step(exp_filter, gene_filter=None)
        self.add_step(
            Kernel_transform,
            kernel_type="mean", centering_kernel=True,
            outer_subtract_absolute=True, double_centering=True
        )
        self.add_transformer(
            PCA(n_components=16))

        @kw_decorator(kw='weights')
        def adding_weights():
            return self.get_attribute_from_transformer('explained_variance_')

        self.add_step(
            func=adding_weights)
        clustering_func = CrossCluster(preset='kmeans',n_clusters=8, method="a", dist_type="a",n_iters=20)
        self.set_cluster_method(
            func=clustering_func,
        )
        self.set_cluster_method(func=build_results)

        return self



class ModuleEmbedder():
    """MATTE embedder. This class is used to embeding the data using MATTE pipeline.

    There are two types of embedding:
    
    1. feature selection: ranking gene by its importance sum in each pair of phenotypes comparasion.
    2. feature extraction: using pca to transfrom data to most variant space 

    .. note:: 
        Inputs of ModuleEmbedder is not the same as pipeline. Row of Expression data is sample, column is gene. 
    """    
    def __init__(self, pipeline=None) -> None:
        """Initialize the MATTE embedder.

        :param pipeline: pipeline used to each pair of phenotypes, defaults to None(default pipeline)
        :type pipeline: :class:`MATTE.AlignPipe`, optional

        .. note:: default pipeline contrains **preprocessing**, and if data is preprocessed, delete it.
        """        
        self.pipeline = AlignPipe() if pipeline is None else pipeline
        self.cluster_res = []

    def _pipeline_clustering_single(self, X, y,):
        CR = self.pipeline.calculate(df_exp=X.T, df_pheno=y, verbose=False)
        self.cluster_res.append(CR)
        return CR

    def pipeline_clustering(self, X, y, verbose=True):
        """Clustering the data using pipeline for each pair of phenotypes.
        A progress bar is set to inform the user about the progress.

        :param X: expression data whose index are samples and columns are genes
        :type X: `pandas.DataFrame`
        :param y: phenotype data whose index are samples
        :type y: `pandas.Series`
        :param verbose: defaults to True
        :type verbose: bool, optional
        """        
        bar = tqdm(total=len(list(combinations(self.labels, 2))))
        for idx, (i, j) in enumerate(combinations(self.labels, 2)):
            bar.set_description(f'round {idx}: {i} vs {j}')
            X_ij = X[(y == j) | (y == i)]
            y_ij = y[(y == i) | (y == j)]
            CR = self._pipeline_clustering_single(X_ij, y_ij,)
            self.cluster_res.append(CR)
            bar.update(1)
        bar.close()

    def _gene_rank_single(self, CR, X):
        """Rank genes by their importance sum in one pair of phenotypes comparasion.
        Importance is calculated by the module SNR, see :func:`MATTE.analysis.ClusterResult.ModuleSNR`.

        :param CR: clustering results
        :type CR: :class:`MATTE.analysis.ClusterResult`
        :param X: expression data whose index are samples and columns are genes
        :type X: `pandas.DataFrame`
        :return: ranked genes
        :rtype: `pandas.Series`
        """        
        sf = CR.SampleFeature()
        snr = CR.ModuleSNR(sf)
        importance = pd.Series(data=0.0, index=X.columns)
        for module,genes in CR.module_genes.items():
            importance[genes] = snr[f'{module}_0']
        return importance

    def _transform_single(self, CR, thres=0.85):
        """transform in single pair of phenotypes comparasion.

        :param CR: clustering results
        :type CR: :class:`MATTE.analysis.ClusterResult`
        :param thres: thereshold used in selecting genes to :class:`sklearn.decomposition.PCA`,(sum of select genes' explain variant ratio reach `thres`) defaults to 0.85
        :type thres: float, optional
        :return: transformer 
        :rtype: :class:`sklearn.decomposition.PCA`
        """        
        sf = CR.SampleFeature()
        snr = CR.ModuleSNR(sf)
        # if not (snr >= 1.5).any():
        #     return None
        i = 0
        while True:
            module = snr.index[i].split('_')[0]
            genes = CR.module_genes[module]
            if len(genes) > 1:
                break
            i += 1

        pca = PCA(n_components=None)
        X = CR.df_exp.loc[genes, :].T
        pca.fit(X)
        ## sort genes by explained variance; 
        # then select the top genes until the explained variance is larger than thres
        exp_vr = pca.explained_variance_ratio_
        select_genes_idx =  np.argsort(exp_vr)[::-1][:np.where(np.cumsum(np.sort(exp_vr)[::-1])>thres)[0][0]]
        select_genes = pca.feature_names_in_[select_genes_idx]

        pca = PCA(n_components=None)
        X = CR.df_exp.loc[select_genes, :].T
        pca.fit(X)
        return pca

    def transform(self, X):
        """transform the data using pipeline for each pair of phenotypes.
        Run :func:`MATTEEmbedder.transform_fit` first.

        :param X: expression data whose index are samples and columns are genes
        :type X: `pandas.DataFrame`
        :return: transformed data
        :rtype: `pandas.DataFrame`
        """        
        assert hasattr(self, 'transformers'), 'Run transform_fit first.'
        Xs = []
        used_genes = set()
        for transformer in self.transformers:
            features = transformer.feature_names_in_
            used_genes = set(features) | used_genes
            Xs.append(transformer.transform(X.loc[:, features]))
        Xs = np.concatenate(Xs, axis=1)
        self.used_genes = list(used_genes)
        return Xs

    def gene_rank(self, X: pd.DataFrame, y, verbose=True):
        """ranking genes by their :func:`MATTE.analysis.ClusterResult.ModuleSNR` in each phenotype pairs.
        Each gene in one phenotype is sum to 1000. Meanwhile, set :attr:`ModuleEmbedder.gene_ranking_sep` to calculte each phenotype score.

        :param X: Expression data whose index are samples and columns are genes.
        :type X: `pd.DataFrame`
        :param y: phenotypes whose index are samples.
        :type y: `pd.Series`
        :param verbose: defaults to True
        :type verbose: bool, optional
        :return: gene ranking score
        :rtype: `pd.Series`
        """
        if len(self.cluster_res) == 0:
            self.pipeline_clustering(X, y, verbose=verbose)
        
        self.labels = np.unique(y)
        if verbose:
            print(f"There are {len(self.labels)} labels: {self.labels}")

        importance_template = pd.Series(data=0.0, index=X.columns)
        clusters_importance = {}
        for idx, (i, j) in enumerate(combinations(self.labels, 2)):
            CR = self.cluster_res[idx]
            X_ij = pd.concat([X[y == i], X[y == j]])
            importance = self._gene_rank_single(CR, X_ij)
            clusters_importance[i] = clusters_importance.get(i,importance_template.copy()) + importance
            clusters_importance[j] = clusters_importance.get(j,importance_template.copy()) + importance
        # overall importance is the sum of the cluster importance
        gene_ranking = pd.Series(data=0.0, index=X.columns)
        for cluster, importance in clusters_importance.items():
            gene_ranking += 1000*importance/importance.sum()
        gene_ranking /= len(self.labels)
        self.gene_ranking = gene_ranking
        self.gene_ranking_sep = clusters_importance
        return gene_ranking

    def transform_fit(self, X: pd.DataFrame, y, verbose=True):
        """transform the data using pipeline for each pair of phenotypes.

        :param X: expression data whose index are samples and columns are genes
        :type X: `pandas.DataFrame`
        :param y: phenotypes whose index are samples.
        :type y: `pandas.Series`
        :param verbose: defaults to True
        :type verbose: bool, optional
        :return: self
        :rtype: :class:`ModuleEmbedder`
        """        
        self.transformers = []
        self.labels = np.unique(y)
        if verbose:
            print(f"There are {len(self.labels)} labels: {self.labels}")

        if len(self.cluster_res) == 0:
            self.pipeline_clustering(X, y, verbose=verbose)

        for idx, (i, j) in enumerate(combinations(self.labels, 2)):
            CR = self.cluster_res[idx]
            transformer = self._transform_single(CR)
            if transformer is not None:
                self.transformers.append(transformer)

        return self

    def save(self, save_path):
        """saving the embedder using `dill`

        :param save_path: path to save the embedder
        :type save_path: str
        """        
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

def make_pipefunc(func,kw='df_exp'):

    @kw_decorator(kw=kw)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return PipeFunc(wrapper, name=func.__name__)

def package_test(n_genes=100, n_samples=100, pipe=None,seed=0,verbose=True):
    """testing the package by generating data.

    :param n_genes: number of genes, defaults to 100
    :type n_genes: int, optional
    :param n_samples: number of samples, defaults to 100
    :type n_samples: int, optional
    :param pipe: pipeline to use, defaults to None(default pipeline)
    :type pipe: :class:`MATTE.AlignPipe`, optional
    :param seed: random seed, defaults to 0
    :type seed: int, optional
    :param verbose: defaults to True, print all the information
    :type verbose: bool, optional
    :return: clustering result
    :rtype: :class:`MATTE.analysis.ClusterResult`
    """    
    np.random.seed(seed)
    ## generate a test data
    genes = [f"gene{i}" for i in range(n_genes)]
    samples = [f"sample{i}" for i in range(n_samples)]
    base_exp = np.random.gamma(3,1,(n_genes))
    ## from base_exp select 20% to change to a new value
    disturb_exp = base_exp.copy()
    disturb_exp[0:int(n_genes/5)] = np.random.gamma(3,1,int(n_genes/5))
    
    df_pheno = pd.Series(
        [f"P{i}" for i in np.random.randint(size=n_samples, low=0, high=2)],
        index=samples,)
    df_exps = []
    for i in range(2):
        sub_samples = df_pheno[df_pheno == f"P{i}"].index
        df_exp = pd.DataFrame(
            np.random.normal(
                loc=[base_exp,disturb_exp][i],
                size=(sub_samples.size,n_genes)),
            index=sub_samples,
            columns=genes,
        ).T
        df_exps.append(np.abs(df_exp))
    df_exp = pd.concat(df_exps, axis=1)

    ## generate a test pipe
    pipe = AlignPipe(init=True) if pipe is None else pipe
    R = pipe.calculate(df_exp, df_pheno, verbose=verbose)
    if verbose:
        print(R.summary())
    return R,{'df_exp':df_exp,'df_pheno':df_pheno}
