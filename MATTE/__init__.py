try:
    import sklearnex
    sklearnex.patch_sklearn()
except Exception:
    print("Import sklearnex failed, highly recommend to use the package to speed up the process.")

from .utils import printv, kw_decorator
from .preprocess import *
import dill as pickle
import os
import pandas as pd
import numpy as np
import warnings
from functools import wraps
from .cluster import Cross_Distance, CrossCluster,build_results
import itertools
from itertools import combinations
from tqdm import tqdm
from random import sample
from copy import deepcopy
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

__version__ = "1.2.1"
__all__ = ["PipeFunc", "AlignPipe",'GeneRanker','merged_pipeline_clustering']

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
    def __init__(self, stats_type='mean', target='cluster',preprocess=True) -> None:
        """Initialize a AlignPipe object.

        :param stats_type: the type of statistics used to calculate the distance between two clusters, defaults to 'mean',should be one of the following: 'mean','median','corr'
        :type stats_type: str, optional
        :param target: the target of clustering, defaults to 'cluster',should be one of the following: 'cluster','distance'
        :type target: str, optional
        :param preprocess: whether to preprocess the data, defaults to True
        :type preprocess: bool, optional
        """        
        self.funcs = []
        self.cluster_func = []
        self.init_pipeline(stats_type,target,preprocess)
        if stats_type not in ['mean','median','corr'] or target not in ['cluster','distance']:
            raise ValueError("init parameters wrong: stats should be one of 'mean','median','corr' and target should be in 'cluster' or 'distance'")
        
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

    def __cal_transform(self, df_exp, df_pheno, saving_temp=False, verbose=True):
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

    def __cal_temp(self, df_exp, df_pheno, saving_temp=False, verbose=True):
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
                printv(f"Running function {str(f)}", verbose=verbose)
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
        tmpt_result = self.__cal_temp(
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
            printv(f"Running function {str(f)}", verbose=verbose)
            f_res = f(**tmpt_result)
            tmpt_result.update(f_res)

        result = tmpt_result["Result"]
        # result.cluster_properties.update({'Process': self.__str__()})
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

    def init_pipeline(self,stats_type,target,preprocess:bool):
        """Initialize the pipeline.

        :param stats_type: type of statistics
        :type stats_type: str
        :param target: target name
        :type target: str
        :param preprocess: whether to preprocess the data
        :type preprocess: bool
        """
        if preprocess:
            self.add_step(inputs_check)
            self.add_step(RPKM2TPM)
            self.add_step(log2transform)
            self.add_step(expr_filter, gene_filter=None)
        
        if stats_type !='corr':
            self.add_step(
                LocKernel_Transform,
                kernel_type=stats_type, centering_kernel=True,
                outer_subtract_absolute=True, double_centering=True
            )

            self.add_transformer(
                PCA(n_components=16))

        else:
            self.add_step(
                CorrKernel_Transform,n_components=16,
            )

        if target == 'cluster':
            self.set_cluster_method(
                func=CrossCluster(),
                preset='kmeans',n_clusters=8, method="a", dist_type="a",n_iters=20
            )
            self.set_cluster_method(func=build_results)
        else:
            self.add_step(Cross_Distance,metric='euclidean')

    def find_best_KernelTrans_params(
        self,df_exp,df_pheno,
        n_downsample = None,n_iters=None,inplace=True,
        verbose=True):
        """Find the best parameters for Kernel_Transform. According to error of cluster.

        :param df_exp: expression data whose index are genes and columns are samples
        :type df_exp: `pandas.DataFrame`
        :param df_pheno: phenotype data whose index are samples
        :type df_pheno: `pandas.Series`
        :param n_downsample: number of down sample, defaults to None
        :type n_downsample: int, optional
        :param n_iters: number of iterations, defaults to None
        :type n_iters: int, optional
        :param inplace: inplace pipeline, defaults to True
        :type inplace: bool, optional
        :param verbose: defaults to True
        :type verbose: bool, optional
        :return: best parameters or :class:`AlignPipe`
        :rtype: :class:`AlignPipe` or dict
        """

        assert (len(self.funcs)!=0) and (len(self.cluster_func)!=0), "Please set the steps first."
        
        n_genes,n_samples = df_exp.shape
        
        if n_downsample is None:
            n_downsample = min(max(int(n_genes * 0.01),100),n_genes)
            printv(f"Auto set n_downsample to {n_downsample}", verbose=verbose)
        if n_iters is None:
            n_iters = n_genes // n_downsample
            printv(f"Auto set n_iters to {n_iters}", verbose=verbose)
        
        # get where 'Kernel_Transform' is  
        for i_funcs,f in enumerate(self.funcs):
            if (type(f) == PipeFunc) and (f.__name__ == 'Kernel_Transform'):
                break
        i_funcs -= 1
        # setup configs/searching space.
        if self.funcs[i_funcs].kwargs['kernel_type'] == 'corr':
            configs = [
                {'centering_kernel':bool(i),
                'double_centering':bool(k)
                } for i in [True,False] for k in [True,False]
                ]
        else:
            configs = [
                {'centering_kernel':bool(i),
                'outer_subtract_absolute':bool(j),
                'double_centering':bool(k)
                } for i in [True,False] for j in [True,False] for k in [True,False]
                ]
        
        # get the best config
        if verbose:
            bar = tqdm(total=len(configs)*n_iters, desc="Auto set parameters")
        for (i, config), _ in itertools.product(enumerate(configs), range(n_iters)):
            tempt_pipe = deepcopy(self)
            tempt_pipe.funcs[i_funcs].add_params(**config)
            sub_dfexp = df_exp.iloc[sample(range(n_genes),n_downsample),:]
            error = tempt_pipe.calculate(sub_dfexp, df_pheno, verbose=False).cluster_properties['error']
            configs[i]['error'] = configs[i].get('error',0) + error
            if verbose:
                bar.update(1)
        if verbose:
            bar.close()
        best_config = min(configs, key=lambda x: x['error'])
        printv(f"Find Best parameters: {best_config}", verbose=verbose)

        if inplace:
            best_config.pop('error')
            self.funcs[i_funcs].add_params(**best_config)
            return self
        else:
            best_config.pop('error')
            return best_config

class GeneRanker():
    """MATTE GeneRanker.

    There are several types of GeneRanker:
    1. `module` and `gene` will cluster genes according to their expression. 
    And Use module SNR to rank genes. In `gene` mode, the SNR will be corrected by the correlation of gene expression and module eigen. 
    2. 'dist','cross-dist'.
    In `dist` mode, the distance of each genes will be calculated. And genes will be ranked according to the sum of distance to each other genes.
    In `cross-dist` mode, the distance of differential expression and differential co-expression will be merged.
    
    .. note:: 
        Inputs of GeneRanker is not the same as pipeline. Row of Expression data is sample, column is gene. 
    """    
    def __init__(self,view,pipeline=None) -> None:
        """Initialize the MATTE GeneRanker.

        :param view: the view of analysis. implemented: 'module','dist','gene','cross-dist'
        :type view: `str`
        :param pipeline: pipeline used to each pair of phenotypes, defaults to None(default pipeline)
        :type pipeline: :class:`MATTE.AlignPipe`, optional

        .. note:: default pipeline contrains **preprocessing**, and if data is preprocessed, delete it.
        """
        self.cluster_res = []
        if view not in ['module','dist','gene','cross-dist']:
            raise ValueError(f"view should be 'module' or 'dist',get {view}")
        self.view = view
        if self.view in ['module','gene']:
            self.pipeline = AlignPipe() if pipeline is None else pipeline
        elif self.view in ['dist']:
            self.pipeline = AlignPipe(target='distance') if pipeline is None else pipeline
        elif self.view in ['cross-view']:
            if pipeline is None:
                pipeline1 = AlignPipe(target='distance',stats_type='mean')
                pipeline2 = AlignPipe(target='distance',stats_type='corr')
                self.pipeline = [pipeline1,pipeline2]
            else:
                assert len(self.pipeline) == 2, "pipeline should be a list of two pipelines in cross-distance."

    def _pipeline_clustering_single(self, X, y):
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
        if verbose:
            bar = tqdm(total=len(list(combinations(self.labels, 2))))
            for idx, (i, j) in enumerate(combinations(self.labels, 2)):
                bar.set_description(f'round {idx}: {i} vs {j}')
                X_ij = X[(y == j) | (y == i)]
                y_ij = y[(y == i) | (y == j)]
                CR = self._pipeline_clustering_single(X_ij, y_ij,)
                self.cluster_res.append(CR)
                bar.update(1)
            bar.close()
        else:
            for i, j in combinations(self.labels, 2):
                X_ij = X[(y == j) | (y == i)]
                y_ij = y[(y == i) | (y == j)]
                CR = self._pipeline_clustering_single(X_ij, y_ij,)
                self.cluster_res.append(CR)

    def _gene_rank_single(self, CR, X,corr=False):
        """Rank genes by their importance sum in one pair of phenotypes comparasion.
        Importance is calculated by the module SNR, see :func:`MATTE.analysis.ClusterResult.ModuleSNR`.

        :param CR: clustering results
        :type CR: :class:`MATTE.analysis.ClusterResult`
        :param X: expression data whose index are samples and columns are genes
        :type X: `pandas.DataFrame`
        :param corr: if True, use correlation of gene co-expression to calculate SNR, defaults to False
        :type corr: bool, optional
        :return: ranked genes
        :rtype: `pandas.Series`
        """        
        sf = CR.SampleFeature(corr=corr)
        if self.view == 'module':
            snr = CR.ModuleSNR(sf)
            importance = pd.Series(data=0.0, index=X.columns)
            for module,genes in CR.module_genes.items():
                importance[genes] = snr[f'{module}_0']
            return importance
        elif self.view == 'gene':
            return CR.GeneSNR(sf)

    def gene_rank_module(self, X: pd.DataFrame, y, verbose=True,**kwargs):
        """ranking genes by their :func:`MATTE.analysis.ClusterResult.ModuleSNR` in each phenotype pairs.
        Each gene in one phenotype is sum to 1000. Meanwhile, set :attr:`GeneRanker.gene_ranking_sep` to calculte each phenotype score.

        :param X: Expression data whose index are samples and columns are genes.
        :type X: `pd.DataFrame`
        :param y: phenotypes whose index are samples.
        :type y: `pd.Series`
        :param verbose: defaults to True
        :type verbose: bool, optional
        :return: gene ranking score
        :rtype: `pd.Series`
        """
        self.labels = np.unique(y)
        if len(self.cluster_res) == 0:
            self.pipeline_clustering(X, y, verbose=verbose)
        if verbose:
            print(f"There are {len(self.labels)} labels: {self.labels}")

        importance_template = pd.Series(data=0.0, index=X.columns)
        clusters_importance = {}
        for idx, (i, j) in enumerate(combinations(self.labels, 2)):
            CR = self.cluster_res[idx]
            X_ij = pd.concat([X[y == i], X[y == j]])
            importance = self._gene_rank_single(CR, X_ij,corr=kwargs.get('corr',False))
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

    def gene_rank_dist(self,X,y,verbose=True,**kwargs):
        """ranking genes of distance in each phenotype pairs.
        by runing this method, :attr:`GeneRanker.dist_mat` will be set.

        :param X: Expression data whose index are samples and columns are genes.
        :type X: `pd.DataFrame`
        :param y: phenotypes whose index are samples.
        :type y: `pd.Series`
        :param verbose: defaults to True
        :type verbose: bool, optional
        :return: gene ranking score
        :rtype: `pd.Series`
        """        """"""
        self.dist_mat = self.pipeline.calculate(X.T,y,verbose=verbose)
        return pd.Series(self.dist_mat.mean(axis=1),index=X.columns)

    def gene_rank(self,X,y,verbose=True,**kwargs):
        """ranking genes.

        :param X: Expression data whose index are samples and columns are genes.
        :type X: `pd.DataFrame`
        :param y: phenotypes whose index are samples.
        :type y: `pd.Series`
        :param verbose: defaults to True
        :type verbose: bool, optional
        :return: gene ranking score
        :rtype: `pd.Series`
        """        """"""
        if self.view in ['module','gene']:
            return self.gene_rank_module(X,y,verbose=verbose,**kwargs)
        elif self.view =='dist':
            return self.gene_rank_dist(X,y,verbose=verbose,**kwargs)
        elif self.view == 'cross-dist':
            z = lambda x:(x-x.mean())/x.std()
            GR1 = GeneRanker(view='dist',pipeline=self.pipeline[0])
            GR2 = GeneRanker(view='dist',pipeline=self.pipeline[1])
            GR1.gene_rank_dist(X,y,verbose=verbose,**kwargs)
            GR2.gene_rank_dist(X,y,verbose=verbose,**kwargs)
            self.sub_steps = [GR1,GR2]
            self.merged_dist = pd.Series(z(GR1.dist_mat).mean(axis=1)+z(GR2.dist_mat).mean(axis=1),index=X.columns)
            return self.merged_dist

    def save(self, save_path):
        """saving the embedder using `dill`

        :param save_path: path to save the embedder
        :type save_path: str
        """        
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

def merged_pipeline_clustering(df_exp:pd.DataFrame,df_pheno:pd.Series,pipelines:list,verbose=True):
    """clustering genes using multiple pipelines.

    :param df_exp: Expression data whose index are genes and columns are samples.
    :type df_exp: `pd.DataFrame`
    :param df_pheno: phenotypes whose index are samples.
    :type df_pheno: `pd.Series`
    :param pipelines: list of :class:`MATTE.AlignPipe`
    :type pipelines: list
    :param verbose: defaults to True
    :type verbose: bool, optional
    :return: Cluster Result
    :rtype: :class:`MATTE.analysis.ClusterResult`
    """
    data_name = 'before_cluster_df'
    z = lambda x: (x - x.mean()) / x.std()
    dfs = []
    for pipeline in pipelines:
        results:dict = pipeline._AlignPipe__cal_temp(df_exp,df_pheno,verbose=verbose)
        dfs.append(z(results[data_name]))

    before_cluster_df = np.concatenate(dfs,axis=1)
    results.update({data_name:before_cluster_df,})
    
    printv(f"merged PCA transformed embedding, get {before_cluster_df.shape} data.",verbose=verbose)
    return pipeline.calculate_from_temp(results,verbose=verbose)

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
    :return: clustering result and data
    :rtype: tuple contains :class:`MATTE.analysis.ClusterResult` and a dict of data
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
    pipe = AlignPipe() if pipe is None else pipe
    R = pipe.calculate(df_exp, df_pheno, verbose=verbose)
    if verbose:
        print(R.summary())
    return R,{'df_exp':df_exp,'df_pheno':df_pheno}