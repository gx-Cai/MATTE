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
from .cluster import CrossCluster,build_results
warnings.filterwarnings('ignore')

__version__ = "1.2.0-dev3"
__all__ = ["PipeFunc", "AlignPipe", 'preprocess_funcs_generate','GeneRanker']

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
    def __init__(self, init='cluster') -> None:
        """Initialize a AlignPipe object.

        :param init: weather to set up to default pipeline or not, defaults to True
        :type init: bool, optional
        """        
        self.funcs = []
        self.cluster_func = []
        if init == 'cluster':
            self.default_cluster()
        elif init == 'generank':
            self.default_generank()
        
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

    def sub_pipe(self, funcs_index, cluster_methods_index):

        new_pipe = AlignPipe(init=False)
        new_pipe.funcs = self.funcs[funcs_index]
        new_pipe.cluster_func = self.cluster_func[cluster_methods_index]
        return new_pipe

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

    def default_cluster(self,diff_type='RDE'):
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
            relative_diff,
            kernel_type=diff_type, 
            centering_kernel=True, double_centering=False,
            n_components=16
        )

        self.set_cluster_method(
            func=CrossCluster(),
            presetting='kmeans',
            method='a',
            dist_type='e',
            n_clusters=8,
            npass = 20
        )
        self.set_cluster_method(func=build_results)

        return self

    def default_generank(self,diff_type='RDE'):
        self.add_step(inputs_check)
        self.add_step(RPKM2TPM)
        self.add_step(log2transform)
        self.add_step(exp_filter, gene_filter=None)
        self.add_step(
            relative_diffdist,
            kernel_type=diff_type, 
            centering_kernel=True, double_centering=False,n_components=16
        )

        @kw_decorator('Result')
        def diff_k(df_exp,dist_mat):
            return pd.Series(dist_mat.sum(axis=1),index=df_exp.index)

        self.add_step(diff_k)

        return self

