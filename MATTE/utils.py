import time
import numpy as np
from Bio.Cluster import distancematrix
from functools import wraps
import pandas as pd
from inspect import getfullargspec
__all__ = ["printv","affinity_matrix","kw_decorator"]

def printv(*text,show_time=True,verbose=True):
    """Prints text with time and verbose.

    :param show_time: defaults to True
    :type show_time: bool, optional
    :param verbose: defaults to True
    :type verbose: bool, optional
    """    
    if verbose:
        if show_time: print(time.ctime()+"\t",*text)
        else:print(*text)

def affinity_matrix(data, dist_type,type="distance", **kwargs):
    """Calculate affinity matrix; both implement from :mod:`Bio.Cluster` and :mod:`scipy.spatial.distance` are supported.

    :param data: data to calculate affinity matrix
    :type data: numpy.array or pandas.DataFrame
    :param dist_type: string of distance type
    :type dist_type: str
    :param type: `distance` or `affinity`, defaults to "distance"
    :type type: str, optional
    :return: matrix
    :rtype: np.array
    """    
    aff_matrix = np.zeros(shape=(data.shape[0], data.shape[0]))
    for ind, value in enumerate(
            distancematrix(data=data, dist=dist_type, **kwargs)):
        if len(value) == 0:
            continue
        aff_matrix[ind, 0:len(value)] = value
    aff_matrix += aff_matrix.T
    if type == "distance":
        return aff_matrix
    elif type == "affinity":
        return 1-aff_matrix

def kw_decorator(kw=None):
    """Import decarated function used in pipeline.This allow function to accept more than keyworks arguments it self accepted. and will make function return dict as kw set.

    :param kw: keywords make of function's return dict's keys, defaults to None
    :type kw: str or list, optional
    """    
    def funcdec(func):

        @wraps(func)
        def new_fun(*args,**in_use_args):
            func_args_names = getfullargspec(func).args
            in_use_args = {k:w for k,w in in_use_args.items() if k in func_args_names}
            if kw is None:
                return func(*args,**in_use_args)
            elif type(kw) in [list,tuple,np.ndarray,pd.Series]:
                return dict(zip(kw,func(*args,**in_use_args)))
            else:
                return {kw:func(*args,**in_use_args)}
        
        return new_fun

    return funcdec

