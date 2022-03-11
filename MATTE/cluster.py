import numpy as np
import pandas as pd
from Bio.Cluster import kcluster
from sklearn.cluster import spectral_clustering, SpectralBiclustering, KMeans
from sklearn.metrics import calinski_harabasz_score as ch_score
from sklearn.model_selection import KFold
from .analysis import ClusterResult, affinity_matrix
from .utils import *

__all__ = ['CrossCluster','build_results']

class CrossCluster():
    """Cross clustering of mixed expression matrix.
    if `preset` is not set, then must use :func:`CrossCluster.build_from_func` or :func:`CrossCluster.build_from_model` to set the function to cluster. 
    """    
    def __init__(
        self, preset=None, 
        use_affinity=False, verbose=True,
        **kwargs) -> None:
        """

        :param preset: presetted mod,accept one of below: `kmeans`,`spectrum` and `spectral_bicluster`, defaults to None
        :type preset: str, optional
        :param use_affinity: fit calculate affinity to cluster, defaults to False
        :type use_affinity: bool, optional
        :param verbose: defaults to True
        :type verbose: bool, optional
        """        
        super().__init__()
        self.__name__ ='CrossCluster'
        self.use_aff = use_affinity
        self.verbose = verbose
        self.properties = {}
        self.kwargs = kwargs
        self.presetting = preset

    def build_from_func(self,func):
        """build this class from function

        :param func: functions to cluster
        :type func: function
        """        
        self.cluster_func = func
    
    def build_from_model(self,model,model_attr='label_',**calling_kwargs):
        """build this class from model

        :param model: model to cluster, must have `model_attr` attribute and `fit` method
        :type model: object
        :param model_attr: defaults to 'label_'
        :type model_attr: str, optional
        """        
        def calling(data):
            model.fit(data,**calling_kwargs)
            return getattr(model,model_attr)
        self.cluster_func = calling
    
    def __call__(self,before_cluster_df,**kwargs) -> dict:
        """call this class from built function or model to cluster

        :param before_cluster_df: dataframe to cluster
        :type before_cluster_df: pandas.DataFrame or np.array
        :param kwargs: kwargs to cluster function
        :return: cluster result including `cluster_label` and `cluster_properties`
        :rtype: dict
        """        
        self.kwargs.update(kwargs)
        self.preset()

        if self.use_aff:
            dist_type = kwargs.get("dist_type", "a")
            weight = kwargs.get("weight", [1]*before_cluster_df.shape[1])

            printv("Using affinity matrix will cost more time and meomory.",verbose=self.kwargs.get('verbose',True))
            if dist_type not in ["a", "x", "cosine", "correlation"]:
                print("Affinity is not distance, make sure your distance type is true.(Correlation-like should be used, like 'a','x','cosine','correlation'). Or it will cost much time and get no result.")
            before_cluster_df = affinity_matrix(data=before_cluster_df, dist_type=dist_type, type="affinity", weight=weight)

        labels,properties = self.cluster_func(before_cluster_df,**kwargs)
        self.properties.update(properties)
        return {'cluster_label':labels, 'cluster_properties':self.properties}

    def preset(self):
        """preset the function to cluster

        :raises NotImplementedError: `preset` is not implemented 
        """        
        preset = self.presetting
        if preset is None:
            return 
        if type(preset) is str:
            if preset == 'kmeans':
                self.preset_kmeans()
            elif preset == 'spectrum':
                self.preset_spectrum()
            elif preset == 'spectral_bicluster':
                self.preset_spectral_bicluster()
            else:
                raise NotImplementedError(f'preset = {preset} is not implemented')
        elif type(preset) is function:
            self.build_from_func(preset)
        else:
            try:
                self.build_from_model(preset)
            except Exception as e:
                TypeError(f"preset should be str/function/model;current {type(preset)} and cause {e}")

    def preset_kmeans(self):
        """preset kmeans, default kwargs: `n_clusters`=8,`method`='a', `npass`=20
        """        
        def kcluster_calling(data,**kwargs):
            method = kwargs.get("method", "a")
            dist_type = kwargs.get("dist_type", "a")
            weight = kwargs.get("weight", [1]*data.shape[1])
            n_clusters = kwargs.get("n_clusters", 8)
            npass = kwargs.get("npass", 20)
            label, error, nfound = kcluster(
                data=data, method=method,
                nclusters=n_clusters, weight=weight,
                npass=npass, dist=dist_type)
            return label,{'error':error,'method':f'kmeans_{method}','dist_type':dist_type,'n_clusters':n_clusters,'npass':npass}
        self.build_from_func(kcluster_calling)

    def preset_spectrum(self,):
        """preset spectral clustering, default kwargs: `n_clusters`=8,`use_aff`=True, `n_init`=10
        """        
        def spectrum_calling(data,**kwargs):
            n_clusters = kwargs.get("n_clusters", 8)
            n_component = max(n_clusters) if n_component is None else n_component
            n_init = kwargs.get("n_init", 10)
            weights = [1]*data.shape[1] if weights is None else weights
            return spectral_clustering(
                n_clusters=n_clusters,
                n_component=n_component,
                n_init=n_init,
                affinity=data,
            ),{'n_clusters':n_clusters,'n_component':n_component,'n_init':n_init,'method':'spectrum',}
        self.use_aff=True
        self.build_from_func(spectrum_calling)

    def preset_spectral_bicluster(self):
        """preset spectral bicluster, default kwargs: `n_clusters`=8,`use_aff`=True, `n_init`=10,`method`='log',`n_component`=`m_cluster`, `use_aff`=True
        """        
        def spectral_bicluster_calling(data,**kwargs):
            n_clusters = kwargs.get("n_clusters", 8)
            n_component = max(n_clusters) if n_component is None else n_component
            n_init = kwargs.get("n_init", 10)
            model = SpectralBiclustering(
                n_clusters=n_clusters, method="log", n_components=n_component, n_init=n_init)
            model.fit(data)
            return model.row_labels_,{'method':model,'n_clusters':n_clusters,'n_component':n_component,'n_init':n_init}
        self.use_aff = True
        self.build_from_func(spectral_bicluster_calling)


@kw_decorator(kw="Result")
def build_results(
        cluster_label, mixed_genes, cluster_properties, df_exp, df_pheno,
        before_cluster_df, order_rule="input", verbose=True):
    """building results of :class:`CrossCluster` to :class:`MATTE.analysis.ClusterResult`
    **decorated by :func:`MATTE.utils.kwdecorator`**

    :param cluster_label: label of clustering
    :type cluster_label: array like
    :param mixed_genes: index of mixed genes(in the order of cluster_label)
    :type mixed_genes: array like
    :param cluster_properties: information get from clustering
    :type cluster_properties: dict
    :param df_exp: inputs to all pipeline
    :type df_exp: `pandas.DataFrame`
    :param df_pheno: inputs to all pipeline
    :type df_pheno: `pandas.Series`
    :param before_cluster_df: inputs to :class:`CrossCluster`
    :type before_cluster_df: `pandas.DataFrame`
    :param order_rule: how to order module. 'input'(mean exp) or 'size'(module genes) or function; defaults to "input"
    :type order_rule: str, optional
    :param verbose: defaults to True
    :type verbose: bool, optional
    :return: results
    :rtype: :class:`MATTE.analysis.ClusterResult`
    """    
    printv("building cluster results", verbose=verbose)
    genes = mixed_genes
    assert len(genes) == len(
        cluster_label), "Contains duplicated genes, please check inputs."
    res = pd.Series(index=genes, data=cluster_label)
    cluster_properties["score"] = ch_score(before_cluster_df, res)
    before_cluster_df = pd.DataFrame(before_cluster_df, index=genes)
    res = ClusterResult(
        cluster_res=res, df_exp=df_exp, df_pheno=pd.DataFrame(
            df_pheno).iloc[:, 0],  # Only first column will be sabved in Result.pheno
        before_cluster_df=before_cluster_df,
        cluster_properties=cluster_properties,
        order_rule=order_rule)

    return res
