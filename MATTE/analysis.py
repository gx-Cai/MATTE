#-- ploting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import seaborn as sns
# -- data process
import pandas as pd
import numpy as np
#-- models
from sklearn.metrics import calinski_harabasz_score as ch_score
from sklearn.metrics import davies_bouldin_score as db_score
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from scipy.spatial.distance import pdist, squareform
#-- tools
from copy import deepcopy
from tqdm import trange
from .utils import *
__all__=["WeightedDataFrame","ClusterResult","Fig_Fuction","Fig_SampleFeature","FunctionEnrich"]

class WeightedDataFrame(pd.DataFrame):
    """ A weighted dataframe inherit `pandas.DataFrame`, and add a new attribute `weight` to store the weight of each column.
    """    
    def __init__(self, weight=None, data=None, index=None, columns=None, dtype=None, copy=None) -> None:
        """except weight, others parameters are same to pandas.DataFrame.

        :param weight: weight of dataframe, defaults to None
        :type weight: array-like, optional
        """        
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.weight = pd.Series(weight, index=columns)
        if len(self.weight) == 0:
            self.weight = [1]*self.values.shape[1]

    def weight_distance(self, metric='c', **kwargs):
        """Calculate weighted distance.

        :param metric: The distance metric to use., defaults to 'c'
        :type metric: str, optional

        .. note:: calling from different module can be very different for their definition is not same.
        The distance function can be :
        --- calling from scipy.spatial.pdist ---
        'braycurtis', 'canberra', 'chebyshev',
        'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
        'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule'.
        --- calling from Bio.cluster.distancematrix ---
        'e': Euclidean ;
        'b': City-block .
        'c': Pearson correlation;
        'a': Absolute of Pearson correlation;
        'u': Uncentered Pearson correlation;
        'x': Absolute of Uncentered Pearson correlation;
        's': Spearman's correlation;
        'k': Kendall's τ.
        
        :return: distance matrix
        :rtype: numpy.array
        """
        if metric not in ["a", "x"]:
            print(
                "Calculate distance will not return negtive value. Result will be absolute!")
        if len(metric) > 1:
            return squareform(
                pdist(self.values, metric=metric,
                      w=self.weight.values, **kwargs)
            )
        else:
            return abs(affinity_matrix(data=self.values,
                                             dist_type=metric, weight=self.weight.values))


class ClusterResult():
    """A class to store the result of clustering. contains inputs and results.
    """    
    def __init__(self, cluster_res, before_cluster_df,df_exp,df_pheno,cluster_properties,order_rule="input") -> None:
        """

        :param cluster_res: cluster result of :class:`MATTE.cluster.CrossCluster`
        :type cluster_res: array-like
        :param before_cluster_df: inputs to :class:`MATTE.cluster.CrossCluster`
        :type before_cluster_df: numpy.array or pandas.DataFrame
        :param df_exp: inputs to all the pipeline
        :type df_exp: pandas.DataFrame
        :param df_pheno: inputs in all the pipeline
        :type df_pheno: pandas.Series
        :param cluster_properties: cluster result of :class:`MATTE.cluster.CrossCluster`
        :type cluster_properties: dict
        :param order_rule: order modules by the rule, defaults to "input"
        :type order_rule: str, optional
        """        
        self.n_cluster = cluster_res.max()+1
        self.cluster_properties = cluster_properties  # prperty record the score and loss of the cluster results
        self.label = pd.Series([i.split("@")[1]
                                for i in cluster_res.index]).unique()
        self.df_exp = df_exp
        self.pheno = df_pheno
        self._reorder(before_cluster_df, cluster_res,by=order_rule)
        self._fixingRes(cluster_res, label=self.label)
        self._JMatrix()


    # --- built-in methods

    def _reorder(self, before_cluster_df, cluster_res, by="input"):
        """reorder cluster by data.

        :param before_cluster_df: preprcossed exp df.
        :type before_cluster_df: pd.DataFrame
        :param cluster_res: cluster result of :class:`MATTE.cluster.CrossCluster`
        :type cluster_res: array-like
        :param by: order rule, defaults to "input". 'size' is also valid option. 
        :type by: str,or function optional
        """
        if by=="input":
            assert len(cluster_res) == before_cluster_df.shape[0]
            df_input_ = deepcopy(before_cluster_df)
            df_input_["group"] = cluster_res
            rank = pd.Series(df_input_.groupby(
                "group").median().median(axis=1).rank(), dtype=int)
        elif by=="size":
            rank = pd.Series(cluster_res.value_counts().rank(ascending=False),dtype=int)
        elif type(by) == type(np.sum):
            rank = pd.Series(by(cluster_res,before_cluster_df),dtype=int)
        for i in range(cluster_res.shape[0]):
            cluster_res[i] = rank[cluster_res[i]]-1


    def _fixingRes(self, cluster_res, label, sep="@"):
        """from cluster results to build a dataframe, remove special sperate @.

        :param cluster_res: cluster result of :class:`MATTE.cluster.CrossCluster`
        :type cluster_res: array-like
        :param label: label of cluster
        :type label: array-like
        :param sep: defaults to "@"
        :type sep: str, optional
        """        
        # fix result. and sperate genes.
        genes_mixed = cluster_res.index
        genes = pd.Series([i.split(sep)[0] for i in genes_mixed]).unique()
        fixed_res = pd.DataFrame(index=genes, columns=label)
        for i in range(cluster_res.shape[0]):
            group = cluster_res[i]
            gene = genes_mixed[i]
            gene, spe = gene.split(sep)
            if spe in label:
                fixed_res.loc[gene, spe] = group
            else:
                raise ValueError(
                    "Label is not consistent with input.(or sep is wrong)")

        # genes that not in same group.
        matched = [(fixed_res.iloc[i, :] == fixed_res.iloc[i, 0]).all()
                   for i in range(fixed_res.shape[0])]
        fixed_res["matched"] = matched

        self.res = fixed_res

    def _JMatrix(self):
        """calculate J matrix(cross tabulation), showing the relationship between clusters of phenotypes.
        """        
        JM = np.zeros(shape=[self.n_cluster]*len(self.label))
        try:
            for i in range(self.res.shape[0]):
                JM[tuple(self.res.iloc[i, :-1])] += 1
        except Exception as e:
            print(e)
            print("Using index:", tuple(self.res.iloc[i, :-1]))
            print("Using result", self.res.head())
            #print("Using matrix",JM,JM.shape)
            raise
        self.JM = JM

    def _module_genes(self):
        """get genes in each module.And store in attibute :attr:`ClusterResult.module_genes`

        :return: keys : module id | values : gene list
        :rtype: dict
        """        
        module_genes = {}
        for ind in zip(*np.where(self.JM > 0)):
            genes = self.res.index[(
                self.res.iloc[:, 0:-1] == list(ind)).all(axis=1)]
            if (pd.Series(ind) == ind[0]).all():
                continue
            module_genes["M"+".".join(list(map(str, list(ind))))] = genes
        self.module_genes = module_genes
        return module_genes

    def summary(self, fig=True):
        """summary of cluster results,print :attr:`ClusterResult.cluster_properties`, and plot J matrix.

        :param fig: whether print figures or not, defaults to True
        :type fig: bool, optional
        :return: summary of cluster results
        :rtype: figures or None
        """        
        print("# --- Number of genes:")
        print("Same Module Genes:", self.res["matched"].sum())
        print("Different Module Genes:",
              self.res.shape[0] - self.res["matched"].sum())
        print("# --- clustering score:")
        for k, v in self.cluster_properties.items():
            print(k, v)
        if fig:
            f1 = self.Vis_Jmat()
            print("# --- samples' distribution:")
            sf = self.SampleFeature(df_exp=self.df_exp)
            f2 = Fig_SampleFeature(
                sf, labels=self.pheno,
                model=PCA(),metric="euclidean",
                dpi=150)
            return f1, f2

    def Vis_Jmat(self):
        """plot J matrix.(heatmap)

        :return: figure
        :rtype: matplotlib.figure.Figure
        """        
        JM = self.JM
        f1 = plt.figure(dpi=300)
        sns.heatmap(
                data=JM, fmt=".0f",
                cmap="RdPu", square=True,
                annot=True, annot_kws={"fontsize": 8})
        plt.title("Genes' distribute")
        plt.xlabel(self.label[-1].split("|")[-1])
        plt.ylabel(self.label[-2].split("|")[-1])
        return f1

    # Cluster Score Calculation
    def MCScore(self, method=ch_score):
        """calculation subscore:the loss between each mc. the result stored in `self.property["MCScore"]

        :param method: method used to calculate, annother recommend function:`sklearn.metrics.davies_bouldin_score`, defaults to `sklearn.metrics.calinski_harabasz_score`.
        :type method: _type_, optional
        """        
        self.property["MCScore"] = method(self.df_exp, [str(
            self.res.iloc[i, 0:2].values) for i in range(self.res.shape[0])])

    def MCFeature(self, df_exp=None, module_genes=None, model=None):
        """calculation group features.using PCA for it reserving more information than others.

        .. note:: 
            if n_components >=2, MCs' eigenvector can reserve more than one to make explained ratio >=80%.

        :param df_exp: defaults to None, :attr:`ClusterResult.df_exp`
        :type df_exp: pd.DataFrame, optional
        :param module_genes: the mapping of module and genes. defaults to None, result of function :func:`MATTE.analysis.ClusterResult._module_genes`, 
        :type module_genes: dict, optional
        :param model: the model to fit transform data.needs attribute `fit_transform`, defaults to None(`sklearn.decomposition.PCA(n_components=1)`)
        :type model: object, optional
        :return: group_feature: `dict`,keys : module id | values : eigenvector, group_weight: `dict`,keys : module id | values : lambda of model 
        :rtype: tuple
        """        
        if df_exp is None:
            df_exp = self.df_exp
        if module_genes is None:
            module_genes = self._module_genes()

        group_feature = {}
        group_weight = {}
        for mc, genes in module_genes.items():
            pca = PCA(n_components=1) if model is None else model
            feature = pca.fit_transform(
                df_exp.loc[genes, :].T)
            all_explained = 0
            for ind, ratio in enumerate(pca.explained_variance_ratio_):
                all_explained += ratio
                if all_explained >= 0.8:
                    break

            group_feature[mc] = feature[:, 0:ind+1]
            group_weight[mc] = pca.explained_variance_ratio_[:ind+1]

        return group_feature, group_weight

    # Samples' feature and phenotype's signature.

    def SampleFeature(self, df_exp=None, module_genes=None, return_df=True, **kwargs):
        """Calculate Samples' module feature.(average default)

        :param df_exp: defaults to None, :attr:`ClusterResult.df_exp`
        :type df_exp: pd.DataFrame, optional
        :param module_genes: the mapping of module and genes. defaults to None, result of function :func:`MATTE.analysis.ClusterResult._module_genes`, 
        :type module_genes: dict, optional
        :param return_df: decide return. If False, then will store result in :attr:`ClusterResult.samples_feature`, defaults to True
        :type return_df: bool, optional
        :return: depended by the arg:return_df
        :rtype: samples_feature is a `WeightedDataFrame` object, whose index is samples id and columns is module id. Module Weights are calculated by function `MCFeature`.
        """        
        if df_exp is None:
            df_exp = self.df_exp

        def lst_sum(lst):
            ret = []
            for l in lst:
                ret.extend(l)
            return ret

        module_genes = self._module_genes() if module_genes is None else module_genes
        group_feature, group_weight = self.MCFeature(
            df_exp=df_exp, module_genes=module_genes, **kwargs)
        samples_feature = WeightedDataFrame(
            index=df_exp.columns, data=np.hstack(
                tuple(group_feature.values())),
            columns=lst_sum(
                [
                    [i+"_"+str(_)] for i in group_feature.keys()
                    for _ in range(group_feature[i].shape[1])]),
            weight=np.hstack(list(group_weight.values()))
        )

        if return_df:
            return samples_feature
        else:
            self.samples_feature = samples_feature

    def PhenoMCCorr(self, sample_feature, pheno_series, pheno=None):
        """Calculate the correlation between Phenotype and MC's Eigenvector and test the correlation and return p-value

        :param sample_feature: can be calculated by the function :func:`ClusterResult.SampleFeature`
        :type sample_feature: :class:`MATTE.WeightedDataFrame`
        :param pheno_series: specify samples' phenotype
        :type pheno_series: pd.Series
        :param pheno: choose which pheno the calculate, defaults to None (the first one of `pheno_series`)
        :type pheno: any, optional
        :return: `MC_corr`: a `pd.Series` whose index is Module ID, and value is the pearson correlation. `corr_p`: p_value of correlation(Student t test)
        :rtype: tuple
        """        
        def covPvalueStudent(cor, n_samples):
            T = (n_samples - 2)**0.5 * cor/((1-cor**2)**0.5)
            return 2*stats.t.sf(abs(T), n_samples-2)

        if pheno is None:
            pheno = pheno_series.unique()[0]
        MC_corr = pd.Series(index=sample_feature.columns, dtype=float)
        for i in range(sample_feature.shape[1]):
            MC_corr[i] = np.corrcoef(
                sample_feature.iloc[:, i],
                pheno_series[sample_feature.index] == pheno)[0, 1]
        corr_p = pd.Series(index=MC_corr.index, data=covPvalueStudent(
            MC_corr, n_samples=len(MC_corr)))
        return MC_corr, corr_p

    def MCkwtest(self, sample_feature, pheno_series):
        """Making the Kruskal-Wallis test on each MC.
        (if Each module's Eigenvector is different)

        :param sample_feature: can be calculated by the function :func:`MATTE.analysis.ClusterResult.SampleFeature`
        :type sample_feature: :class:`MATTE.WeightedDataFrame`
        :param pheno_series: specify samples' phenotype
        :type pheno_series: pd.Series
        :return: `kws`: a `pd.Series` whose index is Module ID, and value is the p-value.
        :rtype: pd.Series
        """        
        Sig = [
            sample_feature.loc[pheno_series[pheno_series == li].index, :]
            for li in pheno_series.unique()
        ]

        kws = pd.Series(index=sample_feature.columns, data=0.0)
        for i in range(len(kws)):
            kw = stats.kruskal(*[sig.iloc[:, i] for sig in Sig]).pvalue
            kws[i] = kw
        return kws

    def ModuleSNR(self, sample_feature, pheno=None):
        """Calculate the SNR of each module.

        :param sample_feature: can be calculated by the function :func:`MATTE.analysis.ClusterResult.SampleFeature`
        :type sample_feature: :class:`MATTE.WeightedDataFrame`
        :param pheno: the label of samples, defaults to None (set to be :attr:`ClusterResult.pheno`)
        :type pheno: _type_, optional
        :return: SNR, a `pd.Series` whose index is Module ID, and value is the SNR.
        :rtype: pd.Series
        """        
        if pheno is None:
            pheno = self.pheno
        SNR = pd.Series(index=sample_feature.columns, dtype=float)
        assert pheno.unique().size == 2, "The pheno should be binary"
        pos,neg = pheno.unique()
        pos_samples = pheno[pheno==pos].index
        neg_samples = pheno[pheno==neg].index
        for i in sample_feature.columns:
            mean_pos = np.mean(sample_feature.loc[pos_samples, i])
            mean_neg = np.mean(sample_feature.loc[neg_samples, i])
            sd_pos = np.std(sample_feature.loc[pos_samples, i])
            sd_neg = np.std(sample_feature.loc[neg_samples, i])
            SNR[i] = np.abs(mean_pos-mean_neg)/ (sd_neg+sd_pos+1e-10)
        return SNR.sort_values(ascending=False)

# Showing

def Fig_SampleFeature(
        sample_feature, labels, color=None,
        model=None, weighted_distcance=False, metric='euclidean', **fig_args):
    """showing the fig of samples reprented by MCs.

    :param sample_feature: index : samples' id | columns : modules' id . the result of :func:`ClusterResult.SampleFeature` or just using some genes to explain a sample.
    :type sample_feature: `WeightedDataFrame` or `pd.DataFrame` 
    :param labels: index : samples' id | columns : phenotype
    :type labels: pd.Series
    :param color: the second label used, and in the form of color. the first one will be in the form of shape., defaults to None
    :type color: array like, optional
    :param model: the model used to present samples in the two dimensions., defaults to None (`sklearn.manifold.PCA(n_component=2)`)
    :type model: object, optional
    :param weighted_distcance: whether to used weighted distance. if sample_feature is `pd.DataFrame`, it should be set as False., defaults to False
    :type weighted_distcance: bool, optional
    :param metric: used to calculate the distance., defaults to 'euclidean'
    :type metric: str, optional
    :return: the figure
    :rtype: `plt.figure`
    """
    model = PCA(n_components=2) if model is None else model
    if weighted_distcance:
        psf = abs(sample_feature.weight_distance(metric=metric))
        psf = model.fit_transform(psf)
    else:
        psf = model.fit_transform(sample_feature)

    if not hasattr(model, "fit_transform"):
        raise TypeError("model has no method named fit_transform.")

    Samples = sample_feature.index
    labels = labels[Samples]
    if color is None:
        fig = plt.figure(**fig_args)
        for i, l in enumerate(labels.unique()):
            plt.scatter(
                x=psf[labels == l][:, 0], y=psf[labels == l][:, 1],
                alpha=.8, label=l, color=matplotlib.cm.Paired(i/len(labels.unique())))
        plt.legend(loc='best', shadow=False, scatterpoints=1)
    else:
        color = color[Samples]
        fig = plt.figure(**fig_args)
        markers = ["o", "x", ",", "v", "^", "<", ">", "*", "+", ".",
                   "1", "2", "3", "4", "8", "s", "p", "h", "d", "D", "|", "_"]
        for i, l in enumerate(labels.unique()):
            plt.scatter(
                x=psf[labels == l][:, 0], y=psf[labels == l][:, 1],
                alpha=.8, label=l, c=color[labels == l], marker=markers[i])
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.colorbar()

    return fig


def FunctionEnrich(annote_file, gene_set, category_seperate_cal=True):
    """perform GO enrichment

    :param annote_file: Function annotation file dir or dataframe
    .. note:: tab-seperate file with columns:["Term_ID","GeneID","Term","Category"];File can be downloaded from https://ftp.ncbi.nih.gov/gene/DATA/
    :type annote_file: `str` or `pd.DataFrame`
    :param gene_set: gene set to analysis
    :type gene_set: array-like
    :param category_seperate_cal: wheather to calculate fdr in each category or mixed., defaults to True
    :type category_seperate_cal: bool, optional

    :return: items,catogory,enriched items number,backgroud item number,p value,fdr
    :rtype: pd.DataFrame
    """    
    if type(annote_file) == str:
        annot = pd.read_table(annote_file, index_col="Term_ID")
    elif type(annote_file) == pd.DataFrame:
        annot = deepcopy(annote_file)
    assert set(["GeneID", "Term", "Category"]) <= set(annote_file.columns)
    annot.index = annote_file["Term_ID"]
    annot["GeneID"] = pd.Series(annot["GeneID"], dtype=type(gene_set[0]))
    gene_set = set(gene_set) & set(annote_file["GeneID"])
    gene_set = pd.Series(list(gene_set), dtype=object)
    try:
        assert len(gene_set) != 0
    except:
        raise ValueError("No genes in the annotation file.")

    all_items = annot[["Term", "Category"]].drop_duplicates()
    enriched = pd.Series(index=all_items.index, data=0, dtype=int)
    backgroud = pd.Series(index=all_items.index, data=0, dtype=int)
    p_value = pd.Series(index=all_items.index, data=0.0, dtype=float)
    n_background = annot["GeneID"].unique().size
    n_set = len(gene_set)
    term_genes = {}
    for i in trange(all_items.shape[0]):
        term_id = all_items.index[i]
        all_gene_item_related = pd.Series(annot.loc[term_id, "GeneID"])
        gene_list = all_gene_item_related.unique()
        gene_list_target = pd.Series(
            list(set(gene_list) & set(gene_set)), dtype=object)

        B = gene_list.size
        b = gene_list_target.size

        term_genes[term_id] = gene_list_target
        enriched[term_id] = b
        backgroud[term_id] = B

        if b != 0:
            p_value[term_id] = stats.hypergeom.sf(b-1, n_background, B, n_set)
        else:
            p_value[term_id] = 1.0

    all_items["n_enriched"] = enriched
    all_items["n_backgroud"] = backgroud
    all_items["p_value"] = p_value
    all_items["fdr"] = pd.NA

    if category_seperate_cal:
        for cat in all_items["Category"].unique():
            idx = all_items[all_items["Category"] == cat].index
            all_items.loc[idx, "fdr"] = fdrcorrection(
                all_items.loc[idx, "p_value"].values)[1]
    else:
        all_items.loc[:, "fdr"] = fdrcorrection(
            all_items.loc[:, "p_value"].values)[1]

    all_items = all_items.sort_values(by=["fdr", "p_value", "n_enriched"])
    all_items['gene_ratio'] = all_items['n_enriched'] / len(gene_set)
    return all_items, term_genes


def Fig_Fuction(df_enrich, color_columns, cmap=cm.Set1, width_height_ratio=2, **figargs):
    """figuring the function enrichment reuslt.
    .. note:: filtering the result first!

    :param df_enrich: the result of :func:`FunctionEnrich`
    :type df_enrich: `pd.DataFrame`
    :param color_columns: used to shown in color, 'fdr' or 'p_value'
    :type color_columns: `str`
    :param cmap: chosen from `matplotlib.cm`, defaults to cm.Set1
    :type cmap: object, optional
    :param width_height_ratio: defaults to 2
    :type width_height_ratio: int, optional
    :return: the figure
    :rtype: plt.figure
    """
    lable_max = int(np.log2(df_enrich[color_columns].min()))
    fig = plt.figure(constrained_layout=True, **figargs)
    gs = fig.add_gridspec(
        df_enrich.shape[0], df_enrich.shape[0]*width_height_ratio)
    used_gride = 0
    axes = []
    for n, cat in enumerate(df_enrich["Category"].unique()):
        ctarget = df_enrich[df_enrich["Category"] == cat]
        ax = fig.add_subplot(gs[used_gride:used_gride+ctarget.shape[0], :])
        used_gride += ctarget.shape[0]
        ax.barh(
            width=ctarget["gene_ratio"],
            y=range(ctarget.shape[0]),
            color=[cm.coolwarm(np.log2(i)/lable_max) for i in ctarget[color_columns]])
        ax.set_yticks(range(ctarget.shape[0]))
        ax.set_yticklabels(ctarget.index.values.tolist())
        ax.tick_params("y", labelcolor=cmap(n))
        ax.text(
            1.02, 0, cat,
            transform=ax.transAxes, rotation=90, color=cmap(n), verticalalignment='bottom',
            weight='semibold')
        axes.append(ax)

    ## setting all axes in the figure share the x axis
    xlim_max = int(df_enrich['gene_ratio'].max()*10+1)/10
    for ax in axes:
        ax.set_xlim(0, xlim_max)
    
    for ax in axes[:-1]:
        ax.set_xticklabels([])
        ax.set_xticks([])
    plt.xlabel("Gene Ratio")

    cbar_ax = fig.add_axes([1.01, 0.12, 0.03, 0.87])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.coolwarm), cax=cbar_ax)
    cbar.set_ticks(ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels([0, -lable_max/4, -lable_max /
                         2, -3*lable_max/4, -lable_max])
    cbar.set_label("-log(%s)" % color_columns)

    return fig
