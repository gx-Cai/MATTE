import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np


if __name__ =='__main__':
    results = pd.read_csv('testresults.csv',index_col=0)
    results.dropna(inplace=True)
    fig_id = 3
    ####  FIG 1: BUBBLE PLOT-comprehensive view
    if fig_id == 1:
        results = results.groupby(by=['fea','k'])[['ari','f1','labled']].mean().reset_index()
        results.index = [f"{results.loc[i,'fea']}_{results.loc[i,'k']}" for i in range(results.shape[0])]

        results = results.groupby(by='fea').apply(
            lambda x: x.iloc[
                (x['ari']**2+x['f1']*x['labled']).argmax(),:]
        )

        feas = results.fea.unique()
        cmap = matplotlib.cm.get_cmap('Paired')
        feas_color = {f:cmap(i) for i,f in enumerate(feas)}

        sns.set_theme(style='whitegrid')
        sns.set_style(style={'grid.color': '.95',})
        f,ax = plt.subplots(dpi=300)
        point_base_size = 300
        for i in range(results.shape[0]):
            color = feas_color[results.iloc[i,0]]
            ari = results.iloc[i,2]
            f1 = results.iloc[i,3]
            labeled = results.iloc[i,-1]

            plt.scatter(
                x = ari,
                y=f1,
                s = point_base_size*labeled,
                color=color,
                alpha=0.75,
            )
        plt.xlabel('ARI')
        plt.ylabel("F1 score")

        handles = [
            plt.Line2D(
                [], [], label=label, 
                lw=0, # there's no line added, just the marker
                marker="o", # circle marker
                markersize=8, 
                markerfacecolor=color, # marker fill color
                markeredgewidth=0,
            )
            for label, color in feas_color.items()
        ]
        legend = f.legend(
            handles=handles,
            bbox_to_anchor=[0.32, 0.925], # Located in the top-mid of the figure.
            fontsize=6,
            handletextpad=0.6, # Space between text and marker/line
            handlelength=1.4, 
            columnspacing=1.4,
            loc="center left", 
            ncol=len(feas_color),
            frameon=False
        )

        handles2 = [
            plt.Line2D(
                [], [], label=label, 
                lw=0, # there's no line added, just the marker
                marker="o", # circle marker
                markersize=8*size, 
                markerfacecolor="#44474c", # marker fill color
                markeredgewidth=0,
            )
            for size,label in zip(np.linspace(0.3,1,4),[""]*3+["labeled ratio"])
        ]

        legend2 = f.legend(
            handles=handles2,
            bbox_to_anchor=[0.32, 0.925], # Located in the top-mid of the figure.
            fontsize=6,
            handletextpad=0.6, # Space between text and marker/line
            handlelength=1.4, 
            columnspacing=-0.3,
            loc="center right", 
            ncol=5,
            frameon=False
        )


        f.savefig('Bubble_results.pdf',)

    ####  FIG 2: BAR PLOT-for supervised learning
    elif fig_id == 2:
        results = results.groupby(by=['fea','k'])[['ari','f1','labled','acc']].mean().reset_index()
        results.index = [f"{results.loc[i,'fea']}_{results.loc[i,'k']}" for i in range(results.shape[0])]

        results = results.groupby(by='fea').apply(
            lambda x: x.iloc[
                (x['ari']**2+x['f1']*x['labled']).argmax(),:]
        )

        results = results[['labled','acc']]
        results.loc[:,'unlabeled'] = 1 - results.labled
        results.loc[:,'error'] = results.labled*(1-results.acc)
        results.loc[:,'accratio'] = results.labled*results.acc

        assert (results.unlabeled + results.error + results.accratio <= 1.001).all()

        sns.set_theme(style='white')
        f,ax = plt.subplots(dpi=300)
        cm = matplotlib.cm.get_cmap('Paired')
        colors = ['#80a7d8','#e5848a','#cdd8e5']
        accumued = [0] * results.shape[0]
        legend_name = {'unlabeled':'unlabeled',
                        'error':'wrong pred',
                        'accratio':'right pred'}
        for i,bar_label in enumerate(['accratio','error','unlabeled']):
            plt.bar(
                x = list(range(1,results.shape[0]+1)),
                height = results[bar_label]*100,
                bottom = accumued,
                color=colors[i],
                edgecolor='white',
                linewidth=0.5,
                label=legend_name[bar_label],
            )
            accumued = [a+b for a,b in zip(accumued,results[bar_label]*100)]
        plt.legend(loc='upper left', bbox_to_anchor=(1,1.03), ncol=1)
        plt.xticks(list(range(1,results.shape[0]+1)),results.index)
        plt.xlabel('Feature Selection Methods')
        plt.ylabel('Rate (%)')
        plt.title(f'Different Feature Selection Methods')

        f.savefig(f'Bar_results_bar.pdf',bbox_inches='tight')
    
    #### FIG 3: Line Plot-for unsupervised learning
    elif fig_id == 3:
        
        f = sns.relplot(
            x='k',y='ari',
            data=results,ci='sd',hue='fea',
            kind='line',aspect=1.5
            )
        f.set_xlabels('Number of Genes Used')
        f.set_ylabels('ARI')
        f.savefig('Line_results.pdf',bbox_inches='tight',dpi=300)

    #### FIG 4: umap plot
    elif fig_id == 4:
        from anndata import AnnData
        import scanpy as sc
        import os
        from scRNA_Evaluation import read_data
        import matplotlib.cm as cm

        test_results = results.copy()
        vis_res = test_results[test_results['testdata'] == '10Xv2']
        vis_res['labled'] = vis_res['labled'].astype(float)
        vis_res = vis_res.groupby(by='fea').apply(lambda x: x.iloc[x['labled'].argmax(),:]).dropna()

        training_data_dir = 'data/PbmcBench/10Xv2'
        data,label = read_data(training_data_dir)

        raw_data = pd.read_hdf(os.path.join(training_data_dir,'unprocessed.h5'),key='data')
        raw_data = raw_data.loc[label.index,:]
        ann = AnnData(raw_data)
        sc.pp.pca(ann)
        sc.pp.neighbors(ann)
        sc.tl.umap(ann)

        decomp_10xv2 = ann.obsm['X_umap']
        plate = cm.rainbow
        #sns.set_theme(style='white')#,rc = {'axes.edgecolor':'#000000'}
        cell_colors = {l:plate((i+1)/6) for i,l in enumerate(label.unique())}
        cell_colors.update({'UnKnown':'#c1cad8'})

        results = test_results.dropna()
        results = results.groupby(by=['fea','k'])[['ari','f1','labled']].mean().reset_index()
        results.index = [f"{results.loc[i,'fea']}_{results.loc[i,'k']}" for i in range(results.shape[0])]

        results = results.groupby(by='fea').apply(
            lambda x: x.iloc[
                (x['ari']**2+x['f1']*x['labled']).argmax(),:]
        )

        feas = results.fea.unique()
        cmap = cm.get_cmap('tab20')
        feas_color = {f:cmap(i) for i,f in enumerate(feas)}

        def tsne_vis(labels,ax_edge_color,ax):
            label = np.unique(labels)
            label = label[::-1]
            for i,l in enumerate(label):
                sample_idx = np.where((labels == l) == True)[0]
                vis_data = decomp_10xv2[sample_idx,:]
                ax.scatter(
                    x=vis_data[:,0],y=vis_data[:,1],
                    alpha=0.3,
                    color=cell_colors[l],label = l,
                    s=1
                    
                )
                ax.set_xticks([])
                ax.set_yticks([])

                for direct in ['bottom','top','right','left']:
                    ax.spines[direct].set_color(ax_edge_color)
                    ax.spines[direct].set_lw(1.5)
        #     plt.legend(
        #         bbox_to_anchor=[1.01, 0.75],edgecolor='#ffffff'
        #     )
            return ax

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)

        fig = plt.figure(dpi=300, constrained_layout=True,figsize=(10,6))
        spec = fig.add_gridspec(ncols=5, nrows=3)
        ax_raw = fig.add_subplot(spec[0:2, 0:2])
        axes = [fig.add_subplot(spec[location]) for location in [(0,2),(0,3),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(0,4),(1,4),(2,4)]]
        tsne_vis(label,'.85',ax_raw)
        ax_raw.text(0.05,0.87,'RAW',transform = ax_raw.transAxes,color='.85',size='large')

        for i in range(vis_res.shape[0]):
            feas_type = vis_res.iloc[i,1]
            k = vis_res.iloc[i,2]
            test_data_dir = vis_res.iloc[i,0]
            key = f"SUP_{feas_type}_{int(k)}_{test_data_dir}"
            predi = pd.read_hdf(os.path.join(training_data_dir,'preds.h5'),key=key)
            tsne_vis(predi,feas_color[feas_type],axes[i])
            axes[i].text(0.05,0.87,feas_type.upper(),
                        transform = axes[i].transAxes,color=feas_color[feas_type],
                        size='medium')
        handles = [
            plt.Line2D(
                [], [], label=cell.replace('.',' '), 
                lw=0, # there's no line added, just the marker
                marker="o", # circle marker
                markersize=8, 
                markerfacecolor=color, # marker fill color
                markeredgewidth=0,)
            for cell,color in cell_colors.items()
        ]
        legend = fig.legend(
            handles=handles,
            bbox_to_anchor=[0.0, 1.05], # Located in the top-mid of the figure.
            fontsize=12,
            handletextpad=0.6, # Space between text and marker/line
            handlelength=1.4, 
            columnspacing=1.4,
            loc="center left", 
            ncol=len(feas_color)//2,
            frameon=False
        )
        # fig.savefig('./results/merged.pdf',bbox_inches='tight')
        # fig.savefig('./results/merged.tiff',bbox_inches='tight')