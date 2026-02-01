import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import SymLogNorm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import os

def default(species_name,base_path='../ath/prediction_results/20250214_184449/',modified_number = 0):

    if modified_number != 0:
        appendix_name = f'_modified_{modified_number}'
    else:
        appendix_name = ''


    droplist = []

    histone_enrich = pd.read_csv(base_path+'/histone'+appendix_name+'_enrichment.tsv', sep='\t', header=None)
    tfbs_enrich = pd.read_csv(base_path+'/TFBS'+appendix_name+'_enrichment.tsv', sep='\t', header=None)
    cluster_list_path = os.path.join(base_path,f'cluster_bed{appendix_name}',f'cluster_list{appendix_name}.txt')

    cluster_names = [rec.strip().split(' ')[-1] for rec in open(cluster_list_path)]

    row_names = [ rec.strip().replace(f'{species_name}_', '', 1)  for rec in open(base_path+'/histone'+appendix_name+'_list.txt').readlines() ]
    histone_enrich.index = row_names
    histone_tag = row_names

    row_names = [ rec.strip().replace(f'{species_name}_', '', 1)  for rec in open(base_path+'/TFBS'+appendix_name+'_list.txt').readlines() ]
    tfbs_enrich.index = row_names
    tfbs_tag = row_names

    histone_enrich_log = histone_enrich.replace(np.inf, 1e7).fillna(1e-7).T
    histone_enrich_log.index = cluster_names
    histone_enrich_log = histone_enrich_log.drop(droplist)


    tfbs_enrich_log = tfbs_enrich.replace(np.inf, 1e7).fillna(1e-7).T
    tfbs_enrich_log.index = cluster_names
    tfbs_enrich_log = tfbs_enrich_log.drop(droplist)



    VMAX=16

    sns.set(font_scale=1)

    histone_cm = sns.clustermap(histone_enrich_log , mask=histone_enrich_log ==1e7, figsize=(histone_enrich_log.shape[1]/2, histone_enrich_log.shape[0]/2),
                annot=False, cmap='RdBu_r', cbar=True,
                norm=SymLogNorm(linthresh=1, linscale=1), vmin=-VMAX, vmax=VMAX, col_cluster=True,row_cluster=True)
    histone_order = [histone_cm.dendrogram_row.reordered_ind,histone_cm.dendrogram_col.reordered_ind]

    tfbs_cm = sns.clustermap(tfbs_enrich_log, mask=tfbs_enrich_log == 1e7, 
                            figsize=(tfbs_enrich_log.shape[1]/2, tfbs_enrich_log.shape[0]/2),
                            annot=False, cmap='RdBu_r', cbar=True,
                            norm=SymLogNorm(linthresh=1, linscale=1), 
                            vmin=-VMAX, vmax=VMAX, col_cluster=True, row_cluster=False)
    # Retrieve the column order of tfbs_enrich_log
    tfbs_order = tfbs_cm.dendrogram_col.reordered_ind

    histone_row_order = histone_cm.dendrogram_row.reordered_ind
    histone_col_order = histone_cm.dendrogram_col.reordered_ind

    # Retrieve the column order for tfbs 
    tfbs_order = tfbs_cm.dendrogram_col.reordered_ind

    # Reorder histone_enrich_log
    histone_reorder = histone_enrich_log.iloc[histone_row_order, histone_col_order]

    # Rearrange tfbs_enrich_log using histone row order and TFB column order
    tfbs_reorder = tfbs_enrich_log.iloc[histone_row_order, tfbs_order]

    sns.set(font_scale=1)
    sns.set_style("white")

    VMAX = 5

    fig = plt.figure(figsize=(40, 20))

    # Set up the GridSpec layout to add a column for the color bar.
    gs = gridspec.GridSpec(1, 3, width_ratios=[histone_reorder.shape[1], tfbs_reorder.shape[1], 0.1], wspace=0.1)

    # Set up three subgraphs
    ax0 = plt.subplot(gs[0, 0])
    ax0.set_title('Histone Marks',fontsize=16, fontweight='bold')

    ax1 = plt.subplot(gs[0, 1])
    ax1.set_title('TFBS',fontsize=16, fontweight='bold')

    # Create a heatmap
    ax = sns.heatmap(histone_reorder, mask=histone_reorder==1e7,
                    cmap='RdBu_r', norm=SymLogNorm(linthresh=1, linscale=1), 
                    vmin=-VMAX, vmax=VMAX, cbar=False, ax=ax0)
    ax.set_facecolor("gray")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha='right',fontsize=20)
    ax.set_yticklabels(ax0.get_yticklabels(), ha='right',fontsize=15)

    ax = sns.heatmap(tfbs_reorder, mask=tfbs_reorder==1e7,
                    cmap='RdBu_r', yticklabels=0, 
                    norm=SymLogNorm(linthresh=1, linscale=1), 
                    vmin=-VMAX, vmax=VMAX, cbar=False, ax=ax1)
    ax.set_facecolor("gray")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha='right',fontsize=20)

    # Create a hidden heatmap to generate a color bar
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])  # Set the position of the color bar [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=SymLogNorm(linthresh=1, linscale=1, vmin=-VMAX, vmax=VMAX))
    sm.set_array([]) 
    fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label="Enrichment Score")
    
    # 保存图像
    figure_path = os.path.join(base_path,'figure',f'default_heatmap_{species_name}{appendix_name}HFT.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')  # Add `bbox_inches=‘tight’` to ensure the legend is not cropped.
    plt.show()
    print(histone_reorder.index)
    print(f'The heatmap is saved in {figure_path}!')

    return histone_reorder , tfbs_reorder , histone_tag , tfbs_tag

def reorder(histone_reorder,tfbs_reorder,cluster_order,histone_cols,tfbs_cols):

    histone_reorder = histone_reorder.loc[cluster_order, :]
    tfbs_reorder = tfbs_reorder.loc[cluster_order, :]

    histone_reorder = histone_reorder[histone_cols]
    tfbs_reorder = tfbs_reorder[tfbs_cols]

    return histone_reorder,tfbs_reorder


def draw(species_name,histone_reorder,tfbs_reorder,base_path='../ath/prediction_results/20250214_184449/',modified_number = 0,times=0,figure_size_1=30,figure_size_2=12):
    if modified_number != 0:
        appendix_name = f'_modified_{modified_number}'
    else:
        appendix_name = ''
    sns.set(font_scale=1)
    sns.set_style("white")

    VMAX = 5

    fig = plt.figure(figsize=(figure_size_1, figure_size_2))

    gs = gridspec.GridSpec(1, 3, width_ratios=[histone_reorder.shape[1], tfbs_reorder.shape[1], 0.1], wspace=0.1)

    ax0 = plt.subplot(gs[0, 0])
    ax0.set_title('Histone Marks')

    ax1 = plt.subplot(gs[0, 1])
    ax1.set_title('TFBS')

    ax = sns.heatmap(histone_reorder, mask=histone_reorder == 1e7,
                    cmap='RdBu_r', norm=SymLogNorm(linthresh=1, linscale=1), 
                    vmin=-VMAX, vmax=VMAX, cbar=False, ax=ax0)
    ax.set_facecolor("gray")

    ax = sns.heatmap(tfbs_reorder, mask=tfbs_reorder == 1e7,
                    cmap='RdBu_r', yticklabels=0, 
                    norm=SymLogNorm(linthresh=1, linscale=1), 
                    vmin=-VMAX, vmax=VMAX, cbar=False, ax=ax1)
    ax.set_facecolor("gray")

    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])  
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=SymLogNorm(linthresh=1, linscale=1, vmin=-VMAX, vmax=VMAX))
    sm.set_array([])  
    fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label="Enrichment Score")
    
    figure_path = os.path.join(base_path,'figure',f'{times}_heatmap_{species_name}{appendix_name}HFT.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')  
    plt.show()
    print(f'The heatmap is saved in {figure_path}!')