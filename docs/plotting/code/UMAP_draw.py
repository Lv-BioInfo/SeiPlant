import torch
import numpy as np
# import utils
# import data
# import preprocess
import pandas as pd
import os
import nmslib
import leidenalg
import igraph as ig
from scipy.sparse import csgraph, csr_matrix
import umap.umap_ as umap
from openTSNE import *
from openTSNE import affinity
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def main(base_path='../ath/prediction_results/20250214_184449/',modified_number = 0):

    if modified_number != 0:
        appendix_name = f'_modified_{modified_number}'
    else:
        appendix_name = ''

    file_name = 'new_data_predictions'+ appendix_name + '.npy'
    prediction_results_path = os.path.join(base_path, file_name)
    data = np.load(prediction_results_path)
    data.shape


    def knn_graph(X, n_neighbors=15, space='l2', num_threads=8, params={'post': 2}):
        """    space: string
                The metric/non-metric distance functions to use to compute distances.
                see https://github.com/nmslib/nmslib/blob/master/manual/manual.pdf
                    * bit_hamming
                    * l1
                    * l1_sparse
                    * l2 -> Euclidean distance
                    * l2_sparse
                    * linf
                    * linf_sparse
                    * lp:p=...
                    * lp_sparse:p=...
                    * angulardist, angulardist sparse, angulardist sparse fast
                    * jsmetrslow, jsmetrfast, jsmetrfastapprox
                    * leven
                    * sqfd minus func, sqfd heuristic func:alpha=..., sqfd gaussian func:alpha=...
                    * jsdivslow, jsdivfast, jsdivfastapprox
                    * cosinesimil, cosinesimil sparse, cosinesimil sparse
                    * normleven
                    * kldivfast
                    * kldivgenslow, kldivgenfast, kldivgenfastrq
                    * itakurasaitoslow, itakurasaitofast, itakurasaitofastrq
                    * negdotprod_sparse
                    * querynorm_negdotprod_sparse
                    * renyi_diverg
                    * ab_diverg
        """
        index = nmslib.init(method='hnsw', space=space)
        index.addDataPointBatch(X)
        index.createIndex(params, print_progress=False)
        neighbours = index.knnQueryBatch(X, k=n_neighbors, num_threads=num_threads)
        ind = np.vstack([i for i,d in neighbours])
        sind=np.repeat(np.arange(ind.shape[0]), ind.shape[1])
        tind=ind.flatten()
        g = csr_matrix((np.ones(ind.shape[0]*ind.shape[1]),(sind,tind)),(ind.shape[0],ind.shape[0]))
        return g

    #Standardize the prediction results
    data = data/data.std(axis=0)[None,:]

    g = knn_graph(data, n_neighbors=20, num_threads=10)
    sources, targets = g.nonzero()
    edgelist = zip(sources.tolist(), targets.tolist())
    G = ig.Graph(g.shape[0],list(edgelist))

    partitionl = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition )

    #Leiden clustering results saved
    leiden_path = os.path.join(base_path, 'new_data_predictions'+appendix_name+'.leiden30.npy')
    np.save(leiden_path , partitionl.membership)




    # Performing Nearest Neighbor Search on Large Datasets Using HNSW
    index = nmslib.init(method='hnsw', space='l2', space_params={'post': 2})
    index.addDataPointBatch(data)
    index.createIndex({'post': 2}, print_progress=False)

    # Perform k-nearest neighbor search using the created index. For each data point in data, find its top 25 nearest neighbors.
    k_num = 25
    neighbors = index.knnQueryBatch(data, k=k_num, num_threads=10)

    # ind = np.vstack([i[:k_num] for i, d in neighbors])
    # dist = np.vstack([d[:k_num] for i, d in neighbors])
    padded_neighbors = []
    for i, d in neighbors:
        if len(i) < k_num:
            i_padded = np.pad(i, (0, k_num - len(i)), mode='constant', constant_values=-1)
            d_padded = np.pad(d, (0, k_num - len(d)), mode='constant', constant_values=np.inf)
        else:
            i_padded = i[:k_num]
            d_padded = d[:k_num]
        padded_neighbors.append((i_padded, d_padded))

    ind = np.vstack([i for i, d in padded_neighbors])
    dist = np.vstack([d for i, d in padded_neighbors])
    #  The first part represents a weighted similarity metric between each data point and its nearest neighbors. The second part calculates the variance (or dispersion) of the logarithmic distances to each data point's 19 nearest neighbors. It measures the overall variability in the distances to each data point's neighbors.
    # The ratio of these two metrics measures the similarity versus dispersion ratio between each data point and its neighbors. A high ratio may indicate clear clustering structures in the feature space, while a low ratio may suggest loose or inconsistent neighbor relationships around the data point.

    beta= np.sum((np.log(dist[:,1:])-np.mean(np.log(dist[:,1:]),axis=1)[:,None])*(np.log(np.arange(1, k_num)) - np.mean(np.log(np.arange(1,k_num))))[None,:],axis=1)/np.sum((np.log(dist[:,1:])-np.mean(np.log(dist[:,1:]),axis=1)[:,None])**2,axis=1)

    # Filter useful row IDs, mark them as True, and store them in allinds.
    allinds = (dist[:,1]!=0)* (beta < 180) 
    logp = np.log(dist[allinds,1])*beta[allinds]

    np.random.seed(0)
    s =  logp + np.random.gumbel(size=logp.shape)


    #  Select the first 500,000 data points for dimensionality reduction.
    selectinds = np.argsort(-s)[:500000]
    vis_sub = umap.UMAP(min_dist=0.,metric='euclidean').fit_transform(data[allinds,:][selectinds,:])

    init_path = os.path.join(base_path, 'univar.subsample.beta.1000000.euclidean'+appendix_name+'.npy')
    inds_path = os.path.join(base_path, 'univar.subsample.beta.1000000.euclidean.inds'+appendix_name+'.npy')

    np.save(init_path, vis_sub)
    np.save(inds_path, np.where(allinds)[0][selectinds])

    selectedInds_path = os.path.join(base_path, 'selectedInds'+appendix_name+'.npy')
    np.save(selectedInds_path, selectinds)



    # Set parameters
    N = 500000
    perplexity = 200
    exaggeration = 1.5
    ee = 3

    # Loading data
    init = np.load(init_path)
    inds = np.load(inds_path)

    # Set the random seed
    np.random.seed(0)

    data = data.astype(np.float32)

    # Calculate the similarity of samples
    sample_affinities = affinity.PerplexityBasedNN(
        np.vstack(data[inds,:]),
        perplexity=perplexity,
        method="approx",
        n_jobs=40,
        random_state=0,
    )

    # Perform the first TSNE embedding optimization
    sample_embedding = TSNEEmbedding(
        init,
        sample_affinities,
        negative_gradient_method="fft",
        n_jobs=40,
    )
    sample_embedding.optimize(n_iter=250, exaggeration=ee, momentum=0.5, inplace=True, learning_rate=1000)
    sample_embedding.optimize(n_iter=750, exaggeration=exaggeration, momentum=0.8, inplace=True, learning_rate=1000)

    # Results can be saved or further processed.
    tsne_result_path = os.path.join(base_path, 'tsne_result'+appendix_name+'.npy')
    np.save(tsne_result_path, sample_embedding)


    c = np.load(leiden_path)
    embedding = np.load(tsne_result_path)
    collections.Counter(c)

    # Initialize two lists to store the drawing data.
    cluster_embedding_plot = []

    # Define the maximum quantity limit
    cluster_count = {}
    max_num = 25000

    # Assume a numpy array containing species information, which represents the actual species labels.

    # Iterate through each data point and plot it based on its cluster.
    for i, cluster in enumerate(c[inds]):
        if cluster in cluster_count:
            cluster_count[cluster] += 1
        else:
            cluster_count[cluster] = 1
        
        # If the number of data points in a cluster exceeds the maximum limit, skip it.
        if cluster_count[cluster] > max_num:
            continue
        
        # Add data to the list
        cluster_embedding_plot.append({'x': embedding[i, 0], 'y': embedding[i, 1], 'cluster': cluster})

    # Convert the results into a DataFrame containing x, y, and cluster information.
    cluster_plot_data = pd.DataFrame(cluster_embedding_plot)

    # Output or process the DataFrame (e.g., for plotting)
    print(cluster_plot_data.head()) 

    # Optional: Save as a CSV file for easy analysis later.
    cluster_plot_data.to_csv(base_path+'cluster_embedding_plot'+appendix_name+'.csv', index=False)



    # Set the theme for Seaborn
    sns.set_theme(style="whitegrid")

    # Custom Palette (colors can be selected as needed)
    custom_palette = ["#ABD1BC", "#E3BBED", "#CCCC99", 
                    "#BED0F9", "#FCB6A5", "#BADAB5", "#72B063",
                    "#E29135","#94C6CD","#4A5F7E",
                    "#925EB0","#7E99F4","#CC7C71",
                    "#8D2F25","#4E1947","#CB9475",
                    "#8CBF87","#3E608D","#909291",
                    "#B7B7EB","#9D9EA3","#EAB883",
                    "#9BBBE0","#F09BA0","#E6B745",
                    "#DCA7EB","#E3E3E1","#EAE935",
                    "#FDEBAA", "#EDC3A5", "#DBE4FB"


    ]

    # Create Drawing
    plt.figure(figsize=(6, 6))
    scatter = sns.scatterplot(
        data=cluster_plot_data,
        x='x', 
        y='y', 
        hue='cluster',  
        palette=custom_palette,  
        alpha=1,  
        s=1  
    )

    norm = mpl.colors.Normalize(vmin=cluster_plot_data['x'].min(), vmax=cluster_plot_data['x'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    scatter.set_title(f'Cluster Embedding Plot', fontsize=14)
    scatter.set_xlabel('TSNE1', fontsize=12)
    scatter.set_ylabel('TSNE2', fontsize=12)

    plt.legend(
        title="Cluster", 
        loc="center left", 
        fontsize=10, 
        bbox_to_anchor=(1.0, 0.5), 
        markerscale=2, 
        ncol=2  
    )


    figure_path = os.path.join(base_path,'figure','Cluster_Embedding'+appendix_name+'.png')

    plt.savefig(figure_path, dpi=300, bbox_inches='tight')  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--base_path", type=str, default='../ath/prediction_results/20250214_184449/', help="base path (default: '../ath/prediction_results/20250214_184449/')")
    parser.add_argument("--modified_number", type=int, default=0, help="modified file number (default:0 )")
    args = parser.parse_args()

    main(base_path, modified_number)