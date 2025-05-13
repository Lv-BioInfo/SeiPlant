from scipy.sparse import csr_matrix
# import leidenalg
import igraph as ig
import leidenalg
import pandas as pd
import nmslib
import numpy as np
from openTSNE import TSNEEmbedding, affinity
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
from datetime import datetime
import os

def plot_auroc_auprc(total_labels, total_outputs, save_path=None, picture=False):
    """
    绘制 AUROC 和 AUPRC，并保存图像
    :param total_labels: 真实标签 (1D NumPy 数组)
    :param total_outputs: 预测值 (1D NumPy 数组，未经过 Sigmoid)
    :param save_path: 可选，保存文件的路径
    """
    # **计算 AUROC**
    fpr, tpr, _ = roc_curve(total_labels, total_outputs)  # **所有类别合并计算 ROC**
    auroc = auc(fpr, tpr)  # **计算 AUROC**
    print(f"AUROC: {auroc:.4f}")

    # **计算 AUPRC**
    precision, recall, _ = precision_recall_curve(total_labels, total_outputs)
    auprc = average_precision_score(total_labels, total_outputs)  # **计算 AUPRC**
    print(f"AUPRC: {auprc:.4f}")

    if picture:
        # **绘制 AUROC 和 AUPRC**
        plt.figure(figsize=(8, 6))

        # **存储绘图对象**
        roc_plot, = plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUROC = {auroc:.3f})')
        pr_plot, = plt.plot(recall, precision, color='blue', lw=2, linestyle='--', label=f'PR curve (AUPRC = {auprc:.3f})')
        random_plot, = plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')  # **随机分类器对角线**

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR) / Recall')
        plt.ylabel('True Positive Rate (TPR) / Precision')
        plt.title('ROC & Precision-Recall Curve')

        # **手动调整 legend 顺序**
        plt.legend(handles=[roc_plot, pr_plot, random_plot], loc='lower right')

        plt.grid(True)

        # **保存或显示图像**
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"AUROC & AUPRC plot saved as {save_path}")
        else:
            plt.show()

        plt.close()

    return auroc, auprc


def load_data(file_path, pca_threshold=None, use_all=False):
    """
    加载数据并进行 PCA 降维和标准化处理。

    参数:
    - file_path: str, 数据文件路径。
    - pca_threshold: int, 使用 PCA 排名前多少的部分（列）。如果为 None，则使用全部列。
    - use_all: bool, 是否使用全部数据。如果为 True，则忽略 pca_threshold。

    返回:
    - data: np.ndarray, 处理后的数据。
    """
    # 加载数据
    data = np.load(file_path)

    # 判断是否使用全部数据
    if use_all:
        print("使用全部数据，忽略 pca_threshold。")
        pca_threshold = data.shape[1]  # 使用所有列
    elif pca_threshold is None:
        raise ValueError("pca_threshold 不能为 None，除非 use_all=True。")

    # 检查 pca_threshold 是否合法
    if pca_threshold > data.shape[1]:
        raise ValueError(f"pca_threshold ({pca_threshold}) 不能超过数据的列数 ({data.shape[1]})。")

    # 截取 PCA 排名前 pca_threshold 的部分
    data = data[:, :pca_threshold]

    # 标准化处理
    data = data / data.std(axis=0)[None, :]

    return data

def knn_graph(X, n_neighbors=15, space='l2', num_threads=8, params={'post': 2}):
    """
    使用 NMSLIB 构建 KNN 图。
    """
    index = nmslib.init(method='hnsw', space=space)
    index.addDataPointBatch(X)
    index.createIndex(params, print_progress=False)
    neighbours = index.knnQueryBatch(X, k=n_neighbors, num_threads=num_threads)
    ind = np.vstack([i for i, d in neighbours])
    sind = np.repeat(np.arange(ind.shape[0]), ind.shape[1])
    tind = ind.flatten()
    g = csr_matrix((np.ones(ind.shape[0] * ind.shape[1]), (sind, tind)), (ind.shape[0], ind.shape[0]))
    return g

def leiden_clustering(pcadata, n_neighbors=20, num_threads=10, output_file=None):
    """
    使用 Leiden 算法对 PCA 数据进行聚类。

    参数:
    - pcadata: np.ndarray, PCA 降维后的数据（样本数 x 特征数）。
    - n_neighbors: int, KNN 图中的近邻数。
    - num_threads: int, 使用的线程数。
    - output_file: str, 聚类结果保存路径（可选）。

    返回:
    - membership: np.ndarray, 每个样本的聚类标签。
    """
    # 标准化 PCA 数据
    pcadata_sd = pcadata.std(axis=0)
    pcadata = pcadata/(pcadata_sd)

    # 构建 KNN 图
    g = knn_graph(pcadata, n_neighbors=n_neighbors, num_threads=num_threads)

    # 构建 igraph 图
    sources, targets = g.nonzero()
    edgelist = zip(sources.tolist(), targets.tolist())
    G = ig.Graph(g.shape[0], list(edgelist))

    # Leiden 聚类
    partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    membership = np.array(partition.membership)

    # 保存聚类结果
    if output_file:
        np.save(output_file, membership)

    return membership

def umap_reduction(data, n_neighbors=15, min_dist=0.0, metric='euclidean'):
    """
    使用 UMAP 对数据进行降维。
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    embedding = reducer.fit_transform(data)
    return embedding

def tsne_reduction(data, init, perplexity=200, exaggeration=1.5, n_iter=1000, n_jobs=40):
    """
    使用 t-SNE 对数据进行降维。
    """
    affinities = affinity.PerplexityBasedNN(data, perplexity=perplexity, method="approx", n_jobs=n_jobs, random_state=0)
    embedding = TSNEEmbedding(init, affinities, negative_gradient_method="fft", n_jobs=n_jobs)
    embedding.optimize(n_iter=n_iter, exaggeration=exaggeration, momentum=0.8, inplace=True, learning_rate=1000)
    return embedding

def plot_clusters(embedding, cluster_labels, palette, output_path):
    """
    绘制聚类结果的散点图。
    """
    cluster_plot_data = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'cluster': cluster_labels})
    plt.figure(figsize=(6, 6))
    scatter = sns.scatterplot(data=cluster_plot_data, x='x', y='y', hue='cluster', palette=palette, alpha=1, s=5)
    scatter.set_title('Cluster Embedding Plot', fontsize=14)
    scatter.set_xlabel('TSNE1', fontsize=12)
    scatter.set_ylabel('TSNE2', fontsize=12)
    plt.legend(title="Cluster", loc="center left", fontsize=10, bbox_to_anchor=(1.0, 0.5), markerscale=4, ncol=2)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_auroc_auprc(total_labels, total_outputs, save_path=None, picture=False):
    """
    绘制 AUROC 和 AUPRC，并保存图像
    :param total_labels: 真实标签 (1D NumPy 数组)
    :param total_outputs: 预测值 (1D NumPy 数组，未经过 Sigmoid)
    :param save_path: 可选，保存文件的路径
    """
    # **计算 AUROC**
    fpr, tpr, _ = roc_curve(total_labels, total_outputs)  # **所有类别合并计算 ROC**
    auroc = auc(fpr, tpr)  # **计算 AUROC**
    print(f"AUROC: {auroc:.4f}")

    # **计算 AUPRC**
    precision, recall, _ = precision_recall_curve(total_labels, total_outputs)
    auprc = average_precision_score(total_labels, total_outputs)  # **计算 AUPRC**
    print(f"AUPRC: {auprc:.4f}")

    if picture:
        # **绘制 AUROC 和 AUPRC**
        plt.figure(figsize=(8, 6))

        # 绘制 AUROC
        plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUROC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')  # **随机分类器对角线**

        # 绘制 AUPRC
        plt.plot(recall, precision, color='blue', lw=2, linestyle='--', label=f'PR curve (AUPRC = {auprc:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR) / Recall')
        plt.ylabel('True Positive Rate (TPR) / Precision')
        plt.title('ROC & Precision-Recall Curve')
        plt.legend(loc='lower right')
        plt.grid(True)

        # **保存或显示图像**
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"AUROC & AUPRC plot saved as {save_path}")
        else:
            plt.show()

        plt.close()

    return auroc, auprc



def knn_graph(X, n_neighbors=15, space='l2', num_threads=8, params={'post': 2}):
    """
    使用 NMSLIB 构建 KNN 图。

    参数:
    - X: np.ndarray, 输入数据矩阵（样本数 x 特征数）。
    - n_neighbors: int, 每个样本的近邻数。
    - space: str, 距离度量空间（默认为 'l2'，即欧氏距离）。
    - num_threads: int, 使用的线程数。
    - params: dict, NMSLIB 索引参数。

    返回:
    - g: scipy.sparse.csr_matrix, KNN 图的稀疏矩阵。
    """
    index = nmslib.init(method='hnsw', space=space)
    index.addDataPointBatch(X)
    index.createIndex(params, print_progress=False)
    neighbours = index.knnQueryBatch(X, k=n_neighbors, num_threads=num_threads)
    ind = np.vstack([i for i, d in neighbours])
    sind = np.repeat(np.arange(ind.shape[0]), ind.shape[1])
    tind = ind.flatten()
    g = csr_matrix((np.ones(ind.shape[0] * ind.shape[1]), (sind, tind)), (ind.shape[0], ind.shape[0]))
    return g

def leiden_clustering(pcadata, n_neighbors=20, num_threads=10, output_file=None):
    """
    使用 Leiden 算法对 PCA 数据进行聚类。

    参数:
    - pcadata: np.ndarray, PCA 降维后的数据（样本数 x 特征数）。
    - n_neighbors: int, KNN 图中的近邻数。
    - num_threads: int, 使用的线程数。
    - output_file: str, 聚类结果保存路径（可选）。

    返回:
    - membership: np.ndarray, 每个样本的聚类标签。
    """
    # 标准化 PCA 数据
    pcadata_sd = pcadata.std(axis=0)
    pcadata = pcadata/(pcadata_sd)

    # 构建 KNN 图
    g = knn_graph(pcadata, n_neighbors=n_neighbors, num_threads=num_threads)

    # 构建 igraph 图
    sources, targets = g.nonzero()
    edgelist = zip(sources.tolist(), targets.tolist())
    G = ig.Graph(g.shape[0], list(edgelist))

    # Leiden 聚类
    partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    membership = np.array(partition.membership)

    # 保存聚类结果
    if output_file:
        np.save(output_file, membership)

    return membership

def generate_cluster_list(cluster_file, output_file='cluster_list.txt'):
    """
    加载 Leiden 聚类结果并生成 cluster_list.txt 文件。

    参数:
    - cluster_file: str, 聚类结果文件路径（.npy 文件）。
    - output_file: str, 输出的 cluster_list.txt 文件路径（默认为 'cluster_list.txt'）。

    返回:
    - unique_clusters: list, 所有唯一的聚类标签。
    """
    # 加载聚类结果
    partitionl = np.load(cluster_file, allow_pickle=True)
    membership = partitionl.tolist()

    # 找到所有唯一的聚类标签
    unique_clusters = sorted(set(membership))

    # 将聚类标签写入文件
    with open(output_file, 'w') as f:
        for i, cluster in enumerate(unique_clusters):
            f.write(f"cluster{i}\n")

    print(f"{output_file} 文件已经生成。")
    return unique_clusters

def load_preprocess_and_knn(file_path, pca_threshold, k_num=25, space='l2', space_params={'post': 2}, num_threads=10):
    """
    加载数据、预处理、构建 HNSW 索引并进行近邻搜索。

    参数:
    - file_path: str, 数据文件路径。
    - pca_threshold: int, 使用 PCA 排名前多少的部分。
    - k_num: int, 每个数据点的近邻数。
    - space: str, 距离度量空间（默认为 'l2'，即欧氏距离）。
    - space_params: dict, HNSW 索引参数。
    - num_threads: int, 使用的线程数。

    返回:
    - data: np.ndarray, 处理后的数据。
    - ind: np.ndarray, 近邻索引。
    - dist: np.ndarray, 近邻距离。
    """
    # 1. 加载数据并进行预处理
    data = np.load(file_path)
    data = data[:, :pca_threshold]
    data = data / data.std(axis=0)[None, :]

    # 2. 构建 HNSW 索引
    index = nmslib.init(method='hnsw', space=space, space_params=space_params)
    index.addDataPointBatch(data)
    index.createIndex(space_params, print_progress=False)

    # 3. 进行近邻搜索
    neighbors = index.knnQueryBatch(data, k=k_num, num_threads=num_threads)
    ind = np.vstack([i[:k_num] for i, d in neighbors])
    dist = np.vstack([d[:k_num] for i, d in neighbors])

    return data, ind, dist

def calculate_beta_umap_and_save(data, dist, k_num, beta_threshold=180, select_num=500000, min_dist=0.0, metric='euclidean', embedding_file=None, inds_file=None):
    """
    计算 beta 值、筛选数据点、使用 UMAP 进行降维并保存结果。

    参数:
    - data: np.ndarray, 输入数据。
    - dist: np.ndarray, 近邻距离。
    - k_num: int, 每个数据点的近邻数。
    - beta_threshold: float, beta 值的筛选阈值。
    - select_num: int, 选择的数据点数量。
    - min_dist: float, UMAP 的最小距离参数。
    - metric: str, 距离度量方法。
    - embedding_file: str, 降维结果保存路径（可选）。
    - inds_file: str, 数据点索引保存路径（可选）。

    返回:
    - embedding: np.ndarray, 降维后的数据。
    - selected_inds: np.ndarray, 选择的数据点索引。
    """
    import umap.umap_ as umap

    # 1. 计算 beta 值并筛选数据点
    beta = np.sum(
        (np.log(dist[:, 1:]) - np.mean(np.log(dist[:, 1:]), axis=1)[:, None]) *
        (np.log(np.arange(1, k_num)) - np.mean(np.log(np.arange(1, k_num))))[None, :],
        axis=1
    ) / np.sum((np.log(dist[:, 1:]) - np.mean(np.log(dist[:, 1:]), axis=1)[:, None]) ** 2, axis=1)
    allinds = (dist[:, 1] != 0) * (beta < beta_threshold)
    logp = np.log(dist[allinds, 1]) * beta[allinds]

    # 2. 使用 UMAP 进行降维
    np.random.seed(0)
    s = logp + np.random.gumbel(size=logp.shape)
    selectinds = np.argsort(-s)[:select_num]
    embedding = umap.UMAP(min_dist=min_dist, metric=metric).fit_transform(data[allinds, :][selectinds, :])
    selected_inds = np.where(allinds)[0][selectinds]

    # 3. 保存结果
    if embedding_file:
        np.save(embedding_file, embedding)
    if inds_file:
        np.save(inds_file, selected_inds)

    return embedding, selected_inds

def run_tsne(
    umap_file,
    inds_file,
    output_file,
    data,
    perplexity=200,
    exaggeration=1.5,
    ee=3,
    n_iter_first=250,
    n_iter_second=750,
    n_jobs=40,
    random_state=0,
    enable_second_optimization=True
):
    """
    运行 t-SNE 流程，支持二次优化开关。

    参数:
    - umap_file: str, UMAP 降维结果文件路径。
    - inds_file: str, 数据点索引文件路径。
    - output_file: str, t-SNE 嵌入结果保存路径。
    - data: np.ndarray, 原始数据矩阵（用于计算相似性）。
    - perplexity: int, t-SNE 的困惑度参数（默认为 200）。
    - exaggeration: float, 第二次优化的早期放大参数（默认为 1.5）。
    - ee: float, 第一次优化的早期放大参数（默认为 3）。
    - n_iter_first: int, 第一次优化的迭代次数（默认为 250）。
    - n_iter_second: int, 第二次优化的迭代次数（默认为 750）。
    - n_jobs: int, 使用的线程数（默认为 40）。
    - random_state: int, 随机种子（默认为 0）。
    - enable_second_optimization: bool, 是否启用第二次优化（默认为 True）。

    返回:
    - sample_embedding: TSNEEmbedding, 优化后的 t-SNE 嵌入。
    """
    # 设置随机种子
    np.random.seed(random_state)

    # 1. 加载 UMAP 降维结果和索引
    init = np.load(umap_file)
    inds = np.load(inds_file)

    # 2. 计算样本的相似性
    sample_affinities = affinity.PerplexityBasedNN(
        np.vstack(data[inds, :]),
        perplexity=perplexity,
        method="approx",
        n_jobs=n_jobs,
        random_state=random_state,
    )

    # 3. 进行第一次 t-SNE 嵌入优化
    sample_embedding = TSNEEmbedding(
        init,
        sample_affinities,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
    )
    sample_embedding.optimize(n_iter=n_iter_first, exaggeration=ee, momentum=0.5, inplace=True, learning_rate=1000)

    # 4. 根据开关决定是否进行第二次优化
    if enable_second_optimization:
        sample_embedding.optimize(n_iter=n_iter_second, exaggeration=exaggeration, momentum=0.8, inplace=True, learning_rate=1000)
        print("已完成第二次优化。")
    else:
        print("跳过第二次优化。")

    # 5. 保存结果
    np.save(output_file, sample_embedding)

    print(f"t-SNE 嵌入结果已保存到 {output_file}。")
    return sample_embedding

def prepare_cluster_plot_data(embedding, cluster_labels, inds, max_num=25000):
    """
    准备聚类绘图数据，确保每个 cluster 的数据点数量不超过最大限制。

    参数:
    - embedding: np.ndarray, 降维后的数据（二维数组，形状为 [n_samples, 2]）。
    - cluster_labels: np.ndarray, 每个数据点的聚类标签（形状为 [n_samples]）。
    - inds: np.ndarray, 数据点索引（形状为 [n_samples]）。
    - max_num: int, 每个 cluster 的最大数据点数量（默认为 25000）。

    返回:
    - cluster_plot_data: pd.DataFrame, 包含 x, y 和 cluster 信息的 DataFrame。
    - selected_inds: list, 被选为绘图的数据点索引。
    """
    # 初始化存储绘图数据的列表
    cluster_embedding_plot = []
    cluster_count = {}
    selected_inds = []

    # 遍历每个数据点
    for i, cluster in enumerate(cluster_labels[inds]):
        # 统计每个 cluster 的数据点数量
        if cluster in cluster_count:
            cluster_count[cluster] += 1
        else:
            cluster_count[cluster] = 1

        # 如果某个 cluster 的数据点数量超过最大限制，则跳过
        if cluster_count[cluster] > max_num:
            continue

        # 将数据添加到列表
        cluster_embedding_plot.append({'x': embedding[i, 0], 'y': embedding[i, 1], 'cluster': cluster})
        selected_inds.append(inds[i])

    # 将结果转换为 DataFrame
    cluster_plot_data = pd.DataFrame(cluster_embedding_plot)
    return cluster_plot_data, selected_inds

def plot_cluster_embedding(cluster_plot_data, custom_palette, output_file='picture/Cluster_Embedding.png', figsize=(6, 6), point_size=5, alpha=1):
    """
    绘制聚类嵌入图并保存为图像文件。

    参数:
    - cluster_plot_data: pd.DataFrame, 包含 x, y 和 cluster 信息的 DataFrame。
    - custom_palette: list, 自定义调色板（颜色列表）。
    - output_file: str, 输出图像文件路径（默认为 'picture/Cluster_Embedding.png'）。
    - figsize: tuple, 图像大小（默认为 (6, 6)）。
    - point_size: int, 点的大小（默认为 5）。
    - alpha: float, 点的透明度（默认为 1）。
    """
    # 设置 Seaborn 的主题
    sns.set_theme(style="whitegrid")

    # 创建绘图
    plt.figure(figsize=figsize)
    scatter = sns.scatterplot(
        data=cluster_plot_data,
        x='x',
        y='y',
        hue='cluster',  # 使用 cluster 列来区分颜色
        palette=custom_palette,  # 使用自定义调色板
        alpha=alpha,  # 透明度
        s=point_size  # 点的大小
    )

    # 添加 colorbar
    norm = mpl.colors.Normalize(vmin=cluster_plot_data['x'].min(), vmax=cluster_plot_data['x'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # 设置标题和标签
    scatter.set_title('Cluster Embedding Plot', fontsize=14)
    scatter.set_xlabel('TSNE1', fontsize=12)
    scatter.set_ylabel('TSNE2', fontsize=12)

    # 显示图例并调整位置和样式
    plt.legend(
        title="Cluster",
        loc="center left",
        fontsize=10,
        bbox_to_anchor=(1.0, 0.5),  # 图例位置
        markerscale=4,  # 调整点的大小
        ncol=2  # 设置两列
    )

    # 保存图像
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # 添加 bbox_inches='tight' 以确保图例不会被裁剪
    plt.show()

def plot_cluster_vs_others(cluster_plot_data, output_dir='picture', cluster_color="#FF6347", other_color="#D3D3D3", figsize=(6, 6), point_size=5, alpha=1):
    """
    绘制每个簇与其他簇的对比图，并保存为图像文件。

    参数:
    - cluster_plot_data: pd.DataFrame, 包含 x, y 和 cluster 信息的 DataFrame。
    - output_dir: str, 输出图像文件保存的目录（默认为 'picture'）。
    - cluster_color: str, 当前簇的颜色（默认为 "#FF6347"，即红色）。
    - other_color: str, 其他簇的颜色（默认为 "#D3D3D3"，即灰色）。
    - figsize: tuple, 图像大小（默认为 (6, 6)）。
    - point_size: int, 点的大小（默认为 5）。
    - alpha: float, 点的透明度（默认为 1）。
    """
    # 设置 Seaborn 的主题
    sns.set_theme(style="whitegrid")

    # 获取所有簇的编号
    unique_clusters = cluster_plot_data['cluster'].unique()

    # 循环每个簇，逐一单独为每个簇设置颜色
    for cluster in unique_clusters:
        # 复制数据以避免修改原始数据
        cluster_data = cluster_plot_data.copy()

        # 给非当前簇的数据标记统一颜色
        cluster_data['cluster_color'] = cluster_data['cluster'].apply(lambda x: f'cluster{cluster}' if x == cluster else 'other')

        # 创建绘图
        plt.figure(figsize=figsize)

        # 使用 hue 参数来为不同的簇设置颜色
        scatter = sns.scatterplot(
            data=cluster_data,
            x='x',
            y='y',
            hue='cluster_color',  # 根据 'cluster_color' 进行颜色区分
            palette={f'cluster{cluster}': cluster_color, 'other': other_color},  # 设置颜色映射
            alpha=alpha,  # 透明度
            s=point_size  # 点的大小
        )

        # 设置标题和标签
        scatter.set_title(f'Cluster {cluster} vs Other Clusters', fontsize=14)
        scatter.set_xlabel('TSNE1', fontsize=12)
        scatter.set_ylabel('TSNE2', fontsize=12)

        # 显示图例并调整位置和样式
        plt.legend(
            title="Cluster",
            loc="center left",
            fontsize=10,
            bbox_to_anchor=(1.0, 0.5),  # 图例位置
            markerscale=4,  # 调整点的大小
            ncol=2  # 设置两列
        )

        # 保存每个簇与其他簇的图像
        output_file = f'{output_dir}/Cluster_{cluster}_vs_Other.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')  # 添加 bbox_inches='tight' 以确保图例不会被裁剪
        plt.close()  # 关闭当前图像，避免图像重叠

        print(f"Cluster {cluster} 的对比图已保存到 {output_file}。")

def calculate_clustering_metrics(coordinates, labels):
    """
    计算聚类的内部评估指标。

    参数:
    - coordinates: np.ndarray, 数据点的坐标（形状为 [n_samples, n_features]）。
    - labels: np.ndarray, 每个数据点的聚类标签（形状为 [n_samples]）。

    返回:
    - metrics: dict, 包含以下指标：
        - silhouette_avg: Silhouette 系数。
        - ch_score: Calinski-Harabasz 指数。
        - db_score: Davies-Bouldin 指数。
    """
    # 计算 Silhouette 系数
    silhouette_avg = silhouette_score(coordinates, labels)

    # 计算 Calinski-Harabasz 指数
    ch_score = calinski_harabasz_score(coordinates, labels)

    # 计算 Davies-Bouldin 指数
    db_score = davies_bouldin_score(coordinates, labels)

    # 返回结果
    metrics = {
        'silhouette_avg': silhouette_avg,
        'ch_score': ch_score,
        'db_score': db_score
    }
    return metrics