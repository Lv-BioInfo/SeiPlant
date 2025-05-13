import os.path

import torch
from plotnine import *
import pandas as pd

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch.nn.functional as F

from Bio import SeqIO
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
# from plotnine import *


def perform_pca_and_plot(total_pre, result_path, n_components=40, batch_size=100000):
    """
    对预测结果进行 PCA 降维，并绘制 PCA 解释方差比例的图表。

    参数:
        total_pre (np.ndarray): 预测结果。
        record_pattern (str): 记录模式。
        suffix (str): 文件名后缀。
        n_components (int): PCA 主成分数量。
        batch_size (int): 批量大小。

    返回:
        PCA_pre (np.ndarray): 降维后的数据。
    """
    # PCA 降维
    PCA_transformer = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    PCA_pre = PCA_transformer.fit_transform(total_pre)
    print("PCA降维后的数据形状:", PCA_pre.shape)

    # 保存降维后的数据
    np.save(os.path.join(result_path,"new_data_predictions_IPCA_100.npy"), PCA_pre)

    # 计算累计方差比例
    cumulative_variance = np.cumsum(PCA_transformer.explained_variance_ratio_)
    n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
    print(f"解释了 99% 方差的主成分数量: {n_components_99}")

    # 获取前 n_components_99 个主成分的解释方差比例
    top_pca = PCA_transformer.explained_variance_ratio_[:n_components_99]
    pca_ratio = pd.DataFrame({'X': range(1, n_components_99 + 1), 'Y': top_pca})

    # 绘制图表
    p = (ggplot(pca_ratio, aes(x='X', y='Y'))
         + geom_point()
         + theme_bw()
         + scale_x_continuous(limits=[0, n_components_99], breaks=list(range(0, n_components_99)),
                              labels=list(range(1, n_components_99 + 1)))
         + ylab('PCA Ratio')
         + xlab('PCA Num')
         )

    # 保存图表
    p.save('PCA_ratio_plot_99.png',width=8, height=6)
    print(p)

    return PCA_pre

def load_data_notag(path, ftype='fasta', label=False, strand=False, pos=False):
    '''
    Loading data from specific file path. Default file type is fasta.
    The function returns two list: seq_list and seq_label
    If label equals to False, returns seq_list
    If strand equals to True, return complement sequence as well
    If pos equals to True, return sequences position info as well
    '''
    seqs = []
    if label:
        seqlabels = []
    if pos:
        seqpos = []

    try:
        # 打开文件时显式指定编码
        with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
            for seq in SeqIO.parse(handle, ftype):
                # 处理序列
                seqs.append(str(seq.seq))
                if strand:
                    seqs.append(str(seq.seq.complement()))

                seqName = seq.id
                # 解析染色体号和位置
                chrom_pos = seqName.split(':')
                chrom = chrom_pos[0]  # 获取染色体号
                position = chrom_pos[1]  # 获取start-end信息

                if label:
                    # 不再处理tagname，直接使用染色体号作为标签
                    seqlabels.append(chrom)
                    if strand:
                        seqlabels.append(chrom)

                if pos:
                    # 直接使用start-end信息
                    seqpos.append(position)
                    if strand:
                        seqpos.append(position)

    except FileNotFoundError as e:
        raise Exception(f"Error: The file at {path} was not found. Please check the file path.") from e
    except UnicodeDecodeError as e:
        raise Exception(
            f"UnicodeDecodeError: The file {path} contains invalid characters. Please check the encoding.") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading data from {path}: {str(e)}") from e

    if label and pos:
        return seqs, seqlabels, seqpos
    elif label:
        return seqs, seqlabels
    elif pos:
        return seqs, seqpos
    else:
        return seqs


def load_data(path, ftype='fasta', label=False, strand=False, pos=False):
    '''
    Loading data from specific file path. Default file type is fasta.
    The function returns two list: seq_list and seq_label
    If label equals to False, returns seq_list
    If strand equals to True, return complement sequence as well
    If pos equals to True, return sequences position info as well
    '''
    seqs = []
    if label:
        seqlabels = []
    if pos:
        seqpos = []

    try:
        # 打开文件时显式指定编码
        with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
            for seq in SeqIO.parse(handle, ftype):
                # 处理序列
                seqs.append(str(seq.seq))
                if strand:
                    seqs.append(str(seq.seq.complement()))

                seqName = seq.id
                if label:
                    seqlabels.append(seqName.split('::')[0])
                    if strand:
                        seqlabels.append(seqName.split('::')[0])

                if pos:
                    seqpos.append(seqName.split('::')[1])
                    if strand:
                        seqpos.append(seqName.split('::')[1])

    except FileNotFoundError as e:
        raise Exception(f"Error: The file at {path} was not found. Please check the file path.") from e
    except UnicodeDecodeError as e:
        raise Exception(
            f"UnicodeDecodeError: The file {path} contains invalid characters. Please check the encoding.") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading data from {path}: {str(e)}") from e

    if label and pos:
        return seqs, seqlabels, seqpos
    elif label:
        return seqs, seqlabels
    elif pos:
        return seqs, seqpos
    else:
        return seqs


def load_data2(path, ftype='fasta', label=False, strand=False, pos=False):
    '''
        Loading data from specific file path. Default file type is fasta.
        The function returns two list: seq_list and seq_label
        If label equals to False, returns seq_list
        If strand equals to True, return complement sequnece as well
        If pos equals to True, return sequences position info as well
     '''
    seqs = []
    if label:
        seqlabels = []
    if pos:
        seqpos = []

    for seq in SeqIO.parse(path, ftype):
        # if 'N' not in seq.seq and len(seq) == 300:
        seqs.append(str(seq.seq))
        if strand:
            seqs.append(str(seq.seq.complement()))

        seqName = seq.id
        if label:
            seqlabels.append(seqName.split('::')[0])
            if strand:
                seqlabels.append(seqName.split('::')[0])

        if pos:
            seqpos.append(seqName.split('::')[1:])
            if strand:
                seqpos.append(seqName.split('::')[1:])

    if label and pos:
        return seqs, seqlabels, seqpos
    elif label:
        return seqs, seqlabels
    elif pos:
        return seqs, seqpos
    else:
        return seqs


# def draw_scores_distribution(scores_list):
#     '''
#         Produce scores(labels)'s values distribution
#         Annotate metrics of distribution too
#     '''
#     x = list(range(len(scores_list)))
#     y = scores_list
#     df = pd.DataFrame({'x': x, 'y': y})

#     p = (ggplot(df, aes(x='x', y='y')) + geom_point(size=.1, color='#9ecae1') + geom_line(size=.05,
#                                                                                           color='#9ecae133') + theme_bw() +
#          annotate(geom='text', x=max(x) - 500, y=max(y) - 500,
#                   label='Scores\nNum:{}\nMin:{:.2f}\nMean:{:.2f}\nMax{:.2f}'.format(len(y), min(y), np.mean(y), max(y)),
#                   color='#fdae6b', size=8))

#     return p


def tag_encode(tags, tag_dict, sep=','):
    '''
        Encode tags as binary vectors representing the absence/presence of the Profiles
    '''
    result = []
    for tag in tags:
        tmp_result = [0] * len(tag_dict)
        tag = tag.split(sep)
        for item in tag:
            tmp_result[tag_dict[item]] = 1
        result.append(tmp_result)

    return np.array(result)


def tag_merged_encode(tags, tag_dict, sep=','):
    '''
        Encode tags as vectors where the last part of the tag (after the last underscore) is the score,
        and everything before that is the tag_name.
    '''
    result = []
    for tag in tags:
        # 初始化一个全零的向量，长度为 tag_dict 的大小
        tmp_result = [0] * len(tag_dict)
        tag = tag.split(sep)  # 按分隔符分割 tags
        for item in tag:
            # 按下划线分割，将最后一部分作为 score，前面的部分整体作为 tag_name
            *tag_name_parts, score = item.rsplit('_', maxsplit=1)
            tag_name = '_'.join(tag_name_parts)  # 将前面的部分重新拼接为 tag_name
            tmp_result[tag_dict[tag_name]] = float(score)  # 将得分转换为浮点数并赋值
        result.append(tmp_result)

    return np.array(result)


def evaluation(outputs, labels):
    '''
        Function that can evaluate models prediction classes with real data label
        Return correct rate
    '''
    # outputs => probability (float)
    # labels => labels
    # 根据outputs的概率分配label，跟真实的label做比较，输出正确率
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()

    return correct


import torch.nn.functional as F


def evaluation_mse(outputs, labels):
    '''
    Function that calculates the mean squared error between outputs and labels
    '''
    mse_loss = F.mse_loss(outputs, labels)
    return mse_loss.item()  # 返回 MSE 损失值


def evaluation_r2(outputs, labels):
    '''
    Function that calculates the R-squared value between outputs and labels
    '''
    ss_total = torch.sum((labels - torch.mean(labels)) ** 2)
    ss_residual = torch.sum((labels - outputs) ** 2)
    r2_score = 1 - ss_residual / ss_total
    return r2_score.item()


import numpy as np


def correlation_coefficient(outputs, labels):
    """
    计算预测值与标签之间的皮尔逊相关系数
    outputs: 模型的预测值 (M行125列)
    labels: 真实标签 (M行125列)
    """
    outputs = outputs.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    correlation = np.corrcoef(labels.flatten(), outputs.flatten())[0, 1]
    return correlation


def mean_squared_error(outputs, labels):
    """
    计算预测值与真实值之间的均方误差（MSE）

    参数:
        outputs (torch.Tensor): 模型的预测值，形状为 (M, 125)
        labels (torch.Tensor): 真实标签值，形状为 (M, 125)

    返回:
        mse (float): 均方误差值
    """
    # 确保 outputs 和 labels 在同一个设备上
    if outputs.device != labels.device:
        labels = labels.to(outputs.device)

    # 计算 MSE
    mse = torch.mean((outputs - labels) ** 2).item()
    return mse


def draw_metrics(x, y, anno, model=None, metric='ROC'):
    '''
        Function that can produce ROC curve using (fpr, tpr) and PR curve using (recall, precision)
        Parameters `metric` should only be ROC or PR, anno could be arbitrary
    '''
    if metric == 'ROC':
        fpr = x
        tpr = y
        au_roc = anno
        if model is not None:
            roc = pd.DataFrame({'x': fpr, 'y': tpr, 'model': model})
            p = (ggplot(roc, aes(x='x', y='y', color='model')) + geom_line())
        else:
            roc = pd.DataFrame({'x': fpr, 'y': tpr})
            p = (ggplot(roc, aes(x='x', y='y')) + geom_line(color='orange'))

        p = (p + geom_abline(intercept=0, slope=1, color='blue', linetype='dashed') +
             xlab('False Positive Rate') + ylab('True Positive Rate') + annotate(geom='text', x=0.65, y=0.125,
                                                                                 label=anno) +
             ggtitle('Receiver operating characteristic') + scale_x_continuous(expand=(0, 0)) + scale_y_continuous(
                    expand=(0, 0)) + theme_bw())
        return p
    elif metric == 'PR':
        recall = x
        precision = y
        aver_precision = anno
        if model is not None:
            pr = pd.DataFrame({'x': recall, 'y': precision, 'model': model})
            p = (ggplot(pr, aes(x='x', y='y', color='model')) + geom_line())
        else:
            pr = pd.DataFrame({'x': recall, 'y': precision})
            p = (ggplot(pr, aes(x='x', y='y')) + geom_line(color='orange'))

        p = (p + xlab('False Positive Rate') + ylab('True Positive Rate') +
             annotate(geom='text', x=0.65, y=0.125, label=anno) + ggtitle('Precision-Recall curve') +
             scale_x_continuous(expand=(0, 0)) + scale_y_continuous(expand=(0, 0)) + theme_bw())
        return p
    else:
        print('Parameter "metric" should be "ROC" or "PR" not "{}"', format(metric))
        return 0


def get_metrics(outputs, labels, metric='ROC'):
    if metric == 'ROC':
        fpr, tpr, _ = roc_curve(labels, outputs)
        au_roc = auc(fpr, tpr)
        return fpr, tpr, au_roc
    elif metric == 'PR':
        recall, precision, _ = precision_recall_curve(labels, outputs)
        aver_precision = average_precision_score(labels, outputs)
        return recall, precision, aver_precision
    else:
        print('Parameter "metric" should be "ROC" or "PR" not "{}"', format(metric))
        return 0

def plot_histogram(outputs, labels):
    """
    绘制预测值和真实值的分布直方图
    outputs: 模型的预测值 (M行125列)
    labels: 真实标签 (M行125列)
    """
    outputs = outputs.cpu().detach().numpy().flatten()
    labels = labels.cpu().detach().numpy().flatten()

    plt.figure(figsize=(8, 6))

    # 绘制真实标签的直方图
    plt.hist(labels, bins=30, alpha=0.6, label='True Values', color='blue', edgecolor='black')

    # 绘制预测值的直方图
    plt.hist(outputs, bins=30, alpha=0.6, label='Predicted Values', color='red', edgecolor='black')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of True vs Predicted Values')
    plt.legend(loc='upper right')
    plt.show()

def plot_scatter_with_regression(outputs, labels):
    """
    绘制真实值与预测值的散点图，并拟合回归线
    outputs: 模型的预测值 (M行125列)
    labels: 真实标签 (M行125列)
    """
    outputs = outputs.cpu().detach().numpy().flatten()
    labels = labels.cpu().detach().numpy().flatten()

    # 绘制散点图
    plt.scatter(labels, outputs, alpha=0.6, label="Predicted vs True")

    # 拟合回归线
    z = np.polyfit(labels, outputs, 1)
    p = np.poly1d(z)
    plt.plot(labels, p(labels), color='red', label="Regression Line")

    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted with Regression Line')
    plt.legend(loc='upper left')
    plt.show()

