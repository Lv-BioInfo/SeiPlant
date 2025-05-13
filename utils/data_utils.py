import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader
from utils.data import *
from utils.preprocessing import *
from utils.preprocess import *
import itertools
import numpy as np
from collections import Counter
import random

def load_and_preprocess_data_notag(fasta_file):
    # 读取序列数据和标签文件
    seqs, tags, pos = load_data_notag(fasta_file, label=True, strand=False, pos=True)

    # 数据预处理：序列数据转为one-hot形式，标签转为数字映射
    nuc_pre = NucPreprocess(seqs)
    X_all = nuc_pre.onehot_for_nuc()

    return X_all

def load_and_preprocess_data(fasta_file, tag_file):
    # 读取序列数据和标签文件
    seqs, tags, pos = load_data(fasta_file, label=True, strand=False, pos=True)

    # 加载标签文件，并创建标签字典
    with open(tag_file) as f:
        tag_list = f.readlines()

    tag_dict = {item.strip(): index for index, item in enumerate(tag_list)}
    feature_size = len(tag_dict)
    print("tag_dict:", feature_size)

    # 数据预处理：序列数据转为one-hot形式，标签转为数字映射
    nuc_pre = NucPreprocess(seqs)
    X_all = nuc_pre.onehot_for_nuc()
    labels = tag_merged_encode(tags, tag_dict, sep=',')

    return X_all, labels, pos, tag_dict

# utils/data_utils.py
def split_data_by_chromosome_cross_val(X_all, labels, pos, tag_dict, num_folds=5):
    # 提取染色体号
    chromosome_numbers = []
    for item in pos:
        if item.startswith("Chr"):
            item = item[3:]  # 去掉前缀 "Chr"
        chrom_number = int(item.split(':')[0])
        chromosome_numbers.append(chrom_number)

    # 使用 Counter 来统计不同染色体的数量
    chromosome_count = Counter(chromosome_numbers)
    print(f"Chromosome count: {chromosome_count}")

    # 获取所有染色体
    chromosomes = list(chromosome_count.keys())

    # 存储每次的训练集、验证集和测试集组合
    fold_data = []

    # 获取所有可能的染色体组合：从5个染色体中选择1个为测试集，1个为验证集，剩余为训练集
    combinations = list(itertools.combinations(chromosomes, 2))

    # 遍历每一个组合
    for comb in combinations:
        test_chrom = comb[0]  # 选择一个染色体作为测试集
        val_chrom = comb[1]  # 选择一个染色体作为验证集

        # 剩下的染色体作为训练集
        train_chromosomes = [chrom for chrom in chromosomes if chrom not in [test_chrom, val_chrom]]

        # 划分训练集、验证集和测试集的索引
        testing_index = [index for index, chrom in enumerate(chromosome_numbers) if chrom == test_chrom]
        validation_index = [index for index, chrom in enumerate(chromosome_numbers) if chrom == val_chrom]
        training_index = [index for index, chrom in enumerate(chromosome_numbers) if chrom in train_chromosomes]

        # 数据切分
        X_training = [X_all[i] for i in training_index]
        y_training = np.array([labels[i] for i in training_index])

        X_validation = [X_all[i] for i in validation_index]
        y_validation = np.array([labels[i] for i in validation_index])

        X_testing = [X_all[i] for i in testing_index]
        y_testing = np.array([labels[i] for i in testing_index])

        # 存储每次的训练集、验证集和测试集
        fold_data.append(
            (X_training, y_training, X_validation, y_validation, X_testing, y_testing, test_chrom, val_chrom, train_chromosomes))
    # 从fold_data中随机选择指定数量的组合（这里选择5对）
    random.seed(42)
    random_folds = random.sample(fold_data, num_folds)

    return random_folds, tag_dict


def split_data_by_chromosome(X_all, labels, pos, tag_dict):
    # 提取染色体号
    chromosome_numbers = []
    for item in pos:
        if item.startswith("Chr"):
            item = item[3:]  # 去掉前缀 "Chr"
        chrom_number = int(item.split(':')[0])
        chromosome_numbers.append(chrom_number)

    # 使用 Counter 来统计不同染色体的数量
    chromosome_count = Counter(chromosome_numbers)
    print(chromosome_count)

    # 定义测试集和验证集的染色体号
    test_chromosomes = [3]  
    validation_chromosomes = [4]  

    # 将染色体号转换为集合，提高查找效率
    test_chromosomes_set = set(test_chromosomes)
    validation_chromosomes_set = set(validation_chromosomes)

    # 根据染色体号划分数据集
    testing_index = [index for index, chrom in enumerate(chromosome_numbers) if chrom in test_chromosomes_set]
    validation_index = [index for index, chrom in enumerate(chromosome_numbers) if chrom in validation_chromosomes_set]

    # 通过集合判断是否属于训练集
    validation_and_testing_index = set(testing_index + validation_index)
    training_index = [i for i in range(len(pos)) if i not in validation_and_testing_index]

    # 输出数据集长度
    print(f"Training set length: {len(training_index)}")
    print(f"Testing set length: {len(testing_index)}")
    print(f"Validation set length: {len(validation_index)}")

    # 数据切分
    X_training = [X_all[i] for i in training_index]
    y_training = np.array([labels[i] for i in training_index])

    X_validation = [X_all[i] for i in validation_index]
    y_validation = np.array([labels[i] for i in validation_index])

    X_testing = [X_all[i] for i in testing_index]
    y_testing = np.array([labels[i] for i in testing_index])

    return X_training, y_training, X_validation, y_validation, X_testing, y_testing, tag_dict

# utils/data_utils.py
def create_data_loaders(X_training, y_training, X_validation, y_validation, batch_size=256):
    # 将数据转为 PyTorch 数据集，并定义批量大小
    train_set = NucDataset(x=X_training, y=y_training)
    val_set = NucDataset(x=X_validation, y=y_validation)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader