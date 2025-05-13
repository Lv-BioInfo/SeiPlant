import argparse
import numpy as np
import torch
import os
import psutil
from utils.data_utils import load_and_preprocess_data_notag
from utils.data import NucDataset
from models.model_architectures import sei_model
from utils.preprocessing import *

def print_mem(tag=""):
    pid = os.getpid()
    process = psutil.Process(pid)
    mem = process.memory_info().rss / (1024**2)
    print(f"[MEM] {tag}: {mem:.2f} MB", flush=True)

def make_tag_dict(tag_file):
    with open(tag_file) as f:
        tag_list = f.readlines()
    return {item.strip(): index for index, item in enumerate(tag_list)}

def predict_new_data(model_path, new_data_path, model_tag_file, seq_len, device, common_tags, batch_size=256):
    print(f"\nLoading and preprocessing new data from {new_data_path} ...")
    x_new = load_and_preprocess_data_notag(new_data_path)
    model_tag_dict = make_tag_dict(model_tag_file)

    model_filtered_indices = []
    print("Filtered tag indices and names:")
    for tag in common_tags:
        if tag in model_tag_dict:
            index = list(model_tag_dict.keys()).index(tag)
            model_filtered_indices.append(index)
            print(f"Index: {index}, Tag: {tag}")
        else:
            print(f"Warning: Tag '{tag}' not found in model_tag_dict!")

    if not model_filtered_indices:
        raise ValueError("No common tags found in the model's dataset!")

    new_dataset = NucDataset(x=x_new, y=None)
    new_loader = torch.utils.data.DataLoader(dataset=new_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Loading trained model...")
    model = sei_model.Sei(sequence_length=seq_len, n_genomic_features=len(model_tag_dict))

    print_mem("Before model load")
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print_mem("Before prediction")

    predictions = []
    with torch.no_grad():
        for inputs in new_loader:
            inputs = inputs.to(device, dtype=torch.float).permute(0, 2, 1)
            outputs = model(inputs).squeeze()
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    filtered_predictions = predictions[:, model_filtered_indices]
    print("Prediction complete!")
    return filtered_predictions

def load_bed_file_with_pandas(bed_path):
    """
    使用 pandas 读取 BED 文件，并更新原始的 `start` 和 `end` 列，
    创建新的 `new` 列用于计算，然后删除 `new` 列。
    """
    print("读取和处理bed文件开始！")
    print_mem("读取bed文件前内存占用")
    # 使用 pandas 读取文件
    bed_data = pd.read_csv(bed_path, sep='\t', header=None, names=['chr', 'start', 'end'],
                           dtype={'chr': str, 'start': int, 'end': int})

    # 创建一个新的 'new' 列，用于存储原始的 'start' 值
    bed_data['new'] = bed_data['start']

    # 更新 `start` 和 `end` 列
    bed_data['start'] = bed_data['new'] + 448
    bed_data['end'] = bed_data['new'] + 576

    # 删除 `new` 列
    bed_data.drop(columns=['new'], inplace=True)
    print("读取和处理bed文件结束！")
    print_mem("读取bed文件后内存占用")

    return bed_data

def load_bed_file(bed_path):
    """
    读取BED文件，返回一个Numpy数组，其中每一行是BED文件中的一行（即包含3列）。
    第二列 (start) 和 第三列 (end) 会分别加上 448 和 576。
    """
    # 读取BED文件，假设文件用制表符分隔
    bed_data = np.loadtxt(bed_path, dtype=str, delimiter="\t")
    # 获取起始位置（第二列）和终止位置（第三列）
    start = bed_data[:, 1].astype(np.int64)  # 第二列是 start
    end = bed_data[:, 2].astype(np.int64)    # 第三列是 end

    # 对第二列 (start) 加上448，对第三列 (end) 加上576
    bed_data[:, 1] = (start + 448).astype(np.int64)  # 修改 start
    bed_data[:, 2] = (start + 576).astype(np.int64)  # 修改 end

    return bed_data

def save_bedgraph_with_predictions(bed_data, predictions, common_tags, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("拼接文件开始！")
    print_mem("拼接文件前内存占用！")

    assert predictions.shape[1] == len(common_tags), "预测列数与标签数量不匹配！"

    for idx, tag in enumerate(common_tags):
        if tag.lower() == 'h3k27me3':
            print(f"Skipping tag '{tag}'")
            continue

        tag_predictions = predictions[:, idx]
        zero_ratio = np.sum(tag_predictions == 0) / len(tag_predictions)
        print(f"Zero ratio before normalization for '{tag}': {zero_ratio:.4f}")

        tag_predictions = process_predictions_narrow_peak(tag_predictions)

        zero_ratio = np.sum(tag_predictions == 0) / len(tag_predictions)
        print(f"Zero ratio after normalization for '{tag}': {zero_ratio:.4f}")

        bedgraph_data = np.column_stack((bed_data, tag_predictions))
        bedgraph_data = bedgraph_data.astype(str)

        bedgraph_file_path = os.path.join(output_dir, f"{tag}.bedgraph")
        np.set_printoptions(precision=20, suppress=True)
        np.savetxt(bedgraph_file_path, bedgraph_data, fmt="%s\t%s\t%s\t%s", delimiter="\t")
        print(f"Saved {tag}.bedgraph to {bedgraph_file_path}")

def process_predictions_narrow_peak(values, new_min=0.1, new_max=1.0):
    """
    处理预测值数据：
    1. 过滤掉 <0.01 的值（设为0）。
    2. 对剩余 >0.01 的值进行 Min-Max 归一化（映射到 new_min - new_max）。
    3. 保持已经为 0 的部分不变。

    参数:
        values (np.ndarray): 原始预测值数组 (M,)。
        new_min (float): 归一化的最小值（默认 0.1）。
        new_max (float): 归一化的最大值（默认 1.0）。

    返回:
        np.ndarray: 归一化后的预测值数组。
    """
    values = values.copy()  # 复制数据，避免修改原始数组

    # 过滤掉 <0.01 的值（设为 0）
    values[values < 0.01] = 0

    # 仅对 >0.01 的部分进行归一化
    mask = values > 0
    if np.any(mask):  # 如果有大于 0.01 的值
        min_val = np.min(values[mask])
        max_val = np.max(values[mask])

        # 避免 min == max 造成除以零的问题
        if max_val > min_val:
            values[mask] = (values[mask] - min_val) / (max_val - min_val)
            values[mask] = values[mask] * (new_max - new_min) + new_min
        else:
            values[mask] = new_min  # 如果所有值都相等，设为 new_min

    return values

def process_predictions_narrow_peak_hvu(values, new_min=0, new_max=1.0):
    """
    处理预测值数据：
    1. 过滤掉 <0.01 的值（设为0）。
    2. 对剩余 >0.01 的值进行 Min-Max 归一化（映射到 new_min - new_max）。
    3. 保持已经为 0 的部分不变。

    参数:
        values (np.ndarray): 原始预测值数组 (M,)。
        new_min (float): 归一化的最小值（默认 0.1）。
        new_max (float): 归一化的最大值（默认 1.0）。

    返回:
        np.ndarray: 归一化后的预测值数组。
    """
    values = values.copy()  # 复制数据，避免修改原始数组

    # 过滤掉 <0.01 的值（设为 0）
    values[values < 0.1] = 0

    # 仅对 >0.01 的部分进行归一化
    mask = values > 0
    if np.any(mask):  # 如果有大于 0.01 的值
        min_val = np.min(values[mask])
        max_val = np.max(values[mask])

        # 避免 min == max 造成除以零的问题
        if max_val > min_val:
            values[mask] = (values[mask] - min_val) / (max_val - min_val)
            values[mask] = values[mask] * (new_max - new_min) + new_min
        else:
            values[mask] = new_min  # 如果所有值都相等，设为 new_min

    return values

def main():
    parser = argparse.ArgumentParser(description="Predict histone modification signals using a trained model.")
    parser.add_argument("--model_path", required=True, help="Path to trained model file (.model)")
    parser.add_argument("--model_tag_file", required=True, help="Path to model tag file")
    parser.add_argument("--species", required=True, help="species to be predicted")
    parser.add_argument("--fa_path", required=True, help="Base path to species fasta data")
    parser.add_argument("--output_dir", required=True, help="Directory to save prediction results")
    parser.add_argument("--seq_len", type=int, default=1024, help="Input sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for prediction")
    parser.add_argument("--bed_file", required=True, help="Bed file")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    species = args.species
    common_tags = set()
    with open(args.model_tag_file, "r", encoding="utf-8") as f:
        for line in f:
            tag = line.strip().split("\t")[0]
            common_tags.add(tag)

    os.makedirs(args.output_dir, exist_ok=True)

    fasta_file = args.fa_path
    model_path = args.model_path
    tag_path = args.model_tag_file

    preds = predict_new_data(
        model_path=model_path,
        new_data_path=fasta_file,
        model_tag_file=tag_path,
        seq_len=args.seq_len,
        device=device,
        common_tags=common_tags,
        batch_size=args.batch_size,
    )

    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, f"{species}_predictions.npy"), preds)
    print(f"Saved predictions for {species} at {result_dir}")

    print(f"Prediciton Finished. Making bedgraph for {species}")
    prediction_npy_path = os.path.join(result_dir, f"{species}_predictions.npy")
    print(f"\n=== Processing for {args.species} ===")

    predictions = np.load(prediction_npy_path)
    print(f"预测矩阵形状：{predictions.shape}")

    bed_file_path = args.bed_file

    bed_data = load_bed_file_with_pandas(bed_file_path)
    save_bedgraph_with_predictions(bed_data, predictions, common_tags, result_dir)
    print(f"Combined BED and predictions saved at: {result_dir}")

if __name__ == "__main__":
    main()
