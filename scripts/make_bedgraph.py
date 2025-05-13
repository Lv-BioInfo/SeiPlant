import numpy as np
import os
import sys
import argparse
import pandas as pd
import psutil

sys.path.append("/public/workspace/lvtongxuan/script/merged_tag_Psei")


def print_mem(tag=""):
    pid = os.getpid()
    process = psutil.Process(pid)
    mem = process.memory_info().rss / (1024**2)
    print(f"[MEM] {tag}: {mem:.2f} MB", flush=True)


def load_bed_file_with_pandas(bed_path):
    print("读取和处理bed文件开始！")
    print_mem("读取bed文件前内存占用")
    bed_data = pd.read_csv(
        bed_path, sep='\t', header=None, names=['chr', 'start', 'end'],
        dtype={'chr': str, 'start': int, 'end': int}
    )
    bed_data['start'] = bed_data['start'] + 448
    bed_data['end'] = bed_data['start'] + 128
    print("读取和处理bed文件结束！")
    print_mem("读取bed文件后内存占用")
    return bed_data


def process_predictions_narrow_peak_hvu(values, new_min=0, new_max=1.0):
    values = values.copy()
    values[values < 0.1] = 0
    mask = values > 0
    if np.any(mask):
        min_val = np.min(values[mask])
        max_val = np.max(values[mask])
        if max_val > min_val:
            values[mask] = (values[mask] - min_val) / (max_val - min_val)
            values[mask] = values[mask] * (new_max - new_min) + new_min
        else:
            values[mask] = new_min
    return values


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

        tag_predictions = process_predictions_narrow_peak_hvu(tag_predictions)

        zero_ratio = np.sum(tag_predictions == 0) / len(tag_predictions)
        print(f"Zero ratio after normalization for '{tag}': {zero_ratio:.4f}")

        bedgraph_data = np.column_stack((bed_data, tag_predictions))
        bedgraph_data = bedgraph_data.astype(str)

        bedgraph_file_path = os.path.join(output_dir, f"{tag}.bedgraph")
        np.set_printoptions(precision=20, suppress=True)
        np.savetxt(bedgraph_file_path, bedgraph_data, fmt="%s\t%s\t%s\t%s", delimiter="\t")
        print(f"Saved {tag}.bedgraph to {bedgraph_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Process predictions and export bedGraph files")
    parser.add_argument('--data_path', required=True, help="Path to input data directory")
    parser.add_argument('--result_path', required=True, help="Path to output results directory")
    parser.add_argument('--species', required=True, help="Species name (e.g., zea_mays)")
    parser.add_argument('--tags', nargs='+', required=True, help="List of common tags (e.g., zma_H3K4ME1 ...)")

    args = parser.parse_args()

    bed_file_path = os.path.join(args.data_path, args.species, f"{args.species}_1024_128_filtered.bed")
    species_result_path = os.path.join(args.result_path, f"results_{args.species}")
    prediction_npy_path = os.path.join(species_result_path, f"{args.species}_predictions.npy")

    print(f"\n=== Processing for {args.species} ===")

    predictions = np.load(prediction_npy_path)
    print(f"预测矩阵形状：{predictions.shape}")

    bed_data = load_bed_file_with_pandas(bed_file_path)
    save_bedgraph_with_predictions(bed_data, predictions, args.tags, species_result_path)

    print(f"Combined BED and predictions saved at: {species_result_path}")


if __name__ == "__main__":
    main()
