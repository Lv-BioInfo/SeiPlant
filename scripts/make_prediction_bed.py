# -*- coding: utf-8 -*-
import os
import re
import argparse
from Bio import SeqIO
import roman


def parse_chromosome(chrom):
    match = re.match(r'chr(\d+)([ABD]?)|chr(Un)', chrom)
    if match:
        num_part = int(match.group(1)) if match.group(1) else float('inf')
        letter_part = match.group(2) if match.group(2) else 'Z'
        return (num_part, letter_part)
    return (float('inf'), 'Z')


def parse_chr_numeric(chrom):
    """
    解析形如 Chr1、Chr2、Chr10 的染色体名称，返回用于排序的整数。
    非数字部分默认排序靠后。
    """
    match = re.match(r"[Cc]hr(\d+)", chrom)
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # 不匹配的染色体放到最后


def generate_sliding_window(chrom_size_file, fa_file, output_path, species_name, window_size=1024, step_size=128):
    output_fa_path = os.path.join(output_path, f"{species_name}_{window_size}_{step_size}.fa")
    output_bed_path = os.path.join(output_path, f"{species_name}_{window_size}_{step_size}_filtered.bed")

    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    chrom_sizes = {}
    with open(chrom_size_file, 'r') as f:
        for line in f:
            chrom, size = line.strip().split('\t')
            if len(chrom) <= 5:
                chrom_sizes[chrom] = int(size)

    if species_name == "triticum_aestivum":
        sorted_chroms = sorted(chrom_sizes.items(), key=lambda x: parse_chromosome(x[0]))
    elif species_name == "roman":
        sorted_chroms = sorted(chrom_sizes.items(), key=lambda x: roman.fromRoman(x[0]))
    else:
        sorted_chroms = sorted(chrom_sizes.items(), key=lambda x: parse_chr_numeric(x[0]))

    # 输出排序后的染色体名称
    sorted_chrom_names = [chrom for chrom, _ in sorted_chroms]
    # 输出字典中保存的key（染色体名称）
    print("按照大小排序后的染色体名称有：", sorted_chrom_names)

    genome_sequences = SeqIO.to_dict(SeqIO.parse(fa_file, 'fasta'))

    with open(output_bed_path, 'w') as bed_output, open(output_fa_path, 'w') as fa_output:
        for chrom, size in sorted_chroms:
            seq = genome_sequences.get(chrom)
            if not seq:
                continue
            for start in range(0, size - window_size + 1, step_size):
                end = start + window_size
                subseq = seq.seq[start:end]
                if len(subseq) != window_size:
                    continue
                if all(base in 'ATCG' for base in subseq.upper()):
                    bed_output.write(f"{chrom}\t{start}\t{end}\n")
                    fa_output.write(f">{chrom}:{start}-{end}\n{subseq}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate genome-wide sliding windows from FASTA and .size file.")
    parser.add_argument("--fasta", required=True, help="Input genome FASTA file")
    parser.add_argument("--size", required=True, help="Input chromosome size file")
    parser.add_argument("--species", required=True, help="Species name for sorting")
    parser.add_argument("--output_path", required=True, help="Output Path")
    parser.add_argument("--window_size", type=int, default=1024, help="Sliding window size (default: 1024)")
    parser.add_argument("--step_size", type=int, default=128, help="Sliding step size (default: 128)")
    args = parser.parse_args()

    generate_sliding_window(
        chrom_size_file=args.size,
        fa_file=args.fasta,
        output_path=args.output_path,
        species_name=args.species,
        window_size=args.window_size,
        step_size=args.step_size
    )
