import pandas as pd
import os
import re
from collections import defaultdict
from tqdm import tqdm
"""
stat_mergedtag6.py
实现：
    处理单物种的tag数据：水稻，拟南芥和玉米
    输入fa序列文件和bed文件，进行merged操作
    输出可以训练的fa文件和tag.txt文件
    针对bedtools 过滤后的 interval区间进行顾虑，保留 bedtools multiinter -i 命令处理后大于2的interval区间 
    在stat_mergedtag5基础上改进，不使用覆盖率，使用0-1的标准化指标

stat_mergedtag7.py
实现：
    处理单物种的tag数据：水稻，拟南芥和玉米
    输入fa序列文件和bed文件，进行merged操作
    输出可以训练的fa文件和tag.txt文件
    针对bedtools 过滤后的 interval区间进行顾虑，保留 bedtools multiinter -i 命令处理后大于2的interval区间 
    在stat_mergedtag6基础上改进，不使用覆盖率，使用0-1的标准化指标
    同时修复正负样本不平衡的问题

该版本实现了二者结合和封装 2025-02-19 更新
包含了生产数据集的诸多部分，包括产生，配对，过滤低质量，计算权重等步骤

"""


# ----------------------------- Utility Functions -----------------------------


def read_fa(fa_file_path):
    """读取FA文件并将其存储为字典格式。每个染色体对应FA文件中的序列。"""
    genome = {}
    with open(fa_file_path, 'r') as f:
        chrom, sequence = None, []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if chrom:
                    genome[chrom] = ''.join(sequence)
                chrom = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if chrom:
            genome[chrom] = ''.join(sequence)
    return genome


def get_sequence_from_fa(genome, chrom, start, end):
    """从FA文件中提取给定染色体和区间的序列。"""
    sequence = genome.get(chrom)
    if sequence is None:
        return None
    seq_substr = sequence[start:end]
    return seq_substr if re.match("^[ATCG]+$", seq_substr) else None


def read_bed_file(file_path):
    """读取BED文件并进行基本的过滤处理"""
    try:
        bed_df = pd.read_csv(file_path, sep="\t", header=None)
        if not pd.api.types.is_string_dtype(bed_df[0]):
            bed_df[0] = bed_df[0].astype(str)
        bed_df = bed_df[~bed_df[0].str.contains('Un|Sy|scaffold', case=False, na=False)]
        bed_df = bed_df.iloc[:, :3]
        bed_df.columns = ['chrom', 'start', 'end']
        return bed_df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def read_bed_file_score(file_path):
    """
    读取BED文件并进行基本的过滤处理
    File_path文件使用  /public/workspace/lvtongxuan/data/test_chiphub/osa_ath_zma/stat_important_bed_interval_shell.py 脚本处理
    包含一个覆盖度Score文件
    """
    try:
        bed_df = pd.read_csv(file_path, sep="\t", header=None)

        # 确保第一列是字符串类型
        if not pd.api.types.is_string_dtype(bed_df[0]):
            bed_df[0] = bed_df[0].astype(str)

        # 过滤掉包含 'Un'、'Sy'、'scaffold' 的染色体行
        bed_df = bed_df[~bed_df[0].str.contains('Un|Sy|scaffold', case=False, na=False)]

        # 过滤掉长度小于 128 的序列
        bed_df = bed_df[(bed_df[2] - bed_df[1]) >= 128]

        # 仅保留前四列，并重命名列名
        bed_df = bed_df.iloc[:, :4]
        bed_df.columns = ['chrom', 'start', 'end', 'score']

        return bed_df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def count_samples_in_chromosomes(genome, window_size, step_size):
    """计算每个染色体根据窗口大小和步长生成的样本数量"""
    chrom_sample_counts = {}
    for chrom, sequence in genome.items():
        chrom_length = len(sequence)
        num_samples = (chrom_length - window_size) // step_size + 1
        chrom_sample_counts[chrom] = num_samples
    return chrom_sample_counts


# ------------------------------- Main Processing -----------------------------

def process_tag_intervals(bed_data, window_size, step_size, coverage_threshold):
    """处理BED数据，计算每个标签与窗口的重叠情况"""
    chrom_interval_length_count = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for tag_name, bed_df in bed_data.items():
        for chrom in bed_df['chrom'].unique():
            chrom_df = bed_df[bed_df['chrom'] == chrom]
            starts, ends = chrom_df['start'].values, chrom_df['end'].values
            for i in tqdm(range(len(chrom_df)), desc=f"Processing {chrom} - {tag_name}"):
                try:
                    start1, end1 = int(starts[i]), int(ends[i])
                    current_tag_length = int(end1 - start1)
                    first_window_start = (start1 // step_size) * step_size
                    current_window_start = first_window_start
                    current_window_end = current_window_start + window_size

                    while current_window_start <= end1:
                        intersection_start = max(start1, current_window_start)
                        intersection_end = min(end1, current_window_end)
                        intersection_length = max(0, intersection_end - intersection_start)

                        # 计算占用百分比
                        if current_tag_length < window_size:
                            overlap_percentage = intersection_length / current_tag_length if current_tag_length > 0 else 0
                        else:
                            overlap_percentage = intersection_length / window_size if current_tag_length > 0 else 0

                        if overlap_percentage >= coverage_threshold:
                            chrom_interval_length_count[tag_name][chrom][(current_window_start, current_window_end)] += 1
                        current_window_start += step_size
                        current_window_end = current_window_start + window_size

                except ValueError:
                    # 如果 start1 或 end1 不是数字，跳过这一行
                    print(f"Skipping invalid start or end values: start1={starts[i]}, end1={ends[i]}")
                    continue
                except Exception as e:
                    # 处理其他异常，输出调试信息
                    print(f"Error occurred while processing interval:")
                    print(f"start1: {start1}, end1: {end1}, step_size: {step_size}, window_size: {window_size}")
                    print(f"Error details: {str(e)}")
                    continue

    return chrom_interval_length_count

def process_tag_intervals_score(bed_data, window_size, step_size, coverage_threshold):
    """
    处理BED数据，计算每个标签与窗口的重叠情况
    处理的bed文件为重叠计算得分  read_bed_file_score 统计后获得
    """
    chrom_interval_length_count = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for tag_name, bed_df in bed_data.items():
        for chrom in bed_df['chrom'].unique():
            chrom_df = bed_df[bed_df['chrom'] == chrom]
            starts, ends, final_scores = chrom_df['start'].values, chrom_df['end'].values,chrom_df['score'].values
            for i in tqdm(range(len(chrom_df)), desc=f"Processing {chrom} - {tag_name}"):
                try:
                    start1, end1, final_score = int(starts[i]), int(ends[i]), float(final_scores[i])
                    current_tag_length = int(end1 - start1)
                    first_window_start = (start1 // step_size) * step_size
                    current_window_start = first_window_start
                    current_window_end = current_window_start + window_size

                    while current_window_start <= end1:
                        # 计算窗口与 BED 片段的重叠范围
                        intersection_start = max(start1, current_window_start)
                        intersection_end = min(end1, current_window_end)
                        intersection_length = max(0, intersection_end - intersection_start)

                        # 计算 BED 片段对窗口的贡献比例
                        if current_tag_length < window_size:
                            overlap_percentage = intersection_length / current_tag_length if current_tag_length > 0 else 0
                        else:
                            overlap_percentage = intersection_length / window_size if current_tag_length > 0 else 0

                        # 只有当 `overlap_percentage` 高于 `coverage_threshold` 时，才进行累加
                        if overlap_percentage >= coverage_threshold:
                            contribution_score = final_score * overlap_percentage
                            chrom_interval_length_count[tag_name][chrom][(current_window_start, current_window_end)] += contribution_score

                        # 滑动窗口
                        current_window_start += step_size
                        current_window_end = current_window_start + window_size

                except ValueError:
                    # 如果 start1 或 end1 不是数字，跳过这一行
                    print(f"Skipping invalid start or end values: start1={starts[i]}, end1={ends[i]}")
                    continue
                except Exception as e:
                    # 处理其他异常，输出调试信息
                    print(f"Error occurred while processing interval:")
                    print(f"start1: {start1}, end1: {end1}, step_size: {step_size}, window_size: {window_size}")
                    print(f"Error details: {str(e)}")
                    continue

    return chrom_interval_length_count


def merge_tag_results(chrom_interval_length_count):
    """合并不同标签的结果，计算每个窗口内不同标签的覆盖情况"""
    merged_results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for tag_name, chrom_data in tqdm(chrom_interval_length_count.items(), desc="Merging tag results"):
        for chrom, window_data in chrom_data.items():
            for (window_start, window_end), overlap_percentage in window_data.items():
                merged_results[chrom][(window_start, window_end)][tag_name] = 1
    return merged_results

from collections import defaultdict
from tqdm import tqdm


def merge_tag_results_score(chrom_interval_score):
    """
    合并不同标签的结果，并进行得分过滤和处理：
    1. 删除所有得分 < 0.1 的区段
    2. 对得分 > 1.0 的区段设为 1.0
    3. 其他得分保留 1 位小数

    参数：
    - `chrom_interval_score`: {tag_name: {chrom: {(window_start, window_end): score}}}

    返回：
    - `merged_results`: {chrom: {(window_start, window_end): {tag_name: processed_score}}}
    """
    merged_results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for tag_name, chrom_data in tqdm(chrom_interval_score.items(), desc="Merging tag results"):
        for chrom, window_data in chrom_data.items():
            for (window_start, window_end), score in window_data.items():
                # 1. 过滤掉得分 < 0.1 的区段
                if score < 0.1:
                    continue  # 直接跳过该区段

                # 2. 处理得分：
                #    - 如果得分 > 1.0，则设为 1.0
                #    - 否则，保留 1 位小数
                processed_score = min(round(score, 1), 1.0)

                # 存储处理后的结果
                merged_results[chrom][(window_start, window_end)][tag_name] = processed_score

    return merged_results


def create_fa_file_from_merged_results(merged_results, genome, output_fa_path, sequence_length):
    """根据合并后的结果创建FA文件"""
    with open(output_fa_path, 'w', encoding='utf-8') as output_file:
        for chrom, window_data in tqdm(merged_results.items(), desc="Creating FA file"):
            for (window_start, window_end), tag_data in window_data.items():
                # 获取ATCG序列
                sequence = get_sequence_from_fa(genome, chrom, window_start, window_end)
                if sequence is None:
                    continue  # 如果没有找到该染色体或区间，跳过
                if sequence and len(sequence) == sequence_length:
                    tag_score_str = ",".join([f"{tag_name}_1" for tag_name in tag_data.keys()])
                    header = f">{tag_score_str}::{chrom}:{window_start}-{window_end}"
                    output_file.write(f"{header}\n{sequence}\n")


def create_fa_file_from_merged_results_score(merged_results, genome, output_fa_path, sequence_length):
    """
    根据合并后的结果创建FA文件，保证每个窗口的FASTA头部包含 tag_name 和对应的得分。

    参数：
    - `merged_results`: {chrom: {(window_start, window_end): {tag_name: score}}}
    - `genome`: 基因组文件路径（FASTA）
    - `output_fa_path`: 输出的FA文件路径
    - `sequence_length`: 期望的序列长度

    输出：
    - 生成的FASTA文件，每个窗口的序列及其对应的 `tag_name_score`
    """
    with open(output_fa_path, 'w', encoding='utf-8') as output_file:
        for chrom, window_data in tqdm(merged_results.items(), desc="Creating FA file"):
            for (window_start, window_end), tag_data in window_data.items():
                # 获取 ATCG 序列
                sequence = get_sequence_from_fa(genome, chrom, window_start, window_end)

                if sequence is None:
                    continue  # 染色体或序列区间无效，跳过
                if sequence and len(sequence) == sequence_length:
                    # 生成 tag_score_str，其中每个 tag_name 及其得分均被保留
                    tag_score_str = ",".join([f"{tag}_{score}" for tag, score in tag_data.items()])
                    header = f">{tag_score_str}::{chrom}:{window_start}-{window_end}"

                    # 写入 FASTA 格式
                    output_file.write(f"{header}\n{sequence}\n")


def write_tag_info(tag_name_list, output_tag_path):
    """将所有标签信息写入到文件"""
    unique_tag_names = sorted(set(tag_name_list))
    with open(output_tag_path, 'w') as tag_file:
        for tag_name in unique_tag_names:
            tag_file.write(f"{tag_name}\n")


def write_tag_weight(tag_weights, output_tag_weight_path):
    """
    将 tag 权重保存到文件。

    输入：
    - tag_weights: dict, 每个 tag_name 的权重值。
    - output_tag_weight_path: str, 输出文件的路径。

    如果 tag_weights 不是字典类型，则打印错误信息并退出函数。
    """
    if not isinstance(tag_weights, dict):
        print("Error: tag_weights 必须是一个字典！")
        return  # 退出函数，防止进一步执行

    with open(output_tag_weight_path, 'w') as weight_file:
        for tag_name, weight in tag_weights.items():
            weight_file.write(f"{tag_name}\t{weight}\n")


# 计算每个 tag_name 在不同染色体上的总计数的均值并计算权重
def compute_tag_weights(merged_results):
    """
    计算每个 tag_name 在不同染色体上的总计数的均值，并基于此计算权重。

    输入：
    - merged_results: dict, 包含染色体、窗口和标签数据的字典。

    输出：
    - tag_weights: dict, 每个 tag_name 的权重值。
    """
    # 初始化用于统计的字典
    tag_count = defaultdict(lambda: defaultdict(int))

    # 遍历 merged_results 字典，统计每个 tag_name 在每个染色体上出现的次数
    for chrom, window_data in merged_results.items():
        for (window_start, window_end), tag_names in window_data.items():
            for tag_name, count in tag_names.items():
                # 增加计数
                tag_count[chrom][tag_name] += 1

    # 初始化字典用于存储每个 tag_name 在所有染色体上的计数总和
    tag_total_count = defaultdict(int)

    # 初始化字典用于存储每个 tag_name 出现的染色体数量
    tag_chrom_count = defaultdict(int)

    # 遍历 tag_count 字典，统计每个 tag_name 在所有染色体上的计数总和和染色体数量
    for chrom, tag_data in tag_count.items():
        for tag_name, count in tag_data.items():
            tag_total_count[tag_name] += count
            tag_chrom_count[tag_name] += 1

    # 计算每个 tag_name 在不同染色体上的总计数的平均值
    tag_average_count = {}

    for tag_name in tag_total_count:
        total_count = tag_total_count[tag_name]
        chrom_count = tag_chrom_count[tag_name]
        # 计算均值
        tag_average_count[tag_name] = total_count / chrom_count

    # 计算每个 tag_name 的权重值
    total_counts = sum(tag_average_count.values())  # 所有 tag_name 的计数总和

    tag_weights = {}
    for tag_name, avg_count in tag_average_count.items():
        # 计算权重值
        tag_weights[tag_name] = total_counts / avg_count

    return tag_weights

def process_merged_results(merged_results, proportion_threshold=0.001, filter_log_path=None):
    """
    处理 merged_results，统计每个染色体中的窗口数量、每个 tag_name 的出现次数，
    并计算每个 tag_name 的比例，过滤比例小于指定阈值的标签。

    输入：
    - merged_results: dict, 染色体及其窗口信息，格式为 {chrom: {window_start_end: {tag_name: count}}}
    - proportion_threshold: float, 过滤标签时使用的比例阈值，小于此比例的标签会被移除（默认0.001）
    - filter_log_path: str, 用于保存过滤结果的日志文件路径，若不需要日志则传入 None

    输出：
    - chrom_window_counts: dict, 每个染色体的窗口数量，格式为 {chrom: window_count}
    - total_window_count: int, 总窗口数量
    - filtered_tag_counts: dict, 过滤后的 tag_name 及其比例，格式为 {tag_name: proportion}
    - removed_tag_counts: dict, 被移除的 tag_name 及其比例，格式为 {tag_name: proportion}
    """
    chrom_window_counts = {}
    total_window_count = 0
    tag_counts = defaultdict(int)  # 用来统计每个 tag_name 出现的次数

    # 遍历 merged_results，统计窗口数量和每个 tag_name 的出现次数
    for chrom, window_data in merged_results.items():
        if chrom.isdigit():  # 判断染色体名是否为纯数字
            chrom_window_count = len(window_data)  # 染色体中窗口的数量
            chrom_window_counts[chrom] = chrom_window_count
            total_window_count += chrom_window_count

            # 统计每个 tag_name 出现的次数
            for window_start_end, tag_data in window_data.items():
                for tag_name in tag_data:
                    tag_counts[tag_name] += 1

    print(f"Total Window Count: {total_window_count}")

    # 计算每个 tag_name 的比例，并过滤占比小于 proportion_threshold 的部分
    filtered_tag_counts = {}
    removed_tag_counts = {}
    filtered_tags = []
    removed_tags = []

    for tag_name, count in tag_counts.items():
        proportion = count / total_window_count
        if proportion >= proportion_threshold:
            filtered_tags.append(tag_name)
            filtered_tag_counts[tag_name] = proportion
        else:
            removed_tags.append(tag_name)
            removed_tag_counts[tag_name] = proportion

    # 输出过滤后的标签信息到日志文件（如果提供了 filter_log_path）
    if filter_log_path:
        with open(filter_log_path, 'w') as output_file:
            output_file.write(f"Tags with proportions >= {proportion_threshold}:\n")
            for tag_name, proportion in filtered_tag_counts.items():
                output_file.write(f"{tag_name}: {proportion:.4f}\n")

            output_file.write(f"\nTags removed (proportion < {proportion_threshold}):\n")
            for tag_name, proportion in removed_tag_counts.items():
                output_file.write(f"{tag_name}: {proportion:.4f}\n")

    # 创建一个新的 merged_results 字典，去除每个窗口中存在的 removed_tags
    new_merged_results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for chrom, window_data in merged_results.items():
        if chrom.isdigit():  # 仅处理纯数字染色体
            for window_start_end, tag_data in window_data.items():
                # 新的标签数据，只保留未被移除的标签
                new_tag_data = {tag_name: count for tag_name, count in tag_data.items() if
                                tag_name not in removed_tags}
                if new_tag_data:  # 仅当有标签剩下时才保留该窗口
                    new_merged_results[chrom][window_start_end] = new_tag_data

    return new_merged_results, chrom_window_counts, total_window_count, filtered_tag_counts, removed_tag_counts


# --------------------------- Main Script Execution ---------------------------

window_size = 1024
step_size = 512
coverage_threshold = 0.5

# 服务器内数据地址为：
# /public/workspace/lvtongxuan/data/test_chiphub/merged_tag 登录后查看，本地地址为如下，自行切换
# 配置路径和参数
bed_folder = 'D:\\data\\tag_file\\merged_tag_bedtools\\zma'
genome_fa_path = 'D:\\data\\tag_file\\fa\\zma\\zea_mays.fa'
output_fa_path = f'E:\\SeiPlant_Supplement_materials\\tag_file\\training_fa_bedtools_seiplant_supplement_materials\\zma_{window_size}_{step_size}.fa'
output_tag_path = 'E:\\SeiPlant_Supplement_materials\\tag_file\\training_fa_bedtools_seiplant_supplement_materials\\tag_zma.txt'
output_weight_path = "E:\\SeiPlant_Supplement_materials\\tag_file\\training_fa_bedtools_seiplant_supplement_materials\\weight_tag_zma.txt"

# 读取所有BED文件并处理
bed_files = {os.path.splitext(file)[0]: os.path.join(bed_folder, file) for file in os.listdir(bed_folder) if file.endswith('.bed')}
bed_data = {name: read_bed_file(path) for name, path in bed_files.items()}

# 处理标签区间
chrom_interval_length_count = process_tag_intervals(bed_data, window_size, step_size, coverage_threshold)

# 合并结果
merged_results = merge_tag_results(chrom_interval_length_count)

# merged_results, chrom_window_counts, total_window_count, filtered_tag_counts, removed_tag_counts = process_merged_results(
#     merged_results, proportion_threshold=0.001, filter_log_path='D:\\data\\tag_file\\training_fa_bedtools_filtered\\osa_filter_log.txt'
# )

tag_weights = compute_tag_weights(merged_results)

# 读取基因组文件
genome = read_fa(genome_fa_path)

# 创建FA文件
create_fa_file_from_merged_results(merged_results, genome, output_fa_path, sequence_length=window_size)

# 提取标签信息并写入
tag_name_list = [tag for tag in bed_data.keys()]
write_tag_info(tag_name_list, output_tag_path)
write_tag_weight(tag_weights, output_weight_path)
print('Merged and Processing completed successfully.')
