# SeiPlant
**SeiPlant** is a deep learning framework for predicting histone modification patterns in plant genomes. Built upon 
the Sei architecture, this model enables high-resolution inference of chromatin states directly from raw DNA sequences 
across diverse plant species.

## Schematic Diagram

<div style="text-align: center;">
    <img src="img/Fig1.jpg" alt="fig1" width="1000" height="500">
</div>

Figure 1. Workflow of the SeiPlant framework for cross-species prediction of chromatin features in plants.

## Key Features
- Cross-species modeling for plant epigenomics
- Multi-task prediction of histone marks (e.g., H3K4me3, H3K27ac)
- Tested on representative monocots and dicots (e.g., *Oryza sativa*, *Zea mays*, *Arabidopsis thaliana*)
- Supports both species-specific and generalization settings
- One-click sequence-to-signal pipeline outputting BigWig and BedGraph

## Quick Start
### Configure the operating environment

```bash
### Python enviroment constructed by Conda
conda create -n SeiPlant python=3.8
conda activate SeiPlant
git clone https://github.com/Lv-BioInfo/SeiPlant.git
pip install -r requirements.txt

# Install PyTorch (adjust the version according to your system environment)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html
```
> **Note**: In our experiments, we used **PyTorch 2.1.2 with CUDA 11.8**.  
> This specific version was chosen because our server’s **GLIBC version** was low to support latest PyTorch releases.  
> Please install the latest PyTorch version compatible with your own system environment 
> (see [PyTorch official installation guide](https://pytorch.org/get-started/locally/)).

### Download the corresponding file to the specified folder.
The SeiPlant project requires some files to be **manually downloaded from Zenodo** and placed into the correct folders. 
Below is the directory structure with notes on which files you need to provide:

- **models/**
  - **model_architectures/**
    - `model.py` — Model architecture definition
  - `model.pth` — **[Download from Zenodo]**
  - `tag` — **[Download from Zenodo]**

- **scripts/**
  - **fasta/**
    - `species.fa` — **[Download from Zenodo]**
    - `species.size` — **[Download from Zenodo]**
  - `make_bedgraph.py` — Convert bigWig to bedGraph
  - `make_prediction_bed.py` — Run predictions in bed format
  - `prediction.py` — Inference script
  - `train.py` — Training script
  - `evaluate.py` — Evaluation script

- **utils/**
  - Utility functions for data processing & model training

> **Note**  
> You can download the **sample reference genomes** and **trained model parameters** from  
> 👉 [Zenodo (DOI: 10.5281/zenodo.15421964)](https://doi.org/10.5281/zenodo.15421964)  
> and place them in the **`/scripts/fasta/`** and **`/models/`** folders, respectively.

### Step 1: Prepare FASTA Input and Generate Genomic Windows

Provide a reference genome in **FASTA** format for the species of interest. To tile the genome:

- Apply a **sliding window** approach (default: 1,024 bp window, 128 bp step size)
- Filter windows to retain only sequences with **standard nucleotides** (A/T/C/G)
- Save:
  - **BED** file for genomic coordinates
  - **FASTA** file for model input sequences

Example usage:
```bash
python make_prediction_bed.py \ 
  --fasta fasta/arabidopsis_thaliana.fa \
  --size fasta/arabidopsis_thaliana.size \
  --species arabidopsis_thaliana \
  --output_path ./bed/ \
  --window_size 1024 \
  --step_size 128
```

---

### Step 2: Run Prediction Using Pretrained SeiPlant Model

Feed the `.fasta` file into the pretrained **SeiPlant** model to obtain chromatin feature predictions.

- Predicts probability scores for multiple histone modifications:
  - **H3K4ME3**, **H3K27AC**, **H3K4ME1**, **H3K9AC**, **H3K36ME3**
- Output: `.npy` file containing **multi-label** prediction scores aligned with each genomic window

Post-process model predictions into standard genome browser formats:

1. **Align scores** to central genomic coordinates (e.g., `start+448`, `start+576`)
2. **Filter** weak signals (< 0.01) and **normalize** (Min–Max scaling to 0.1–1.0)
3. Export **per-mark BedGraph files**

Example usage:
```bash
python prediction.py --model_path ../models/Brassicaceae_20250312_203749_1024_nip_feature7.model \
  --model_tag_file ../models/histone_modification_tag.txt \
  --species arabidopsis_thaliana \
  --fa_path ./bed/arabidopsis_thaliana_1024_128.fa \
  --output_dir ./bedgraph \
  --bed_file ./bed/arabidopsis_thaliana_1024_128_filtered.bed \
  --seq_len 1024 \
  --batch_size 256
```
---

### Step 3: Exchange Signal Files (BedGraph & BigWig)

1. Prepare your **BedGraph** file (e.g., `H3K4ME3.bedgraph`).
2. Make sure you have the chromosome sizes file (e.g., `chrom.sizes`).
3. Install **UCSC tools** (provides `bedGraphToBigWig`).
4. Convert to **BigWig** format:

```bash
bedGraphToBigWig H3K4ME3.bedgraph chrom.sizes H3K4ME3.bw
```
> **Note**  
> **bedGraphToBigWig** is part of the **UCSC utilities**.  
> 📌 You can download it from [UCSC Genome Browser utilities](http://hgdownload.soe.ucsc.edu/admin/exe/).  
> Make sure the **`chrom.sizes`** file matches the reference genome you are using.

### Train from Scratch Guide Provided
We provide a complete from-scratch training guide used in this study, including data preparation, scoring criteria, and training procedures.  
For details, please refer to: [`train_from_scratch`](docs/train_from_scratch.md)

### Ablation Study Used
For specific details on the ablation experiment, please visit the following files in the `experiments/ablation` directory:
[`ablation`](experiments/ablation/ablation.md)   

### Compare Methods Study Used
For specific details on the compare methods experiment, please visit the following files in the `experiments/comparative_methods` directory:
[`comparative_methods`](experiments/comparative_methods/compare_methods.md)  

### Plotting Code Provided

For specific details on the plotting methods used in our study, please visit the following files in the `docs/plotting` directory:
[`plotting`](docs/plotting/code/prediction_remake.ipynb)

## Citation
If you use `SeiPlant` in your work, please cite:

> **Lv T, Han Q, Li Y, Liang C, Ruan Z, Chao H, Chen M, Chen D.**
> Cross-species prediction of histone modifications in plants via deep learning.
> *Genome Biology* (2026). https://doi.org/10.1186/s13059-025-03929-4

We also welcome citation of related studies:

> **A sequence-based global map of regulatory activity for deciphering human genetics**  
> Chen KM, Wong AK, Troyanskaya OG, Zhou J  
> *Nature Genetics*. 2022; 54:940–949. doi: [https://doi.org/10.1038/s41588-022-01102-2](https://doi.org/10.1038/s41588-022-01102-2)  

> **Deep learning on chromatin profiles reveals the cis-regulatory sequence code of the rice genome**  
> Zhou X, Ruan Z, Zhang C, Kaufmann K, Chen D  
> *Journal of Genetics and Genomics*. 2024; S1673852724003564. doi: [https://doi.org/10.1016/j.jgg.2024.12.007](https://doi.org/10.1016/j.jgg.2024.12.007)  

## Contact
Any questions or suggestions on SeiPlant are welcomed! Please report it on issues, or contact Dijun Chen (dijunchen@nju.edu.cn).