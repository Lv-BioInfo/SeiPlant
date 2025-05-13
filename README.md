# SeiPlant
Cross-species prediction of histone modifications in plants using the Sei deep learning architecture.

## Introduction
Single-cell multi-omics integration enables joint analysis at the single-cell level of resolution to provide more 
accurate understanding of complex biological systems, while spatial multi-omics integration is benefit to the 
exploration of cell spatial heterogeneity to facilitate more comprehensive downstream analyses. Existing methods are 
mainly designed for single-cell multi-omics data with little consideration of spatial information, and still have room 
for performance improvement. A reliable multi-omics integration method designed for both single-cell and spatially 
resolved data is necessary and significant. We propose a multi-omics integration method based on dual-path graph 
attention auto-encoder (SSGATE). It can construct the neighborhood graphs based on single-cell expression profiles or 
spatial coordinates, enabling it to process single-cell data and utilize spatial information from spatially resolved 
data. It can also perform self-supervised learning for integration through the graph attention auto-encoders from two 
paths. SSGATE is applied to integration of transcriptomics and proteomics, including single-cell and spatially 
resolved data of various tissues from different sequencing technologies. SSGATE shows better performance and stronger 
robustness than competitive methods and facilitates downstream analysis.