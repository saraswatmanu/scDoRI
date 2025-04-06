# scDoRI: Single-cell Deep Multi-Omic Regulatory Inference

![scDoRI Schematic](docs/_static/scdori_schematic_main.png)

**scDoRI** is a deep learning framework for single-cell **multiome** data (RNA + ATAC) that infers **enhancer-mediated gene regulatory networks (eGRNs)**. By combining an **encoder–decoder** approach with mechanistic constraints (enhancer–gene links, TF binding logic), scDoRI learns **topics** that group co-accessible peaks, their cis-linked genes and upstream activator and repressor TFs – all while scaling to large datasets via mini-batches.

## Key Highlights

- **Unified** approach: single model for dimensionality reduction + eGRN inference
- **Continuous eGRN modelling** : each cell is a mixture of topics, allowing study of changes in GRNs. No need for predefined clusters  
- **Scalable**: mini-batch training for millions of cells
  
## 📦 Installation 


### 1) Clone this repo
```bash
git clone https://github.com/saraswatmanu/scDoRI.git
cd scDoRI
```

### 2) Create conda environment
```bash
conda env create -f environment.yml
conda activate scdori-env
```

## Usage
### 1) Edit config files
#### First edit preprocessing_pipeline/config.py 
to specify location of RNA and ATAC anndata .h5ad files, motif file, and set number of peaks/genes/TFs to train on. - scdori/config.py for scDoRI hyperparameters (number of topics, learning rate, epochs etc.)
#### Second edit scdori/config.py
for scDoRI hyperparameters (number of topics, learning rate, epochs etc.)

### 2) Run Notebooks
first run notebooks/preprocessing.ipynb, then notebooks/training.ipynb
