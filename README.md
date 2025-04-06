# scDoRI: Single-cell Deep Multi-Omic Regulatory Inference

![scDoRI Schematic](docs/_static/scdori_schematic_main.png)

**scDoRI** is a deep learning framework for single-cell **multiome** data (RNA + ATAC) that infers **enhancer-mediated gene regulatory networks (eGRNs)**. By combining an **encoder–decoder** approach with mechanistic constraints (enhancer–gene links, TF binding logic), scDoRI learns **topics** that group co-accessible peaks, their cis-linked genes and upstream activator and repressor TFs – all while scaling to large datasets via mini-batches.

## 🚀 Highlights
- 🔄 **Unified** approach: single model for dimensionality reduction + eGRN inference
- 🧠 Learns **topics** that represent cell-state-specific regulatory programs
- 🧬**Continuous eGRN modelling** : each cell is a mixture of topics, allowing study of changes in GRNs. No need for predefined clusters  
- 🧰 **Scalable** to large datasets via **mini-batch training**

  
## 📦 Installation 

Clone the repo and create the conda environment:

```bash
git clone https://github.com/saraswatmanu/scDoRI.git
cd scDoRI
conda env create -f environment.yml
conda activate scdori-env
```

## ⚙️ Usage
You’ll work through two notebooks, using two separate config files to set parameters for your dataset preprocessing and training.
### 🧹 Step 1: Preprocessing
#### Edit paths and parameters in:
```bash
preprocessing_pipeline/config.py
```
to specify location of RNA and ATAC anndata .h5ad files, motif file, and set number of peaks/genes/TFs to train on. 
#### Run preprocessing notebook
```bash
notebooks/preprocessing.ipynb
```
### 🧠 Step 2: Training and Downstream analysis

#### Edit paths and parameters in:
```bash
scdori/config.py
```
for scDoRI hyperparameters (number of topics, learning rate, epochs etc.) and specify path for preprocessed anndata objects and insilico-chipseq files
#### Run training and downstream analysis notebook
```bash
notebooks/training.ipynb
```
## 📚 Documentation
📖 Full documentation and API reference is hosted at:

👉 https://saraswatmanu.github.io/scDoRI/

Includes:
API reference (docstrings)
(upcoming) In-depth method overview
(upcoming) Preprocessing + training guides
(upcoming) Customization tips

## 📣 Citation
If you use scDoRI in your work, please cite our preprint/paper (coming soon).
Until then, feel free to open an issue or get in touch at manu.saraswat@dkfz.de
