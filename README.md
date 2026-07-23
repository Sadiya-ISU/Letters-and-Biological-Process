# Letters and Biological Process

This repository contains a Jupyter Notebook-based analysis of the Gene Ontology (GO) biological process Directed Acyclic Graph (DAG), with a focus on term depth, graph structure, and text-based clustering patterns.

## Repository Contents

- `Count the DAG_Final.ipynb` — primary end-to-end analysis notebook
- `go_lingdepth_all_in_one.ipynb` — consolidated analysis notebook variant
- `go_lingdepth/` — reusable Python modules for OBO parsing, depth analysis, plotting, clustering, and entropy/statistical workflows
- `go-basic.obo` — Gene Ontology source file used by the notebooks
- `go-basic.zip` — compressed copy of `go-basic.obo`
- `emb_BP.npy` — precomputed biological process embeddings used in analysis steps
- `LICENSE` — project license

## Scope

The notebook performs:
- GO OBO parsing and DAG traversal
- node/level count analysis across depth
- statistical checks (correlation, non-parametric tests)
- visualization (line/bar/box/violin/scatter-style plots)
- semantic embedding and clustering of GO term names
- entropy-focused analysis across DAG depth

## How to Clone

```bash
git clone https://github.com/Sadiya-ISU/Letters-and-Biological-Process.git
cd Letters-and-Biological-Process
```

## Software Requirements

- Python 3.8+ (3.9/3.10 recommended)
- Jupyter Notebook or JupyterLab

## Python Dependencies

Install the libraries used in the notebook:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn networkx tqdm sentence-transformers umap-learn jupyter
```

## Data Preparation

`go-basic.obo` is already included in the repository root.

If needed, you can regenerate it from the archive:

```bash
unzip -o go-basic.zip
```

## How to Run

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```
2. Open `Count the DAG_Final.ipynb`.
3. Run cells in order from top to bottom.

## Expected Outcome

When executed successfully, the notebook should:
- parse GO terms and build parent/child DAG relationships
- generate depth/level summary tables and counts
- create multiple plots showing count distributions and trends by level
- compute correlation/statistical summaries
- produce clustered views of GO terms using embeddings + dimensionality reduction
- print cluster samples and entropy-related results

Outputs are shown directly in notebook cells (tables, plots, and printed metrics).

## Notes

- First run may take longer because embedding/clustering steps can be computationally heavy.
- `sentence-transformers` may download model files on first use.
- There is no standalone CLI script in this repository; the notebook is the primary execution entry point.
