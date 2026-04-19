# Letters and Biological Process

This repository contains a Jupyter Notebook-based analysis of the Gene Ontology (GO) biological process Directed Acyclic Graph (DAG), with a focus on term depth, graph structure, and text-based clustering patterns.

## Repository Contents

- `Count the DAG_Final.ipynb` — main analysis notebook
- `go-basic.zip` — compressed GO ontology file (`go-basic.obo`)
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

The notebook expects `go-basic.obo` in the repository root. Extract it from `go-basic.zip`:

```bash
unzip go-basic.zip
```

After extraction, ensure this file exists:

- `./go-basic.obo`

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
