# Quantum-state-tomography

<<<<<<< ours
Libraries : 
Qutip
Quantum State Tomography (QST) seeks to reconstruct an unknown quantum state from measurement data. This repository explores QST with classical and quantum-inspired machine-learning methods, comparing approaches such as variational circuits, kernel methods, and supervised learning baselines.

https://www.overleaf.com/6514264713mxcfbqkffgsx#2bd943
=======
Quantum State Tomography (QST) seeks to reconstruct an unknown quantum state from measurement data. This repository explores QST with classical and quantum-inspired machine-learning methods, comparing approaches such as variational circuits, kernel methods, and supervised learning baselines.

>>>>>>> theirs
## Repository layout
- **`depot_final_propre/`** — main project code and experiments. Explore these notebooks to see the implemented pipelines and results (Maximum Likelihood Estimation, variational quantum circuits with DNN heads, regression baselines, classification workflows, and quantum kernels).
  - `notebook_MLE`
  - `notebook_VQC_DNN`
  - `notebook_regression`
  - `notebook_classification`
  - `notebook_Quantum_kernel`
- **`reports/`** — final documents and references (`Definitions.pdf`, `Quantum_computing.pdf`, `report_ML.pdf`).
- **`ML_quantum_classification.ipynb`** — supplementary classification experiments and earlier explorations.
- **`requirements.txt`** — minimal Python dependencies for running the notebooks.
- **`Consignes projet`** — project notes and guidelines.

## Getting started
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks in `depot_final_propre/` with JupyterLab or VS Code and run the cells to reproduce experiments or adapt them to new datasets.

## Documentation
Consult the PDFs in `reports/` for definitions, background on quantum computing, and the final machine-learning report that summarizes experimental findings.
