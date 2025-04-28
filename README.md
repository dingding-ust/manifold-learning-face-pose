# Comparative Analysis of Manifold Learning Methods for Face Pose Ordering

## Project Overview

This project investigates and compares the effectiveness of nine different dimensionality reduction methods in recovering the 1D rotational pose manifold from a dataset of 33 grayscale face images of the same individual. The goal is to reconstruct the underlying manifold and evaluate how well each method orders the face poses according to rotation angle.

The methods compared include:
* Principal Component Analysis (PCA)
* Multidimensional Scaling (MDS)
* Isomap
* Locally Linear Embedding (LLE)
* Local Tangent Space Alignment (LTSA)
* Modified LLE (MLLE)
* Hessian LLE (HLLE)
* Spectral Embedding
* t-Distributed Stochastic Neighbor Embedding (t-SNE)

## Project Structure

```
Project/
├── data/                  # (Likely contains face.mat dataset, downloaded by io_utils.py)
├── figs/                  # (Contains exported figures like sample faces, embeddings, metric plots)
├── notebooks/             # Jupyter notebooks for analysis workflow
│   ├── 01_data_overview.ipynb  # Load, visualize, and understand the dataset
│   ├── 02_baseline_PCA_MDS.ipynb # Run and evaluate PCA and MDS
│   ├── 03_manifold_methods.ipynb # Run and evaluate manifold learning methods
│   ├── 04_parameter_tuning.ipynb # Tune hyperparameters (k, perplexity)
│   └── 05_fig_export.ipynb       # Generate final figures for the report
│
├── report/                # LaTeX report files
│   ├── main.tex           # Main LaTeX source file
│   ├── main.pdf           # Compiled PDF report
│   ├── neurips_2019.sty   # LaTeX style file
│   ├── references.bib     # Bibliography file
│   └── ...                # Other potential report files (logs, structure doc)
│
├── src/                   # Source code modules
│   ├── __init__.py        # Corrected filename
│   ├── embedding.py       # Functions for dimensionality reduction algorithms
│   ├── io_utils.py        # Data loading and utility functions
│   └── metrics.py         # Functions for evaluation metrics (TAE, T, C)
│
├── .DS_Store              # macOS system file (should be in .gitignore)
├── .gitignore             # Git ignore file
├── LICENSE                # License file
├── README.md              # This README file
└── requirements.txt       # Project dependencies
```

## Dataset

The project uses a dataset of 33 grayscale face images (92x112 pixels) of a single individual viewed from different angles. The data is loaded from `face.mat` using functions in `src/io_utils.py`.

## Methodology

1.  **Data Loading & Preprocessing**: Images are loaded, flattened into vectors (10304 features), and potentially scaled.
2.  **Ground Truth**: A ground truth sequence representing the correct rotational order is established.
3.  **Embedding**: Various dimensionality reduction methods are applied to project the high-dimensional image data into 2D space.
4.  **Parameter Tuning**: Hyperparameters like the number of neighbors (k) and perplexity (p) are tuned for relevant methods.
5.  **Evaluation**: Embeddings are evaluated quantitatively using:
    * Total Absolute Error (TAE) for pose ordering accuracy.
    * Trustworthiness (T) and Continuity (C) for local structure preservation.
6.  **Visualization**: 2D embeddings are visualized using scatter plots and by plotting the actual face images at their embedded coordinates.

## Setup and Usage

1.  **Clone the repository.**
2.  **Create a virtual environment (recommended).**
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some functions might require `scikit-learn-extra`. Install via `pip install scikit-learn-extra` if needed.*
4.  **Run the Jupyter notebooks (`notebooks/`) sequentially** to reproduce the analysis. The `src/io_utils.py` script will attempt to download the `face.mat` dataset if it's not found in the `data/` directory.

## Results Summary (from Report Abstract)

Isomap (k=5) achieved the best ordering performance (TAE=6), followed closely by LLE (k=5, TAE=14) and Spectral Embedding (k=5, TAE=14). Most methods showed high Trustworthiness and Continuity. t-SNE and MDS performed poorly on this specific ordering task. Parameter selection significantly impacts performance.