# File: src/embedding.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE # Import others later as needed
# from diffusion_maps import DiffusionMap # Example if using diffusion_maps library
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
# Also ensure other necessary imports like numpy, PCA, MDS, LLE, TSNE are there

def run_pca(data, n_components=2, **kwargs):
    """
    Runs PCA on the data.

    Args:
        data (np.ndarray): Input data (n_samples, n_features).
        n_components (int): Number of principal components to keep.
        **kwargs: Additional arguments for sklearn.decomposition.PCA.

    Returns:
        np.ndarray: The embedded data (n_samples, n_components).
        sklearn.decomposition.PCA: The fitted PCA object.
    """
    print(f"Running PCA with n_components={n_components}...")
    pca = PCA(n_components=n_components, **kwargs)
    embedding = pca.fit_transform(data)
    print("PCA completed.")
    return embedding, pca

def run_mds(data, n_components=2, random_state=42, n_init=4, max_iter=300, n_jobs=-1, **kwargs):
    """
    Runs MDS (Multidimensional Scaling) on the data.

    Args:
        data (np.ndarray): Input data (n_samples, n_features).
                           Note: MDS complexity is high, works best with fewer samples or
                           precomputed distance matrix. sklearn uses SMACOF algorithm.
        n_components (int): Number of dimensions for the embedding.
        random_state (int): Seed for reproducibility.
        n_init (int): Number of times the SMACOF algorithm will be run with different initializations.
        max_iter (int): Maximum number of iterations of the SMACOF algorithm for a single run.
        n_jobs (int): The number of jobs to use for the computation (-1 means all processors).
        **kwargs: Additional arguments for sklearn.manifold.MDS.

    Returns:
        np.ndarray: The embedded data (n_samples, n_components).
        sklearn.manifold.MDS: The fitted MDS object.
    """
    print(f"Running MDS with n_components={n_components}...")
    # Using metric=True for classical MDS approximation if possible,
    # but default metric=True, nonmetric=False uses SMACOF on distances.
    # For high-dimensional data, calculating the full distance matrix can be slow/memory intensive.
    # Sklearn MDS calculates the distance matrix internally if data is passed.
    mds = MDS(n_components=n_components,
              random_state=random_state,
              n_init=n_init,
              max_iter=max_iter,
              n_jobs=n_jobs,
              dissimilarity='euclidean', # Use Euclidean distance between feature vectors
              verbose=1, # Print progress
              **kwargs)
    embedding = mds.fit_transform(data)
    print("MDS completed.")
    return embedding, mds

# File: src/embedding.py (Add these functions)
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.preprocessing import StandardScaler # Ensure this is imported
# For Diffusion Map using scikit-learn-extra (diffusion_maps was in reqs, but let's use this first)
# If you installed scikit-learn-extra: pip install scikit-learn-extra
try:
    from skextra.diffusion_map import DiffusionMap
    use_skextra_dmap = True
except ImportError:
    print("Warning: scikit-learn-extra not found. DiffusionMap function will not work.")
    print("Install it via: pip install scikit-learn-extra")
    use_skextra_dmap = False
    # Define a placeholder if needed, or handle the error in the notebook
    DiffusionMap = None


def run_isomap(data, n_components=2, n_neighbors=5, scale=True, **kwargs):
    """ Runs Isomap. """
    print(f"Running Isomap with n_components={n_components}, n_neighbors={n_neighbors}...")
    if scale:
        data = StandardScaler().fit_transform(data)
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
    embedding = isomap.fit_transform(data)
    print("Isomap completed.")
    return embedding, isomap

def run_lle(data, n_components=2, n_neighbors=5, method='standard', scale=True, random_state=42, **kwargs):
    """ Runs LLE and its variants. """
    print(f"Running LLE (method='{method}') with n_components={n_components}, n_neighbors={n_neighbors}...")
    if scale:
        data = StandardScaler().fit_transform(data)
    # Note: Increased default max_iter for potentially better convergence
    lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors,
                                method=method, random_state=random_state, max_iter=500, n_jobs=-1, **kwargs)
    embedding = lle.fit_transform(data)
    print(f"LLE (method='{method}') completed.")
    return embedding, lle

def run_tsne(data, n_components=2, perplexity=10, scale=True, random_state=42, **kwargs):
    """ Runs t-SNE. """
    print(f"Running t-SNE with n_components={n_components}, perplexity={perplexity}...")
    if scale:
        # t-SNE is sensitive to scale, but often applied to raw or PCA-reduced data.
        # Let's scale here for consistency, but be aware this might affect results.
        data = StandardScaler().fit_transform(data)

    # Common practice: Apply PCA first for high-dimensional data before t-SNE
    # pca_tsne = PCA(n_components=50, random_state=random_state)
    # data_pca = pca_tsne.fit_transform(data)
    # print("Applied PCA pre-processing for t-SNE.")
    # Use data directly for now, as n_samples is small
    data_pca = data

    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, verbose=1, n_jobs=-1, **kwargs)
    embedding = tsne.fit_transform(data_pca)
    print("t-SNE completed.")
    return embedding, tsne

# Diffusion Map using scikit-learn-extra (recommended implementation)
def run_diffmap_skextra(data, n_components=2, n_neighbors=5, alpha=0.5, scale=True, **kwargs):
    """ Runs Diffusion Map using scikit-learn-extra. """
    if not use_skextra_dmap:
         print("Error: Diffusion Map (skextra) cannot run. Library not found.")
         return None, None

    print(f"Running Diffusion Map (skextra) with n_components={n_components}, n_neighbors={n_neighbors}, alpha={alpha}...")
    if scale:
         data = StandardScaler().fit_transform(data)

    # DiffusionMap in skextra uses n_components for the number of eigenvectors *including* the trivial one
    diffmap = DiffusionMap(n_components=n_components + 1, # Request n+1 eigenvectors
                              k=n_neighbors,
                              alpha=alpha,
                              **kwargs)
    embedding = diffmap.fit_transform(data)
    print("Diffusion Map (skextra) completed.")
    # Return components excluding the trivial first one (constant eigenvector)
    # Also, typically scale by eigenvalues, but skextra might return them directly or handle scaling.
    # Check the documentation or output shape/values if needed. Let's return non-trivial components for now.
    return embedding[:, 1:], diffmap

# File: src/embedding.py (Add this new function)

def run_spectral(data, n_components=2, n_neighbors=5, scale=True, random_state=42, **kwargs):
    """ Runs Spectral Embedding. """
    print(f"Running Spectral Embedding with n_components={n_components}, n_neighbors={n_neighbors}...")
    if scale:
        data = StandardScaler().fit_transform(data)

    # Use affinity='nearest_neighbors' for consistency
    spectral = SpectralEmbedding(n_components=n_components,
                                 affinity='nearest_neighbors',
                                 n_neighbors=n_neighbors,
                                 random_state=random_state,
                                 n_jobs=-1,
                                 **kwargs)
    embedding = spectral.fit_transform(data)
    print("Spectral Embedding completed.")
    return embedding, spectral
