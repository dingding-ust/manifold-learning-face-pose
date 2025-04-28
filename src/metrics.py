# File: src/metrics.py
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def check_pairwise_distances(X):
    """Checks if the distance matrix has non-negative values."""
    D = pairwise_distances(X)
    if np.any(D < 0.0):
        print("Warning: Negative distances found!")
        D[D < 0] = 0 # Correct negative distances to zero
    return D

def trustworthiness(X_high, X_low, n_neighbors=5):
    """
    Computes Trustworthiness score.
    Higher is better (max 1.0). Measures the extent to which the local structure
    is retained in the low-dimensional embedding (few points that are far apart
    in high-dim are brought together in low-dim).
    """
    n_samples = X_high.shape[0]
    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be less than n_samples")

    # Calculate distances ensuring non-negativity
    dist_high = check_pairwise_distances(X_high)
    dist_low = check_pairwise_distances(X_low)

    # Find nearest neighbors in high and low dimensions (excluding self)
    neigh_high = np.argsort(dist_high, axis=1)[:, 1:n_neighbors + 1]
    neigh_low = np.argsort(dist_low, axis=1)[:, 1:n_neighbors + 1]

    rank_low = np.zeros((n_samples, n_samples), dtype=int)
    # Compute ranks in the low-dimensional space based on distances
    ranks_low_matrix = np.argsort(np.argsort(dist_low, axis=1), axis=1)
    # Set rank of self to 0, adjust others
    for i in range(n_samples):
        rank_low[i] = ranks_low_matrix[i]

    # Compute Trustworthiness score
    t = 0.0
    for i in range(n_samples):
        # Find points that are in the K-neighborhood in low-dim BUT NOT in high-dim
        low_neighbors = set(neigh_low[i])
        high_neighbors = set(neigh_high[i])
        intruders = low_neighbors - high_neighbors
        if intruders:
            for j in intruders:
                # Sum ranks of intruders in high-dim space (based on low-dim ordering)
                # Find rank of j in high-dim ordering of i
                rank_high_j = np.where(np.argsort(dist_high[i]) == j)[0][0]
                if rank_high_j > n_neighbors: # Check if it's truly an intruder rank-wise
                     t += (rank_high_j - n_neighbors)


    # Normalization constant
    # Note: Sklearn's normalization constant might differ slightly.
    # Using the formula from the original paper by Venna & Kaski (2001).
    if n_samples <= n_neighbors: # Avoid division by zero or negative normalization
         return 1.0 # Perfect score if K >= N-1

    C = 2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))

    trustworthiness = 1.0 - C * t
    return max(0.0, trustworthiness) # Ensure score is not negative due to float precision

def continuity(X_high, X_low, n_neighbors=5):
    """
    Computes Continuity score.
    Higher is better (max 1.0). Measures the extent to which the K-neighborhoods
    are preserved from high-dim to low-dim (few points that are close in high-dim
    are moved far apart in low-dim).
    """
    n_samples = X_high.shape[0]
    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be less than n_samples")

    # Calculate distances ensuring non-negativity
    dist_high = check_pairwise_distances(X_high)
    dist_low = check_pairwise_distances(X_low)

    # Find nearest neighbors in high and low dimensions (excluding self)
    neigh_high = np.argsort(dist_high, axis=1)[:, 1:n_neighbors + 1]
    neigh_low = np.argsort(dist_low, axis=1)[:, 1:n_neighbors + 1]

    rank_high = np.zeros((n_samples, n_samples), dtype=int)
    # Compute ranks in the high-dimensional space based on distances
    ranks_high_matrix = np.argsort(np.argsort(dist_high, axis=1), axis=1)
     # Set rank of self to 0, adjust others
    for i in range(n_samples):
        rank_high[i] = ranks_high_matrix[i]


    # Compute Continuity score
    c = 0.0
    for i in range(n_samples):
        # Find points that are in the K-neighborhood in high-dim BUT NOT in low-dim
        low_neighbors = set(neigh_low[i])
        high_neighbors = set(neigh_high[i])
        extruders = high_neighbors - low_neighbors
        if extruders:
            for j in extruders:
                # Sum ranks of extruders in low-dim space (based on high-dim ordering)
                # Find rank of j in low-dim ordering of i
                rank_low_j = np.where(np.argsort(dist_low[i]) == j)[0][0]
                if rank_low_j > n_neighbors: # Check if it's truly an extruder rank-wise
                   c += (rank_low_j - n_neighbors)


    # Normalization constant
    if n_samples <= n_neighbors: # Avoid division by zero or negative normalization
         return 1.0 # Perfect score if K >= N-1

    C = 2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))

    continuity = 1.0 - C * c
    return max(0.0, continuity) # Ensure score is not negative

def total_absolute_error(computed_order, ground_truth_order):
    """
    Calculates the Total Absolute Error (TAE) between two orderings.

    Args:
        computed_order (list or np.ndarray): Array of indices representing the computed order.
        ground_truth_order (list or np.ndarray): Array of indices representing the true order.

    Returns:
        int: The Total Absolute Error. Lower is better.
    """
    if len(computed_order) != len(ground_truth_order):
        raise ValueError("Orderings must have the same length.")

    n = len(ground_truth_order)
    # Create mapping from item index to its position in each order
    pos_computed = {index: pos for pos, index in enumerate(computed_order)}
    pos_truth = {index: pos for pos, index in enumerate(ground_truth_order)}

    tae = 0
    # Sum absolute differences in position for each item
    for index in ground_truth_order: # Iterate through items in truth order
        if index in pos_computed:
            tae += abs(pos_computed[index] - pos_truth[index])
        else:
             # Should not happen if both orders contain the same elements
             raise ValueError(f"Index {index} not found in computed_order")

    return tae

# File: src/metrics.py (Replace the compute_all_metrics function)
# Ensure other functions (check_pairwise_distances, trustworthiness, continuity, total_absolute_error)
# and necessary imports (numpy, pandas, etc.) are still present in the file.

def compute_all_metrics(X_high, embeddings_dict, ground_truth_order, n_neighbors=5):
     """
     Computes TAE, Trustworthiness, and Continuity for multiple embeddings.
     Checks for reversed order when calculating TAE and reports the minimum TAE.

     Args:
         X_high (np.ndarray): Original high-dimensional data (flattened, n_samples, n_features).
         embeddings_dict (dict): Dictionary where keys are method names and values are
                                  the corresponding 2D embeddings (n_samples, 2).
         ground_truth_order (np.ndarray): Array of indices representing the true order.
         n_neighbors (int): Number of neighbors for Trustworthiness/Continuity.

     Returns:
         pd.DataFrame: DataFrame containing the metrics for each method.
     """
     results = []
     n_samples = X_high.shape[0]

     for name, X_low in embeddings_dict.items():
         if X_low is None or X_low.shape[0] != n_samples or X_low.shape[1] < 1: # Check if embedding is valid and has at least 1 dim
             print(f"Skipping {name}: Invalid embedding data.")
             continue

         # --- Calculate TAE (Check for reversed order) ---
         # Get 1D embedding (first dimension)
         embedding_1d = X_low[:, 0]

         # Get order based on sorting the 1D embedding (ascending)
         computed_order_fwd = np.argsort(embedding_1d)
         tae_fwd = total_absolute_error(computed_order_fwd, ground_truth_order)

         # Get order based on sorting the NEGATED 1D embedding (descending)
         computed_order_rev = np.argsort(-embedding_1d)
         tae_rev = total_absolute_error(computed_order_rev, ground_truth_order)

         # Choose the minimum TAE (handles potential orientation flip)
         tae = min(tae_fwd, tae_rev)
         if tae != tae_fwd:
             print(f"Note: TAE for {name} was calculated using reversed order (min({tae_fwd}, {tae_rev})).")


         # --- Calculate Trustworthiness & Continuity ---
         # Ensure X_low is 2D for these metrics
         if X_low.shape[1] >= 2:
              trust = trustworthiness(X_high, X_low[:, :2], n_neighbors=n_neighbors)
              cont = continuity(X_high, X_low[:, :2], n_neighbors=n_neighbors)
         else:
              print(f"Warning: Cannot compute T&C for {name} as embedding is not 2D.")
              trust, cont = np.nan, np.nan # Use NaN if embedding is not 2D


         results.append({
             'Method': name,
             'TAE (min)': tae, # Indicate that it's the minimum TAE
             'Trustworthiness': trust,
             'Continuity': cont
         })

     # Convert results to DataFrame and sort
     metrics_df = pd.DataFrame(results)
     if 'TAE (min)' in metrics_df.columns:
         metrics_df = metrics_df.sort_values(by='TAE (min)') # Sort by corrected TAE

     return metrics_df