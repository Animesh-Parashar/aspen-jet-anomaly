"""
k-NN anomaly scoring in the learned latent space.
Anomaly score = distance to the k-th nearest neighbour in the background set.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, roc_curve
import torch


class KNNAnomalyScorer:
    def __init__(self, k=10, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
        self._fitted = False

    def fit(self, background_embeddings):
        """background_embeddings: (N, D) numpy array or torch tensor."""
        bg = _to_numpy(background_embeddings)
        self.nn.fit(bg)
        self._fitted = True

    def score(self, embeddings):
        """Returns anomaly scores (higher = more anomalous)."""
        assert self._fitted, "Call fit() first"
        z = _to_numpy(embeddings)
        dists, _ = self.nn.kneighbors(z)
        # dists[:, k-1] = k-th nearest neighbour distance.
        # With n_neighbors=k+1: if a query point is in the fit set, dists[:,0]=0
        # (self-match), so dists[:,k-1] still gives the k-th *non-self* neighbour.
        # When the query is not in the fit set, dists[:,k-1] is simply the k-th NN.
        return dists[:, self.k - 1]

    def evaluate(self, test_embeddings, labels):
        """
        Compute ROC-AUC and return (auc, fpr, tpr, thresholds).
        labels: 0=background, 1=signal.
        """
        scores = self.score(test_embeddings)
        labels = _to_numpy(labels)
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        return auc, fpr, tpr, thresholds


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_rejection(fpr, tpr, signal_eff):
    """Background rejection (1/FPR) at a given signal efficiency (TPR)."""
    idx = np.searchsorted(tpr, signal_eff)
    if idx >= len(fpr):
        return np.inf
    f = fpr[idx]
    return 1.0 / f if f > 0 else np.inf
