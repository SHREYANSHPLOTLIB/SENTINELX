import numpy as np

def build_fingerprint(vectors):
    X = np.vstack(vectors)
    return {
        "mean": np.mean(X, axis=0),
        "cov": np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
    }
