import numpy as np

def anomaly_score(vector, fingerprint):
    diff = vector - fingerprint["mean"]
    inv_cov = np.linalg.inv(fingerprint["cov"])
    return np.sqrt(diff.T @ inv_cov @ diff)
