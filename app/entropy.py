import numpy as np
from collections import Counter

def shannon_entropy(values):
    if not values:
        return 0.0
    counts = Counter(values)
    probs = np.array(list(counts.values())) / len(values)
    return -np.sum(probs * np.log2(probs))
