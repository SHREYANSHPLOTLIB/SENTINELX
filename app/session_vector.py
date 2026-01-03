import numpy as np
from entropy import shannon_entropy

def build_session_vector(events):
    commands = [e["data"] for e in events if e["type"] == "command"]
    files = [e["data"] for e in events if e["type"] == "file"]

    return np.array([
        len(commands),
        shannon_entropy(commands),
        len(files),
        shannon_entropy(files)
    ])
