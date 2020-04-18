import json
from dataclasses import dataclass

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

