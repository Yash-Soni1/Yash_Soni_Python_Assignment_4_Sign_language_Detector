import json
from pathlib import Path

def save_label_map(label_map, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(label_map, f)

def load_label_map(path):
    with open(path, 'r') as f:
        return json.load(f)
