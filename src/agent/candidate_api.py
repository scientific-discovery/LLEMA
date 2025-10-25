import json
import random
from typing import List, Dict
from config import ALL_MATERIALS_PATH

def get_example_candidates(n=5) -> List[Dict]:
    candidates = []
    with open(ALL_MATERIALS_PATH, 'r') as f:
        # Only read first 100 lines to avoid memory issues
        for i, line in enumerate(f):
            if i >= 100:  # Limit to first 100 materials
                break
            candidates.append(json.loads(line))
    return random.sample(candidates, min(n, len(candidates))) 