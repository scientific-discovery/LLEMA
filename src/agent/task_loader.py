import csv
import os
from typing import List, Dict
from config import TASK_DESCRIPTION_PATH

def load_tasks() -> List[Dict]:
    tasks = []
    # Use environment variable if set, otherwise use config default
    task_path = os.environ.get('TASK_DESCRIPTION_PATH', TASK_DESCRIPTION_PATH)
    with open(task_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = {
                'name': row['Task'],
                'goal': row['Goal'],
                'properties_and_models': row['Properties and Models']
            }
            tasks.append(task)
    return tasks