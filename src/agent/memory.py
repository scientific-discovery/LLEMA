import json
from config import FULL_LOG_PATH, EVOLUTION_MEMORY_PATH

class MemoryManager:
    def __init__(self):
        self.full_log = []
        self.evolution_memory = {'success': [], 'failure': []}
        self.load()

    def log_candidate(self, candidate_info):
        self.full_log.append(candidate_info)
        self.save()

    def update_evolution_memory(self):
        successes = [c for c in self.full_log if c.get('overall', False)]
        failures = [c for c in self.full_log if not c.get('overall', False)]
        self.evolution_memory['success'] = sorted(successes, key=lambda x: x.get('score', 0), reverse=True)[:10]
        self.evolution_memory['failure'] = sorted(failures, key=lambda x: x.get('score', 0))[:10]
        self.save()

    def save(self):
        with open(FULL_LOG_PATH, 'w') as f:
            json.dump(self.full_log, f, indent=2)
        with open(EVOLUTION_MEMORY_PATH, 'w') as f:
            json.dump(self.evolution_memory, f, indent=2)

    def load(self):
        try:
            with open(FULL_LOG_PATH, 'r') as f:
                self.full_log = json.load(f)
        except Exception:
            self.full_log = []
        try:
            with open(EVOLUTION_MEMORY_PATH, 'r') as f:
                self.evolution_memory = json.load(f)
        except Exception:
            self.evolution_memory = {'success': [], 'failure': []} 