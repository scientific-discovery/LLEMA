import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..')
ALL_MATERIALS_PATH = os.path.join(DATA_DIR, 'test_materials.jsonl')
TASK_DESCRIPTION_PATH = os.path.join(DATA_DIR, 'all_tasks.csv')
MODEL_LOOKUP_PATH = os.path.join(DATA_DIR, 'surrogate_models', 'task_to_model_lookup.json')

# LLM
# Prefer OpenAI style; model can be overridden by env `LLM_MODEL`
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
LLM_MODEL = os.environ.get('LLM_MODEL', 'gpt-4o-mini')

# Iteration
MAX_ITERATIONS = int(os.environ.get('MAX_ITERATIONS', '1000'))

# Memory
FULL_LOG_PATH = os.path.join(BASE_DIR, 'full_log_memory.json')
EVOLUTION_MEMORY_PATH = os.path.join(BASE_DIR, 'evolution_memory.json')

# Other settings can be added as needed 

# Multi-island settings
ISLAND_ID = os.environ.get('ISLAND_ID', '0')
ISLANDS_DIR = os.environ.get('ISLANDS_DIR', os.path.join(BASE_DIR, 'runs_mistral', 'islands'))
os.makedirs(ISLANDS_DIR, exist_ok=True)

# Centralized islands config
NUM_ISLANDS = int(os.environ.get('NUM_ISLANDS', '5'))
ISLAND_FUNCTIONS_PER_PROMPT = int(os.environ.get('ISLAND_FUNCTIONS_PER_PROMPT', '4'))
ISLAND_TEMP_INIT = float(os.environ.get('ISLAND_TEMP_INIT', '1.0'))
ISLAND_TEMP_PERIOD = int(os.environ.get('ISLAND_TEMP_PERIOD', '10'))
ISLAND_RESET_PERIOD_SECONDS = int(os.environ.get('ISLAND_RESET_PERIOD_SECONDS', '43200'))  # 30 minutes

# Memory refresh/pruning
MEMORY_REFRESH_INTERVAL = int(os.environ.get('MEMORY_REFRESH_INTERVAL', '100'))  # iterations
MEMORY_MAX_ITEMS_PER_ISLAND = int(os.environ.get('MEMORY_MAX_ITEMS_PER_ISLAND', '5000'))
MEMORY_TOP_K_SUCCESS = int(os.environ.get('MEMORY_TOP_K_SUCCESS', '10'))
MEMORY_BOTTOM_K_FAILURE = int(os.environ.get('MEMORY_BOTTOM_K_FAILURE', '10'))

# Debug controls
EVOLUTION_DEBUG_PRINT_PROMPT = os.environ.get('EVOLUTION_DEBUG_PRINT_PROMPT', '1') == '1'

# Surrogate model paths
SURROGATE_MODELS_DIR = os.environ.get('SURROGATE_MODELS_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'surrogate_models'))
CGCNN_DIR = os.path.join(SURROGATE_MODELS_DIR, 'cgcnn')
ALIGNN_DIR = os.path.join(SURROGATE_MODELS_DIR, 'alignn')