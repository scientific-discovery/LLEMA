import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def log_info(msg):
    logging.info(msg)

def log_error(msg):
    logging.error(msg)

def extract_json_from_llm(raw_response):
    s = raw_response.strip()
    
    # Handle markdown code blocks
    if s.startswith('```'):
        s = s.split('```', 2)[1]
        if s.strip().startswith('json'):
            s = s.strip()[4:]
    
    # Try to find JSON object or array in the response
    import re
    
    # First try to find complete JSON objects by counting braces
    objects = []
    start = -1
    brace_count = 0
    
    for i, char in enumerate(s):
        if char == '{':
            if start == -1:
                start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start != -1:
                objects.append(s[start:i+1])
                start = -1
    
    if len(objects) > 1:
        # Multiple objects found, wrap in array
        return f"[{','.join(objects)}]"
    elif len(objects) == 1:
        # Single object found, return as-is
        return objects[0]
    
    # If no objects found, try to find JSON arrays
    # Use a more precise regex that matches balanced brackets
    json_array_match = re.search(r'\[(?:[^\[\]]|\[[^\]]*\])*\]', s)
    if json_array_match:
        return json_array_match.group(0).strip()
    
    return s.strip()

def try_parse_json_response(raw_response, context_info, prompt=None):
    cleaned_response = extract_json_from_llm(raw_response)
    if not cleaned_response:
        return []
    try:
        parsed = json.loads(cleaned_response)
        return parsed
    except Exception as e:
        return [] 

# --- Island memory helpers ---
import os, random, time, glob
from config import ISLANDS_DIR, ISLAND_ID

def _island_file(island_id: str) -> str:
    return os.path.join(ISLANDS_DIR, f"island_{island_id}.json")

def save_island_memory(island_id: str, evolution_memory: dict):
    payload = {
        "island_id": island_id,
        "timestamp": time.time(),
        "success": list(evolution_memory.get("success", [])),
        "failure": list(evolution_memory.get("failure", [])),
    }
    with open(_island_file(island_id), 'w') as f:
        json.dump(payload, f, indent=2)

def load_all_island_memories(exclude_id: str | None = None) -> list[dict]:
    memories = []
    for fp in glob.glob(os.path.join(ISLANDS_DIR, "island_*.json")):
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            if exclude_id is not None and str(data.get("island_id")) == str(exclude_id):
                continue
            if not isinstance(data.get("success", []), list) or not isinstance(data.get("failure", []), list):
                continue
            memories.append(data)
        except Exception:
            continue
    return memories

def sample_other_island_memory(exclude_id: str | None = None) -> dict | None:
    others = load_all_island_memories(exclude_id=exclude_id)
    if not others:
        return None
    chosen = random.choice(others)
    return {
        "success": chosen.get("success", []),
        "failure": chosen.get("failure", []),
        "source_island_id": chosen.get("island_id"),
        "timestamp": chosen.get("timestamp"),
    }

def wait_and_sample_other_island_memory(exclude_id: str | None = None, timeout_seconds: int = 20, poll_interval_seconds: float = 2.0) -> dict | None:
    """
    Poll for other islands' memories for up to timeout_seconds.
    Returns a sampled memory or None if none found.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        m = sample_other_island_memory(exclude_id=exclude_id)
        if m:
            return m
        time.sleep(poll_interval_seconds)
    return None