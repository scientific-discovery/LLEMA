import os
import json
import logging
import sys
from datetime import datetime
from functools import lru_cache
from config import MAX_ITERATIONS
from config import (
    NUM_ISLANDS,
    ISLAND_FUNCTIONS_PER_PROMPT,
    ISLAND_TEMP_INIT,
    ISLAND_TEMP_PERIOD,
    ISLAND_RESET_PERIOD_SECONDS,
    MEMORY_REFRESH_INTERVAL,
    MEMORY_MAX_ITEMS_PER_ISLAND,
    MEMORY_TOP_K_SUCCESS,
    MEMORY_BOTTOM_K_FAILURE,
    EVOLUTION_DEBUG_PRINT_PROMPT,
)
from islands import ExperienceBuffer
from task_loader import load_tasks
from candidate_api import get_example_candidates
from llm_interface import call_llm
from structure_builder import build_cif
from surrogate_runner import run_surrogates
from memory import MemoryManager
from evolution import evolve_candidates
from utils import log_info, log_error, extract_json_from_llm, try_parse_json_response
import random
import property_extractor

# Map task names to property names
TASK_TO_PROPERTY_MAP = {
    "Stable Wide-Bandgap Semiconductors": ["band_gap", "formation_energy", "energy_above_hull"],
    "Photovoltaic Absorbers": ["band_gap", "formation_energy", "energy_above_hull"],
    "Hard Coating Materials": ["bulk_modulus", "formation_energy", "band_gap", "energy_above_hull"],
    "Transparent Conductors": ["band_gap", "electrical_conductivity", "energy_above_hull"],
    "Structural Materials for Aerospace": ["bulk_modulus", "shear_modulus" , "density", "energy_above_hull"],
    "Thermoelectric Candidates (n-type or p-type)": ["seebeck_coefficient", "thermal_conductivity", "band_gap", "formation_energy", "energy_above_hull"],
    "Electrically Insulating Dielectrics": ["band_gap", "dielectric_constant", "energy_above_hull"],
    "Solid-State Electrolytes": ["formation_energy", "energy_above_hull", "band_gap"],
    "Hard, Stiff Ceramics": ["bulk_modulus", "shear_modulus", "energy_above_hull"],
    "High-k Dielectrics": ["dielectric_constant", "band_gap", "energy_above_hull"],
    "SAW/BAW Acoustic Substrates": ["shear_modulus", "dielectric_constant", "energy_above_hull"],
    "Piezo Energy Harvesters": ["piezo_max_dij", "piezo_max_dielectric", "energy_above_hull"],
    "Piezoelectric Sensors / Actuators": ["piezo_max_dij", "piezo_max_dielectric", "energy_above_hull"],
    "Acousto-optic Hybrids": ["piezo_max_dij", "piezo_max_dielectric", "energy_above_hull"],
    "Low_Density_Structural_Aerospace": ["density", "shear_modulus", "energy_above_hull"],
    "Toxic_Free_Perovskite_Oxide": ["band_gap", "bulk_modulus", "energy_above_hull"]
}

METRIC_TO_PROPERTY_MAP = {
    "band_gap": ("Band Gap", "eV"),
    "formation_energy": ("Formation Energy", "eV/atom"),
    "energy_above_hull": ("Energy Above Hull", "eV/atom"),
    "density": ("Density", "g/cm³"),
    "dielectric_constant": ("Dielectric Constant", ""),
    "bulk_modulus": ("Bulk modulus", "GPa"),
    "shear_modulus": ("Shear modulus", "GPa"),
    "seebeck_coefficient": ("Seebeck Coefficient", "μV/K"),
    "thermal_conductivity": ("Thermal Conductivity", "W·m⁻¹·K⁻¹"),
    "electrical_conductivity": ("Electrical Conductivity", "S/m"),
    "poisson_ratio": ("Poisson Ratio", ""),
    "piezo_max_dielectric": ("Max piezo dielectric (κ)", ""),
    "piezo_max_dij": ("Max piezo d_ij", "pC/N")
}

# Timestamped base run directory (per-task subfolders will be created inside)
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = os.path.join(os.path.dirname(__file__), f'runs_llmatdesign/run_{RUN_TIMESTAMP}')
os.makedirs(RUN_DIR, exist_ok=True)

PIPELINE_LOG = os.path.join(RUN_DIR, 'pipeline.log')
# Candidate and LLM logs will be per-task in subfolders; keep base paths for fallback
CANDIDATES_LOG = os.path.join(RUN_DIR, 'candidates.log')
LLM_CALLS_LOG = os.path.join(RUN_DIR, 'llm_calls.log')
FULL_LOG_MEMORY = os.path.join(RUN_DIR, 'full_log_memory.json')
EVOLUTION_MEMORY = os.path.join(RUN_DIR, 'evolution_memory.json')
# Comprehensive log file that captures all output
COMPREHENSIVE_LOG = os.path.join(RUN_DIR, 'comprehensive.log')

# Set up comprehensive logging
class TeeOutput:
    """Class to duplicate output to both console and file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Configure logging to capture all output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(COMPREHENSIVE_LOG),
        logging.StreamHandler(sys.stdout)
    ]
)

# Redirect stdout to both console and file
log_file = open(COMPREHENSIVE_LOG, 'a')
original_stdout = sys.stdout
sys.stdout = TeeOutput(sys.stdout, log_file)

# Also set up the original pipeline logging
pipeline_logger = logging.getLogger('pipeline')
pipeline_handler = logging.FileHandler(PIPELINE_LOG)
pipeline_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
pipeline_logger.addHandler(pipeline_handler)
pipeline_logger.setLevel(logging.INFO)

def log_pipeline(msg):
    print(msg)
    logging.info(msg)
    pipeline_logger.info(msg)

def cleanup_logging():
    """Clean up logging resources"""
    global log_file, original_stdout
    if 'log_file' in globals() and log_file:
        log_file.close()
    if 'original_stdout' in globals():
        sys.stdout = original_stdout

def _sanitize_name(name: str) -> str:
    name = (name or "").strip()
    safe = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in name)
    return safe[:80] or 'task'

def _task_dir(task_name: str) -> str:
    tdir = os.path.join(RUN_DIR, _sanitize_name(task_name))
    os.makedirs(tdir, exist_ok=True)
    return tdir

def log_candidate(candidate, task_name: str):
    path = os.path.join(_task_dir(task_name), 'candidates.log')
    with open(path, 'a') as f:
        f.write(json.dumps(candidate) + '\n')

def log_candidate_enhanced(candidate_data, task_name, iteration, island_id, materials_api_used=False):
    """
    Enhanced candidate logging with all required fields for plotting
    """
    # Extract property values from predictions using the same logic as simple_score
    property_values = _values_from_predictions(candidate_data.get('predictions', {}))
    
    # Count constraint violations using the same logic as simple_score
    vios = _violations_for(task_name, property_values, candidate_data)
    total_constraints = len(TASK_CONSTRAINTS.get(task_name, {}).get("numeric", []))
    
    # Count categorical constraints if they exist
    if TASK_CONSTRAINTS.get(task_name, {}).get("categorical"):
        cat_constraints = TASK_CONSTRAINTS[task_name]["categorical"]
        if cat_constraints.get("requires_any_element"):
            total_constraints += 1
        if cat_constraints.get("non_toxic"):
            total_constraints += 1
        if cat_constraints.get("earth_abundant"):
            total_constraints += 1
    
    failed_constraints = len(vios)
    successful_constraints = total_constraints - failed_constraints
    
    enhanced_candidate = {
        'iteration': iteration,
        'compound_formula': candidate_data.get('formula', candidate_data.get('compound', 'Unknown')),
        'score': candidate_data.get('score', 0.0),
        'property_values': property_values,
        'successful_constraints': successful_constraints,
        'failed_constraints': failed_constraints,
        'rules_used': candidate_data.get('rules_used', ''),
        'island_id': island_id,
        'materials_api_used': materials_api_used,
        'timestamp': datetime.now().isoformat(),
        'step': candidate_data.get('step', 'unknown'),
        'task_name': task_name
    }
    
    path = os.path.join(_task_dir(task_name), 'candidates.log')
    with open(path, 'a') as f:
        f.write(json.dumps(enhanced_candidate) + '\n')

def log_llm_call(prompt, response, context, task_name: str | None = None):
    if task_name is not None:
        log_path = os.path.join(_task_dir(task_name), 'llm_calls.log')
    else:
        log_path = LLM_CALLS_LOG
    with open(log_path, 'a') as f:
        f.write(json.dumps({
            'context': context,
            'prompt': prompt,
            'response': response
        }) + '\n')

# Load evolution templates
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'templates', 'evolution_templates.json')
with open(TEMPLATE_PATH, 'r') as f:
    EVOLUTION_TEMPLATES = json.load(f)
# Load rules text
RULES_PATH = os.path.join(os.path.dirname(__file__), 'generation_rules.md')
with open(RULES_PATH, 'r') as f:
    RULES_TEXT = f.read().split('---')[0]

RAW_LLM_LOG = os.path.join(os.path.dirname(__file__), 'raw_llm_responses.log')
PARSE_ERROR_LOG = os.path.join(os.path.dirname(__file__), 'llm_parse_errors.log')

RETRY_EMPTY_LIMIT = 3

# Cache for Materials Project API calls to avoid repeated requests
@lru_cache(maxsize=100)
def get_cached_materials_properties(structure_path: str, formula: str = None):
    """Cached wrapper for Materials Project API calls"""
    try:
        return property_extractor.api.get_all_5_properties(structure_path, formula)
    except Exception as e:
        print(f"[Agent] Materials API error: {e}")
        return {}

def has_structure(candidate):
    return all(k in candidate for k in ['lattice', 'species', 'coords'])

def normalize_candidate_keys(candidate):
    key_map = {
        'compound': 'compound',
        'rules_used': 'rules_used',
        'rule(s) used': 'rules_used',
        'justification': 'justification',
        'lattice': 'lattice',
        'species': 'species',
        'fractional coordinates': 'coords',
        'coords': 'coords',
        'formula': 'formula',
        'name': 'name',
    }
    norm = {}
    for k, v in candidate.items():
        k_lower = k.lower().strip()
        mapped = key_map.get(k_lower, k_lower)
        norm[mapped] = v
    return norm

def constraints_to_natural_language(task_name):
    """Convert task constraints to natural language descriptions"""
    if task_name not in TASK_CONSTRAINTS:
        return "No specific constraints defined."
    
    constraints = TASK_CONSTRAINTS[task_name]
    descriptions = []
    
    # Property name mapping
    prop_mapping = {
        'band_gap': 'band gap',
        'formation_energy': 'formation energy',
        'energy_above_hull': 'energy above hull',
        'dielectric_constant': 'dielectric constant',
        'bulk_modulus': 'bulk modulus',
        'shear_modulus': 'shear modulus',
        'density': 'density'
    }
    
    # Process numeric constraints
    if 'numeric' in constraints:
        for prop, op, value in constraints['numeric']:
            readable_prop = prop_mapping.get(prop, prop)
            if op == ">=":
                descriptions.append(f"• {readable_prop} must be at least {value}")
            elif op == "<=":
                descriptions.append(f"• {readable_prop} must be at most {value}")
            elif op == "in":
                lo, hi = value
                descriptions.append(f"• {readable_prop} must be between {lo} and {hi}")
            elif op == "≈":
                target, tol = value
                descriptions.append(f"• {readable_prop} must be approximately {target} (±{tol})")
    
    # Process categorical constraints
    if 'categorical' in constraints:
        cat = constraints['categorical']
        if cat.get('requires_any_element'):
            elements = cat['requires_any_element']
            element_groups = []
            for group in elements:
                if len(group) == 1:
                    element_groups.append(group[0])
                else:
                    element_groups.append(f"any of {', '.join(group)}")
            descriptions.append(f"• Must contain {', or '.join(element_groups)}")
        
        if cat.get('non_toxic'):
            descriptions.append("• Must be non-toxic (avoid Hg, Cd, Pb, Tl, As, Be, Se, U, Th, Sb)")
        
        if cat.get('earth_abundant'):
            if task_name == "Photovoltaic Absorbers":
                descriptions.append("• Must use ONLY earth-abundant and non-toxic elements (H, Li, B, C, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, Sc, Ti, V, Mn, Fe, Co, Ni, Cu, Zn, Ga, Rb, Sr, Y, Zr, Nb, La, Ce, Nd)")
                descriptions.append("• Earth-abundant elements are chemical elements that are relatively common in the Earth's crust (>10 ppm by weight) and therefore inexpensive, sustainable, and widely available for large-scale applications. They are the opposite of scarce or critical elements (like rare earth metals, platinum-group elements, or tellurium) that are limited in supply, geographically concentrated, or costly to extract.")
            else:
                descriptions.append("• Must use earth-abundant and non-toxic elements (H, Li, B, C, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, Sc, Ti, V, Mn, Fe, Co, Ni, Cu, Zn, Ga, Rb, Sr, Y, Zr, Nb, La, Ce, Nd)")
    
    return '\n'.join(descriptions) if descriptions else "No specific constraints defined."
#####################
# ======== DROP-IN SCORER (replace old simple_score and its helpers) ========

import re, math

# --- Constraints & Objectives (simple, explicit) ---
TASK_CONSTRAINTS = {
    "Stable Wide-Bandgap Semiconductors": {
        "numeric": [
            ("band_gap", ">=", 2.5),
            ("formation_energy", "<=", -1.0),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "Thermoelectric Candidates (n-type or p-type)": {
        "numeric": [
            ("seebeck_coefficient", ">=", 80),          # Seebeck coefficient in μV·K⁻¹ (higher is better)
            ("thermal_conductivity", "<=", 3.0),         # thermal conductivity in W·m⁻¹·K⁻¹ (lower is better)
            ("band_gap", "in", (0.1, 1.75)),            # band gap in eV
            ("formation_energy", "<=", 0.0),            # formation energy in eV·atom⁻¹
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "Electrically Insulating Dielectrics": {
        "numeric": [
            ("band_gap", ">=", 2.5),
            ("dielectric_constant", ">=", 8.0),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "Solid-State Electrolytes": {
        "numeric": [
            ("formation_energy", "<=", -1.0),
            ("energy_above_hull", "<=", 2.0),
            ("band_gap", ">=", 2.0),
        ],
        "categorical": {"requires_any_element": [["Li"], ["Na"], ["K"], ["Mg"], ["Ca"], ["Al"]]},
    },
    "Photovoltaic Absorbers": {
        "numeric": [
            ("band_gap", "in", (0.7, 2.0)),
            ("formation_energy", "<=", 0.0),
            ("energy_above_hull", "<=", 2.0),
        ],
        "categorical": {"earth_abundant": True, "non_toxic": True},
    },
    "Hard Coating Materials": {
        "numeric": [
            ("bulk_modulus", ">=", 200.0),
            ("formation_energy", "<=", -1.0),
            ("band_gap", ">=", 3.0),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "Transparent Conductors": {
        "numeric": [
            ("band_gap", ">=", 1.5),
            ("electrical_conductivity", "in", (500, 30000)),  # conductivity in S/m (500-15,000 S/m)
            ("energy_above_hull", "<=", 5.0),
        ]
    },
    "Structural Materials for Aerospace": {
        "numeric": [
            ("density", "<=", 5.0),
            ("bulk_modulus", ">=", 100.0),
            ("shear_modulus", ">=", 40.0),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "Hard, Stiff Ceramics": {
        "numeric": [
            ("bulk_modulus", "in", (100.0, 300.0)),
            ("shear_modulus", "in", (60.0, 200.0)),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "High-k Dielectrics": {
        "numeric": [
            ("dielectric_constant", "in", (10.0, 90.0)),
            ("band_gap", "in", (2.5, 6.5)),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "SAW/BAW Acoustic Substrates": {
        "numeric": [
            ("shear_modulus", "in", (25.0, 150.0)),
            ("dielectric_constant", "in", (3.7, 95.0)),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "Piezo Energy Harvesters": {
        "numeric": [
            ("piezo_max_dij", ">=", 8.0),
            ("piezo_max_dielectric", "in", (10.0, 8000.0)),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "Piezoelectric Sensors / Actuators": {
        "numeric": [
            ("piezo_max_dij", ">=", 80.0),
            ("piezo_max_dielectric", ">=", 250.0),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "Acousto-optic Hybrids": {
        "numeric": [
            ("piezo_max_dij", "in", (2.0, 9.0)),
            ("piezo_max_dielectric", "in", (8.0, 95.0)),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "Low_Density_Structural_Aerospace": {
        "numeric": [
            ("density", "<=", 3.5),                      # density in g/cm³
            ("shear_modulus", "in", (65.0, 195.0)),      # shear modulus in GPa
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "Toxic_Free_Perovskite_Oxide": {
        "numeric": [
            ("band_gap", ">=", 2.0),                     # band gap in eV
            ("bulk_modulus", "in", (90.0, 135.0)),       # bulk modulus in GPa
            ("energy_above_hull", "<=", 2.0),
        ],
        "categorical": {"earth_abundant": True, "non_toxic": True},
    },
}

# Global target overrides for value-equals-target style objectives
# Map: property_name -> (target_value, tolerance_window)
TARGET_OVERRIDES = {
    # Example: treat energy_above_hull as target 0 with ±1.0 tolerance
    "energy_above_hull": (0.0, 5.0),
}

TASK_OBJECTIVES = {
    "Stable Wide-Bandgap Semiconductors": {"lower": ["formation_energy", "energy_above_hull"], "higher": ["band_gap"]},
    "Thermoelectric Candidates (n-type or p-type)": {"lower": ["thermal_conductivity", "formation_energy", "energy_above_hull"], "higher": ["seebeck_coefficient"]},
    "Electrically Insulating Dielectrics": {"lower": ["energy_above_hull"], "higher": ["band_gap", "dielectric_constant"]},
    "Solid-State Electrolytes": {"lower": ["formation_energy", "energy_above_hull"], "higher": ["band_gap"]},
    "Photovoltaic Absorbers": {"lower": ["formation_energy", "energy_above_hull"], "higher": ["band_gap"]},
    "Hard Coating Materials": {"lower": ["formation_energy", "energy_above_hull"], "higher": ["bulk_modulus", "band_gap"]},
    "Transparent Conductors": {"lower": ["energy_above_hull"], "higher": ["band_gap", "electrical_conductivity"]},
    "Structural Materials for Aerospace": {"lower": ["density", "energy_above_hull"], "higher": ["bulk_modulus", "shear_modulus"]},
    "Hard, Stiff Ceramics": {"lower": ["energy_above_hull"], "higher": ["bulk_modulus", "shear_modulus"]},
    "High-k Dielectrics": {"lower": ["energy_above_hull"], "higher": ["dielectric_constant", "band_gap"]},
    "SAW/BAW Acoustic Substrates": {"lower": ["energy_above_hull"], "higher": ["shear_modulus", "dielectric_constant"]},
    "Piezo Energy Harvesters": {"lower": ["energy_above_hull", "piezo_max_dielectric"], "higher": ["piezo_max_dij"]},
    "Piezoelectric Sensors / Actuators": {"lower": ["energy_above_hull"], "higher": ["piezo_max_dij", "piezo_max_dielectric"]},
    "Acousto-optic Hybrids": {"lower": ["energy_above_hull", "piezo_max_dielectric"], "higher": ["piezo_max_dij"]},
    "Low_Density_Structural_Aerospace": {"lower": ["density", "energy_above_hull"], "higher": ["shear_modulus"]},
    "Toxic_Free_Perovskite_Oxide": {"lower": ["energy_above_hull"], "higher": ["band_gap", "bulk_modulus"]},
}

# --- Property name handling ---
PROPERTY_SET = {
    "formation_energy","energy_above_hull","band_gap","thermal_conductivity",
    "seebeck_coefficient","dielectric_constant","ionic_conductivity","bulk_modulus",
    "shear_modulus","electrical_conductivity","density","poisson_ratio",
    "piezo_max_dielectric","piezo_max_dij"
}
ALIASES = {
    # canonical → same
    "formation_energy":"formation_energy","energy_above_hull":"energy_above_hull","band_gap":"band_gap",
    "thermal_conductivity":"thermal_conductivity","seebeck_coefficient":"seebeck_coefficient",
    "dielectric_constant":"dielectric_constant","ionic_conductivity":"ionic_conductivity",
    "bulk_modulus":"bulk_modulus","shear_modulus":"shear_modulus",
    "electrical_conductivity":"electrical_conductivity","density":"density",
    "poisson_ratio":"poisson_ratio",
    # common variants
    "formation energy":"formation_energy","e_form":"formation_energy","ef":"formation_energy",
    "energy above hull":"energy_above_hull","e_above_hull":"energy_above_hull","e above hull":"energy_above_hull",
    "band gap":"band_gap","bandgap":"band_gap","e_g":"band_gap",
    "kappa":"thermal_conductivity",
    "seebeck":"seebeck_coefficient","s":"seebeck_coefficient",
    "dielectric":"dielectric_constant","κ":"dielectric_constant","k":"dielectric_constant",
    "sigma":"electrical_conductivity",
    "k_bulk":"bulk_modulus","g_shear":"shear_modulus",
    # piezo-related aliases
    "piezo_max_dielectric":"piezo_max_dielectric","piezoelectric_max_dielectric":"piezo_max_dielectric","jv_dfpt_piezo_max_dielectric_alignn":"piezo_max_dielectric","max piezo dielectric":"piezo_max_dielectric",
    "piezo_max_dij":"piezo_max_dij","jv_dfpt_piezo_max_dij_alignn":"piezo_max_dij","max piezo d_ij":"piezo_max_dij",
}
def _canon(s: str) -> str:
    s = (s or "").strip().lower()
    return ALIASES.get(s, s)

# Infer property from stdout content
def _infer_prop_from_stdout(stdout: str) -> str | None:
    s = (stdout or "").lower()
    checks = [
        ("energy above hull","energy_above_hull"), ("e_above_hull","energy_above_hull"), ("e above hull","energy_above_hull"),
        ("formation energy","formation_energy"), ("formation-energy","formation_energy"), ("e_form","formation_energy"),
        ("band gap","band_gap"), ("bandgap","band_gap"),
        ("thermal conductivity","thermal_conductivity"), ("kappa","thermal_conductivity"),
        ("seebeck","seebeck_coefficient"),
        ("dielectric constant","dielectric_constant"), ("dielectric","dielectric_constant"), ("κ","dielectric_constant"),
        ("ionic conductivity","ionic_conductivity"),
        ("bulk modulus","bulk_modulus"),
        ("shear modulus","shear_modulus"),
        ("electrical conductivity","electrical_conductivity"), ("sigma","electrical_conductivity"),
        ("density","density"),
        # piezo prints
        ("max piezo dielectric","piezo_max_dielectric"),
        ("max piezo d_ij","piezo_max_dij"),
    ]
    for key, prop in checks:
        if key in s:
            return prop
    return None

def _first_number(text: str):
    if not text: return None
    m = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(text))
    return float(m[0]) if m else None

def _values_from_predictions(predictions: dict) -> dict:
    """
    Build {property: value}. Accepts:
      - structured numeric fields in dicts
      - parses 'stdout' and infers property from its text (your case)
    Averages duplicates if multiple models produce the same property.
    """
    sums, counts = {}, {}
    for model_name, res in (predictions or {}).items():
        v, stdout = None, None
        if isinstance(res, dict):
            for k in ("value","pred","prediction","y","score"):
                if k in res and isinstance(res[k], (int,float)):
                    v = float(res[k]); break
            stdout = res.get("stdout")
            if v is None and stdout:
                v = _first_number(stdout)
        if v is None:
            continue

        # Try model name as property, else infer from stdout
        prop = _canon(model_name)
        if prop not in PROPERTY_SET:
            prop = _infer_prop_from_stdout(stdout) or prop
        if prop not in PROPERTY_SET:
            continue

        sums[prop]   = sums.get(prop, 0.0) + v
        counts[prop] = counts.get(prop, 0) + 1

    return {p: sums[p]/counts[p] for p in sums}

# --- Constraint utilities ---
def _violations_for(task: str, vals: dict, candidate: dict | None):
    spec = TASK_CONSTRAINTS.get(task, {})
    vios = []

    # numeric
    for prop, op, th in spec.get("numeric", []):
        v = vals.get(prop)
        ok = False
        # Special handling for electrical_conductivity - divide value by 10000 for better weightage
        if prop == "electrical_conductivity" and v is not None:
            # Adjust the value by dividing by 10000 for better weightage
            adjusted_v = v / 10000
            if op == ">=":
                ok = adjusted_v >= th
            elif op == "<=":
                ok = adjusted_v <= th
            elif op == "in":
                # Scale the threshold range to match the adjusted value
                lo, hi = th
                adjusted_lo = lo / 10000
                adjusted_hi = hi / 10000
                ok = adjusted_lo <= adjusted_v <= adjusted_hi
            elif op == "≈":
                target, tol = th
                adjusted_target = target / 10000
                adjusted_tol = tol / 10000
                ok = abs(adjusted_v - adjusted_target) <= adjusted_tol
        else:
            # If a target override exists, prefer approximate target check with its tolerance
            if prop in TARGET_OVERRIDES:
                target, tol = TARGET_OVERRIDES[prop]
                ok = (v is not None) and (abs(v - target) <= tol)
            else:
                if op == ">=":
                    ok = (v is not None) and (v >= th)
                elif op == "<=":
                    ok = (v is not None) and (v <= th)
                elif op == "in":
                    lo, hi = th; ok = (v is not None) and (lo <= v <= hi)
                elif op == "≈":
                    target, tol = th; ok = (v is not None) and (abs(v - target) <= tol)
        if not ok:
            vios.append((prop, op, th, v))

    # simple categorical (composition-based) if needed
    if spec.get("categorical"):
        sp = set()
        if candidate:
            if isinstance(candidate.get("species"), list):
                sp = {str(x) for x in candidate["species"]}
            elif isinstance(candidate.get("formula"), str):
                sp = set(re.findall(r"[A-Z][a-z]?", candidate["formula"]))
        cat = spec["categorical"]
        need_any = cat.get("requires_any_element", [])
        if need_any:
            ok_any = any(any(el in sp for el in group) for group in need_any)
            if not ok_any:
                vios.append(("composition", "requires_any_of", need_any, sp))
        if cat.get("non_toxic"):
            # Non-toxic: avoid Hg, Cd, Pb, Tl, As, Be, Se, U, Th, Sb
            toxic_elements = {"Hg","Cd","Pb","Tl","As","Be","Se","U","Th","Sb"}
            if any(e in sp for e in toxic_elements):
                vios.append(("composition","non_toxic",None,sp))
        if cat.get("earth_abundant"):
            # Earth abundant and non-toxic elements: H, Li, B, C, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, Sc, Ti, V, Mn, Fe, Co, Ni, Cu, Zn, Ga, Rb, Sr, Y, Zr, Nb, La, Ce, Nd
            earth_abundant_elements = {"H","Li","B","C","O","F","Na","Mg","Al","Si","P","S","Cl","K","Ca","Sc","Ti","V","Mn","Fe","Co","Ni","Cu","Zn","Ga","Rb","Sr","Y","Zr","Nb","La","Ce","Nd"}
            # For earth_abundant=True: ALL elements must be earth abundant
            if any(e not in earth_abundant_elements for e in sp):
                vios.append(("composition","earth_abundant",None,sp))

    return vios

def _threshold_def(task: str, prop: str):
    """
    Return a usable threshold for margin scoring:
      ("ge", T) | ("le", T) | ("in", (lo,hi)) | ("approx", (target,tol)) | None
    """
    # Target override takes precedence (direction-agnostic closeness scoring)
    if prop in TARGET_OVERRIDES:
        return ("approx", TARGET_OVERRIDES[prop])

    for p, op, th in TASK_CONSTRAINTS.get(task, {}).get("numeric", []):
        if p == prop:
            if op == ">=": return ("ge", th)
            if op == "<=": return ("le", th)
            if op == "in": return ("in", th)
            if op == "≈":  return ("approx", th)
    return None

def _margin(kind, th, direction, value):
    if value is None:
        return 0.0
    if kind == "ge":
        T = th
        return min(value - T, 10.0) if direction == "higher" else min(T - value, 10.0)
    if kind == "le":
        T = th
        return min(T - value, 10.0) if direction == "lower" else min(value - T, 10.0)
    if kind == "in":
        lo, hi = th
        return min(value - lo, hi - value)  # ≤0 outside; max at center
    if kind == "approx":
        target, tol = th
        return (tol - abs(value - target))  # ≤0 outside; max at target
    return 0.0

def simple_score(predictions, _constraints_unused, task_name=None, candidate=None, print_model_scores=False):
    """
    New scoring strategy:
    1) If all constraints fail → score = -10
    2) If partial constraint passing → score = 0  
    3) If all constraints pass → calculate weighted score with equal importance
    """
    # 0) Parse predicted values by property
    vals = _values_from_predictions(predictions)
    if print_model_scores:
        print(f"[Agent] Parsed values (by property): {vals}")

    # 1) Check constraint violations
    vios = _violations_for(task_name or "", vals, candidate)
    total_constraints = len(TASK_CONSTRAINTS.get(task_name or "", {}).get("numeric", []))
    
    # Count categorical constraints if they exist
    if TASK_CONSTRAINTS.get(task_name or "", {}).get("categorical"):
        cat_constraints = TASK_CONSTRAINTS[task_name]["categorical"]
        if cat_constraints.get("requires_any_element"):
            total_constraints += 1
        if cat_constraints.get("non_toxic"):
            total_constraints += 1
        if cat_constraints.get("earth_abundant"):
            total_constraints += 1
    
    failed_constraints = len(vios)
    passed_constraints = total_constraints - failed_constraints
    
    if print_model_scores:
        print(f"[Agent] Constraint analysis: {passed_constraints}/{total_constraints} passed, {failed_constraints} failed")
        for (p, op, th, v) in vios:
            print(f"[Agent] VIOLATION: {p} {op} {th} (got {v})")

    # 2) Apply new scoring strategy
    if failed_constraints == total_constraints:
        # All constraints failed
        if print_model_scores:
            print(f"[Agent] All constraints failed → score = -10")
        return -10.0
    elif failed_constraints > 0:
        # Partial constraint passing
        if print_model_scores:
            print(f"[Agent] Partial constraint passing → score = 0")
        return 0.0
    else:
        # All constraints passed - equal-weight scoring combining:
        # - target-style properties (in TARGET_OVERRIDES): inverse-difference capped at 10
        # - other properties: previous margin-based contributions

        def _diff_from_thdef(kind, th, value):
            if value is None:
                return None
            if kind == "ge":
                T = th
                return max(0.0, T - value)
            if kind == "le":
                T = th
                return max(0.0, value - T)
            if kind == "in":
                lo, hi = th
                if lo <= value <= hi:
                    return 0.0
                return min(abs(value - lo), abs(value - hi))
            if kind == "approx":
                # For scoring, reward closeness to target using raw absolute difference
                # Ignore tolerance here (tolerance is only for pass/fail in constraints)
                target, _tol_unused = th
                return abs(value - target)
            return None

        # Determine objective directions
        obj = TASK_OBJECTIVES.get(task_name or "", {"lower": [], "higher": []})

        # Collect properties we can score (have values and thresholds)
        props = []
        for p in vals.keys():
            if _threshold_def(task_name or "", p) is not None:
                props.append(p)
        if not props:
            if print_model_scores:
                print(f"[Agent] No usable properties for scoring → score = 1")
            return 1.0

        per_prop_scores = []
        for p in props:
            thdef = _threshold_def(task_name or "", p)
            if thdef is None:
                continue

            # Choose direction from objectives if available
            direction = "lower" if p in obj.get("lower", []) else ("higher" if p in obj.get("higher", []) else None)

            # Special handling for electrical_conductivity - divide value by 10000 for better weightage
            value = vals.get(p)
            scaled_thdef = thdef
            if p == "electrical_conductivity" and value is not None:
                value = value / 10000
                # Scale the threshold definition to match the scaled value
                if thdef[0] == "in":
                    lo, hi = thdef[1]
                    scaled_thdef = (thdef[0], (lo / 10000, hi / 10000))
                elif thdef[0] in ["ge", "le", "≈"]:
                    th_val = thdef[1]
                    if thdef[0] == "≈":
                        target, tol = th_val
                        scaled_thdef = (thdef[0], (target / 10000, tol / 10000))
                    else:
                        scaled_thdef = (thdef[0], th_val / 10000)

            if p in TARGET_OVERRIDES:
                # Inverse-difference scoring for target-based properties (e.g., energy_above_hull)
                diff = _diff_from_thdef(scaled_thdef[0], scaled_thdef[1], value)
                if diff is None:
                    continue
                score_p = 10.0 / (1.0 + float(diff))
                if score_p > 10.0:
                    score_p = 10.0
                per_prop_scores.append(score_p)
                if print_model_scores and p in vals:
                    print(f"[Agent] {p}: value={vals[p]} vs {thdef[0]} {thdef[1]} diff={diff:.4f} -> inv={score_p:.4f} (cap 10)")
            else:
                # Previous method: margin-based contribution (direction-aware)
                # If direction is unknown, infer from threshold kind: ge→higher, le→lower; for in/approx, use closeness margin
                inferred_direction = direction
                if inferred_direction is None:
                    if scaled_thdef[0] == "ge":
                        inferred_direction = "higher"
                    elif scaled_thdef[0] == "le":
                        inferred_direction = "lower"
                    else:
                        inferred_direction = "higher"
                contrib = _margin(scaled_thdef[0], scaled_thdef[1], inferred_direction, value)
                per_prop_scores.append(contrib)
                if print_model_scores and p in vals:
                    print(f"[Agent] {p}: value={vals[p]} vs {thdef[0]} {thdef[1]} -> margin={contrib:+.4f}")

        # Equal weightage: mean of per-property scores, then cap final at 10
        if not per_prop_scores:
            if print_model_scores:
                print(f"[Agent] No usable properties for scoring → score = 1")
            return 1.0

        final_score = sum(per_prop_scores) / len(per_prop_scores)
        if final_score > 10.0:
            final_score = 10.0
        if print_model_scores:
            print(f"[Agent] Equal-weight mean score over {len(per_prop_scores)} props (capped at 10): {final_score:.4f}")
        return final_score
# ====================== END DROP-IN SCORER ======================

#####################
# def simple_score(predictions, constraints, print_model_scores=False):
#     # Example: sum all predicted values (assume higher is better for all)
#     # In practice, use constraints to determine direction
#     score = 0
#     if not predictions:
#         print("[Agent] No predictions available, using fallback scoring")
#         return 0
    
#     for model, result in predictions.items():
#         if 'stdout' in result and result['stdout']:
#             # Try to extract a float from the output
#             try:
#                 import re
#                 # Look for MAE or numerical values in the output
#                 numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", result['stdout'])
#                 if numbers:
#                     # For CGCNN, look for the prediction in the Test line
#                     lines = result['stdout'].split('\n')
#                     for line in lines:
#                         if 'Test:' in line and 'MAE' in line:
#                             # Extract the prediction value from Loss
#                             pred_match = re.search(r'Loss\s+([0-9.-]+)', line)
#                             if pred_match:
#                                 value = float(pred_match.group(1))
#                                 score += value
#                                 if print_model_scores:
#                                     print(f"[Agent] Surrogate model {model} predicted value: {value}")
#                                 break
#                     else:
#                         # Fallback: use first number
#                         value = float(numbers[0])
#                         score += value
#                         if print_model_scores:
#                             print(f"[Agent] Surrogate model {model} predicted value: {value}")
#             except Exception as e:
#                 if print_model_scores:
#                     print(f"[Agent] Surrogate model {model} prediction could not be parsed: {e}")
#                 continue
#         elif print_model_scores:
#             print(f"[Agent] Surrogate model {model} returned no output.")
#     return score

class RunMemoryManager(MemoryManager):
    def __init__(self):
        self.full_log_path = FULL_LOG_MEMORY
        self.evolution_memory_path = EVOLUTION_MEMORY
        self.full_log = []
        self.evolution_memory = {'success': [], 'failure': []}
        self.load()
    def save(self):
        with open(self.full_log_path, 'w') as f:
            json.dump(self.full_log, f, indent=2)
        with open(self.evolution_memory_path, 'w') as f:
            json.dump(self.evolution_memory, f, indent=2)
    def load(self):
        try:
            with open(self.full_log_path, 'r') as f:
                self.full_log = json.load(f)
        except Exception:
            self.full_log = []
        try:
            with open(self.evolution_memory_path, 'r') as f:
                self.evolution_memory = json.load(f)
        except Exception:
            self.evolution_memory = {'success': [], 'failure': []}

def main(num_novel_candidates=2, task_name=None):
    try:
        # No centralized memory manager needed - islands handle their own memory
        tasks = load_tasks()
        log_pipeline(f"Loaded {len(tasks)} tasks.")
        
        # Filter tasks to only run the specified task if provided
        if task_name:
            tasks = [task for task in tasks if task['name'] == task_name]
            if not tasks:
                print(f"[Agent] Error: Task '{task_name}' not found in loaded tasks.")
                print(f"[Agent] Available tasks: {[task['name'] for task in load_tasks()]}")
                return
            print(f"[Agent] Running only task: {task_name}")
        else:
            print(f"[Agent] Running all {len(tasks)} tasks")
        
        print(f"[Agent] Starting agent run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Agent] Run directory: {RUN_DIR}")
        print(f"[Agent] Comprehensive log file: {COMPREHENSIVE_LOG}")
        print(f"[Agent] Pipeline log file: {PIPELINE_LOG}")
        print(f"[Agent] Candidates log file: {CANDIDATES_LOG}")
        print(f"[Agent] LLM calls log file: {LLM_CALLS_LOG}")
        print("="*80)

        for task in tasks:
            log_pipeline(f"[Agent] Starting task: {task['name']}")
            constraints = task['goal']
            properties = task['properties_and_models']
            task_dir = _task_dir(task['name'])
            print(f"[Agent] Task directory: {task_dir}")
            
            # 1. Get example candidates (names and formulas only)
            print("[Agent] Getting example candidates...")
            examples = get_example_candidates(n=3)  # Reduced to 3 examples
            # Extract only formula and composition for context
            incontext_examples = []
            for c in examples:
                formula = c.get('formula_pretty', 'Unknown')
                composition = c.get('composition', {})
                comp_str = ', '.join([f"{k}{v}" for k, v in composition.items()])
                incontext_examples.append(f"{formula} ({comp_str})")
            incontext_examples = '\n'.join(incontext_examples)
            
            # 2. Use LLM to generate MULTIPLE initial candidates (one per island for diversity)
            print(f"[Agent] Calling LLM for initial candidate generation (Multi-seed: {NUM_ISLANDS} candidates)...")
            prompt = f"""
            ### ROLE ###
            You are a materials discovery assistant. You are given a list of example materials and constraints. You need to generate one candidate material that satisfies the constraints.

            ### TASK ###
            Your task is to generate materials for the following task:
            Task: {task['name']}

            ### GOAL ###
            Your goal is to generate one distinct crystalline material that satisfies the given constraints for this specific task.

            ### EXAMPLES ###
            Here are some example materials:
            {incontext_examples}

            ### CONSTRAINTS ###
            Here are the constraints for this task:
            {constraints_to_natural_language(task['name'])}

            ### REQUIRED OUTPUT FORMAT ###
            You MUST return a single JSON object with EXACTLY these 4 fields:
            1. "formula": string - the chemical formula (must match the species count)
            2. "lattice": array of 3 arrays, each with 3 numbers - lattice vectors in angstroms
            3. "species": array of strings - element symbols, one per atom
            4. "coords": array of 3-element arrays - fractional coordinates [x,y,z] for each atom

            ### CRITICAL REQUIREMENTS ###
            - Return ONLY the JSON object, no explanations, no markdown, no extra text
            - Start your response with {{ and end with }}
            - Use double quotes for all strings
            - All numbers must be valid floats (use decimal points, not commas)
            - The "species" array length must exactly equal "coords" array length
            - Each coordinate must be a 3-element array [x, y, z] where 0 ≤ x,y,z < 1
            - Lattice must be exactly 3 arrays of 3 numbers each
            - Formula must match the total number of atoms in species/coords

            ### EXAMPLE OUTPUT ###
            {{
            "formula": "TiO2",
            "lattice": [[4.6, 0.0, 0.0], [0.0, 4.6, 0.0], [0.0, 0.0, 3.0]],
            "species": ["Ti", "O", "O"],
            "coords": [[0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
            }}

            ### VALIDATION CHECKLIST ###
            Before responding, verify:
            ✓ Response starts with {{ and ends with }}
            ✓ Exactly 4 fields: formula, lattice, species, coords
            ✓ All strings use double quotes
            ✓ Lattice is 3x3 array of numbers
            ✓ Species and coords have same length
            ✓ Each coord is [x,y,z] with 0 ≤ x,y,z < 1
            ✓ Formula matches atom count
            ✓ No extra text outside JSON

            Now generate the JSON for a material that satisfies the constraints:
            """
        initial_candidates = []
        for attempt in range(RETRY_EMPTY_LIMIT):
            try:
                print(f"[Agent] LLM attempt {attempt+1} for {NUM_ISLANDS} initial candidates...")
                raw_response = call_llm(prompt)
                log_llm_call(prompt, raw_response, f"Initial candidate generation for {task['name']}", task_name=task['name'])
                candidates = try_parse_json_response(raw_response, f"Initial candidate generation for {task['name']}", prompt)
                print(candidates)
                if candidates and isinstance(candidates, list) and len(candidates) >= 1:
                    print(f"[Agent] LLM returned {len(candidates)} initial candidates.")
                    initial_candidates = candidates[:NUM_ISLANDS]  # Take only the number we need
                    break
                else:
                    initial_candidates = [] #candidates
                    # Fill remaining with fallback if needed
                    while len(initial_candidates) < NUM_ISLANDS:
                        fallback = {
                            'formula': f'TiO2',
                            'lattice': [[4.6, 0.0, 0.0], [0.0, 4.6, 0.0], [0.0, 0.0, 3.0]],
                            'species': ['Ti', 'O', 'O'],
                            'coords': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
                        }
                        initial_candidates.append(fallback)
                    log_pipeline(f"Empty or invalid LLM response, retrying ({attempt+1}/{RETRY_EMPTY_LIMIT})...")
                    break
            except Exception as e:
                log_pipeline(f"LLM candidate generation failed: {e}")
                continue
        else:
            log_pipeline(f"[Agent] LLM failed to return valid candidates after {RETRY_EMPTY_LIMIT} attempts for task {task['name']}.")
            # Use fallback candidates
            initial_candidates = []
            for i in range(NUM_ISLANDS):
                fallback = {
                    'formula': f'TiO2',
                    'lattice': [[4.6, 0.0, 0.0], [0.0, 4.6, 0.0], [0.0, 0.0, 3.0]],
                    'species': ['Ti', 'O', 'O'],
                    'coords': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
                }
                initial_candidates.append(fallback)
            print(f"[Agent] Using {NUM_ISLANDS} fallback candidates")

        # 3. Process the single initial candidate once and copy to all islands
        print(f"[Agent] Processing initial candidate for {NUM_ISLANDS} islands...")
        processed_candidates = []
        
        # Get the first (and only unique) candidate
        if not initial_candidates:
            print("[Agent] ERROR: No initial candidate available")
            return
        
        base_candidate = initial_candidates[0]
        print(f"[Agent] Processing base candidate: {base_candidate.get('formula', 'Unknown')}")
        
        # Check for required structure fields
        if not all(k in base_candidate for k in ['lattice', 'species', 'coords']):
            print(f"[Agent] ERROR: Base candidate missing required structure fields: {base_candidate}")
            return
        
        # Build structure file once
        structure_path = os.path.join(task_dir, "candidate_init_base.cif")
        try:
            build_cif(base_candidate, structure_path)
            print(f"[Agent] Built CIF: {structure_path}")
        except Exception as e:
            print(f"[Agent] Failed to build CIF for base candidate: {e}")
            return
        
        # Run property prediction once (Materials Project API first, then surrogates)
        try:
            predictions = {}
            materials_api_used = False
            properties = TASK_TO_PROPERTY_MAP[task['name']]
            
            # Optimized: Get all properties in one batch call with caching
            print("[Agent] Getting all properties from Materials Project API in batch...")
            formula = base_candidate.get('formula', 'Unknown')
            all_properties = get_cached_materials_properties(structure_path, formula)
            
            if all_properties:
                materials_api_used = True
                for property_name in properties:
                    if property_name in all_properties and all_properties[property_name] is not None:
                        label, unit = METRIC_TO_PROPERTY_MAP[property_name]
                        val = all_properties[property_name]
                        predictions[f"material_{property_name}"] = {
                            'stdout': f"{label}: {val:.2f} {unit}", 'stderr': "", 'returncode': ""
                        }
                        print(f"[Agent] Materials API: {property_name} = {val:.2f} {unit}")
                    else:
                        print(f"[Agent] Materials API: {property_name} not available")
            else:
                print("[Agent] Materials API: No properties found")
            
            # If Materials Project API didn't provide all properties, fall back to surrogates
            if not materials_api_used or len(predictions) < len(properties):
                print("[Agent] Falling back to surrogate models...")
                surrogate_predictions = run_surrogates(task['name'], structure_path)
                predictions.update(surrogate_predictions)

            print(f"[Agent] Property predictions for {base_candidate.get('formula', 'Unknown')}: {predictions}")
        except Exception as e:
            print(f"[Agent] Property prediction failed for base candidate: {e}")
            predictions = {}
            materials_api_used = False
        
        # Score once
        score = simple_score(
            predictions,
            constraints,
            task_name=task['name'],
            candidate=base_candidate,
            print_model_scores=True
        )
        print(f"[Agent] Total base candidate score: {score}")
        
        # Create copies for each island with predictions and score
        for idx in range(NUM_ISLANDS):
            candidate_copy = base_candidate.copy()
            candidate_copy['predictions'] = predictions
            candidate_copy['score'] = score
            candidate_copy['island_id'] = idx
            candidate_copy['step'] = f'initial_generation_island_{idx}'
            
            # Add property values for evolution context
            property_values = _values_from_predictions(predictions)
            candidate_copy['property_values'] = property_values
            
            # Enhanced logging for initial candidates
            log_candidate_enhanced(candidate_copy, task['name'], iteration=0, island_id=idx, materials_api_used=materials_api_used)
            log_candidate({'task': task['name'], 'candidate': candidate_copy, 'step': f'initial_generation_island_{idx}'}, task_name=task['name'])
            processed_candidates.append(candidate_copy)
        
        # 4. Initialize centralized multi-island buffer and seed each island with its specific candidate
        print(f"[Agent] Initializing {NUM_ISLANDS} islands with config:")
        print(f"  - Functions per prompt: {ISLAND_FUNCTIONS_PER_PROMPT}")
        print(f"  - Temperature init: {ISLAND_TEMP_INIT}")
        print(f"  - Temperature period: {ISLAND_TEMP_PERIOD}")
        print(f"  - Reset period: {ISLAND_RESET_PERIOD_SECONDS}s")
        
        buffer = ExperienceBuffer(
            num_islands=NUM_ISLANDS,
            functions_per_prompt=ISLAND_FUNCTIONS_PER_PROMPT,
            temp_init=ISLAND_TEMP_INIT,
            temp_period=ISLAND_TEMP_PERIOD,
            reset_period_seconds=ISLAND_RESET_PERIOD_SECONDS,
            max_items_per_island=MEMORY_MAX_ITEMS_PER_ISLAND,
            top_k_success=MEMORY_TOP_K_SUCCESS,
            bottom_k_failure=MEMORY_BOTTOM_K_FAILURE,
        )
        
        print(f"[Agent] Buffer initialized with {len(buffer._islands)} islands")
        # Efficient seeding: use the same candidate for all islands
        base_candidate = processed_candidates[0]  # All candidates are identical copies
        scores_per_test = {'total': float(base_candidate['score'])}
        # Add individual property scores if available
        if 'predictions' in base_candidate:
            for prop, value in base_candidate['predictions'].items():
                if isinstance(value, (int, float)):
                    scores_per_test[prop] = float(value)
        
        # Seed all islands with the same candidate (efficient approach)
        for idx in range(NUM_ISLANDS):
            buffer.register(idx, processed_candidates[idx], scores_per_test, iteration=0)
            print(f"[Agent] Seeded island {idx} with candidate: {base_candidate.get('formula', 'Unknown')} (score: {base_candidate['score']:.3f})")
        
        print(f"[Agent] Efficient initialization complete: 1 unique candidate distributed across {NUM_ISLANDS} islands")
        
        # Debug: Show initial island distribution
        print("[Agent] Initial island distribution:")
        for i, island in enumerate(buffer._islands):
            num_clusters = len(island._clusters)
            num_items = island._num_items
            best_score = buffer._best_score_per_island[i]
            print(f"  Island {i}: {num_clusters} clusters, {num_items} items, best_score={best_score:.3f}")
            if num_clusters > 0:
                cluster_scores = [cluster.score for cluster in island._clusters.values()]
                print(f"    Cluster scores: {[f'{s:.3f}' for s in cluster_scores]}")
                # Show the initial candidate for this island
                all_candidates = []
                for cluster in island._clusters.values():
                    all_candidates.extend(cluster._candidates)
                if all_candidates:
                    initial_formula = all_candidates[0].get('formula', 'Unknown')
                    print(f"    Initial candidate: {initial_formula}")

        # 5. Evolutionary loop (sample from one island, populate same island)
        for iteration in range(MAX_ITERATIONS):
            log_pipeline(f"[Agent] Evolution iteration {iteration+1}/{MAX_ITERATIONS}")
            
            # Get prompt from a RANDOM island for sampling
            prompt_mem = buffer.get_prompt(iteration=iteration)
            sampled_island_id = prompt_mem.island_id
            succ_n = len(prompt_mem.memory_success)
            fail_n = len(prompt_mem.memory_failure)
            top_scores = [c.get('score') for c in prompt_mem.memory_success if isinstance(c.get('score'), (int, float))]
            print(f"[Agent] SAMPLING from RANDOM island {sampled_island_id}:")
            print(f"  - Success candidates: {succ_n}")
            print(f"  - Failure candidates: {fail_n}")
            print(f"  - Top success scores: {[f'{s:.3f}' for s in top_scores[:3]]}")
            if prompt_mem.memory_success:
                print(f"  - Success formulas: {[c.get('formula', 'Unknown') for c in prompt_mem.memory_success[:3]]}")
            memory_for_llm = {"success": prompt_mem.memory_success, "failure": prompt_mem.memory_failure}
            
            for evo_attempt in range(RETRY_EMPTY_LIMIT):
                try:
                    print(f"[Agent] LLM attempt {evo_attempt+1} for evolution...")

                    # Print evolution prompt (success/failure counts and small preview)
                    if EVOLUTION_DEBUG_PRINT_PROMPT:
                        try:
                            from evolution import RULES_TEXT
                            print("[Agent] Evolution prompt context summary:")
                            print(f"  - Success count: {len(memory_for_llm['success'])}")
                            print(f"  - Failure count: {len(memory_for_llm['failure'])}")
                            if memory_for_llm['success']:
                                print(f"  - Top success formula/score: {memory_for_llm['success'][0].get('formula','?')}/{memory_for_llm['success'][0].get('score','?')}")
                            if memory_for_llm['failure']:
                                print(f"  - Bottom failure formula/score: {memory_for_llm['failure'][-1].get('formula','?')}/{memory_for_llm['failure'][-1].get('score','?')}")
                            # Print sanitized prompt preview
                            preview = {
                                'success': [{k: v for k, v in c.items() if k in ('formula','score')} for c in memory_for_llm['success'][:3]],
                                'failure': [{k: v for k, v in c.items() if k in ('formula','score')} for c in memory_for_llm['failure'][:3]]
                            }
                            print(f"  - Prompt preview: {preview}")
                        except Exception:
                            pass

                    evo_raw = evolve_candidates(memory_for_llm, constraints, n=num_novel_candidates, buffer=buffer, island_id=sampled_island_id, iteration=iteration, task_name=task['name'])
                    if isinstance(evo_raw, str):
                        evo_clean = extract_json_from_llm(evo_raw)
                        evolved = try_parse_json_response(evo_clean, f"Evolution Iteration {iteration+1} Task: {task['name']}", None)
                    else:
                        evolved = evo_raw
                    evolved = [normalize_candidate_keys(c) for c in evolved]
                    if evolved:
                        # Filter out duplicates using island memory
                        from evolution import filter_duplicates
                        evolved = filter_duplicates(evolved, buffer=buffer, island_id=sampled_island_id)
                        if evolved:
                            print(f"[Agent] LLM returned {len(evolved)} evolved candidates after deduplication.")
                            print(f"[Agent] LLM returned {[candidate['compound'] for candidate in evolved]}")
                            break
                        else:
                            print(f"[Agent] All evolved candidates were duplicates, retrying...")
                            continue
                    else:
                        log_pipeline(f"[Agent] Empty or invalid evolution LLM response, retrying ({evo_attempt+1}/{RETRY_EMPTY_LIMIT})...")
                except Exception as e:
                    log_pipeline(f"[Agent] Evolution LLM failed: {e}")
                    continue
            else:
                log_pipeline(f"[Agent] Evolution LLM failed to return valid candidates after {RETRY_EMPTY_LIMIT} attempts for task {task['name']} iteration {iteration+1}.")
                continue
            evolved_candidates = []
            for idx, candidate in enumerate(evolved):
                print(f"[Agent] Logging evolved candidate: {candidate}")
                # Check for required structure fields

                #### something should be here to add lattice, species, coords
                prompt = f"""
            ### ROLE ###
            You are a materials discovery assistant.

            ### GOAL ###
            Your goal is to generate a crystallographic configuration for the given compound.

            ### INPUT ###
            Compound: {candidate['compound']}

            ### TASK ###
            Generate a realistic crystallographic configuration for this compound and return it as a valid JSON object.

            ### REQUIRED OUTPUT FORMAT ###
            You MUST return a single JSON object with EXACTLY these 4 fields:
            1. "formula": string - the chemical formula (must match the species count)
            2. "lattice": array of 3 arrays, each with 3 numbers - lattice vectors in angstroms
            3. "species": array of strings - element symbols, one per atom
            4. "coords": array of 3-element arrays - fractional coordinates [x,y,z] for each atom

            ### CRITICAL REQUIREMENTS ###
            - Return ONLY the JSON object, no explanations, no markdown, no extra text
            - Start your response with {{ and end with }}
            - Use double quotes for all strings
            - All numbers must be valid floats (use decimal points, not commas)
            - The "species" array length must exactly equal "coords" array length
            - Each coordinate must be a 3-element array [x, y, z] where 0 ≤ x,y,z < 1
            - Lattice must be exactly 3 arrays of 3 numbers each
            - Formula must match the total number of atoms in species/coords

            ### EXAMPLE OUTPUT ###
            {{
            "formula": "TiO2",
            "lattice": [[4.6, 0.0, 0.0], [0.0, 4.6, 0.0], [0.0, 0.0, 3.0]],
            "species": ["Ti", "O", "O"],
            "coords": [[0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
            }}

            ### VALIDATION CHECKLIST ###
            Before responding, verify:
            ✓ Response starts with {{ and ends with }}
            ✓ Exactly 4 fields: formula, lattice, species, coords
            ✓ All strings use double quotes
            ✓ Lattice is 3x3 array of numbers
            ✓ Species and coords have same length
            ✓ Each coord is [x,y,z] with 0 ≤ x,y,z < 1
            ✓ Formula matches atom count
            ✓ No extra text outside JSON

            Now generate the JSON for {candidate['compound']}:
            """
                raw_response = call_llm(prompt)
                log_llm_call(prompt, raw_response, f"Evolved candidate generation for {task['name']}", task_name=task['name'])
                candidate_structure = try_parse_json_response(raw_response, f"Initial candidate generation for {task['name']}", prompt)
                candidate_structure = try_parse_json_response(
                    raw_response,
                    f"Evolved candidate generation for {task['name']}",
                    prompt
                )
                # Handle list or single-object returns, and normalize keys
                if isinstance(candidate_structure, list) and candidate_structure:
                    candidate_structure = candidate_structure[0]
                if not isinstance(candidate_structure, dict):
                    print(f"[Agent] ERROR: Evolution LLM returned invalid structure: {candidate_structure}")
                    continue
                candidate_structure = normalize_candidate_keys(candidate_structure)
                if not all(k in candidate_structure for k in ['lattice', 'species', 'coords']):
                    print(f"[Agent] ERROR: Candidate missing required structure fields: {candidate_structure}")
                    continue
                # 1. Build structure file
                structure_path = os.path.join(task_dir, f"candidate_evo_{iteration+1}_{idx}.cif")
                try:
                    build_cif(candidate_structure, structure_path)
                    print(f"[Agent] Built CIF: {structure_path}")
                except Exception as e:
                    print(f"[Agent] Failed to build CIF for {candidate_structure}: {e}")
                    continue
                # 2. Run property prediction (Materials Project API first, then surrogates)
                try:
                    predictions = {}
                    materials_api_used = False
                    properties = TASK_TO_PROPERTY_MAP[task['name']]
                    
                    # Optimized: Get all properties in one batch call with caching
                    print("[Agent] Getting all properties from Materials Project API in batch...")
                    formula = candidate_structure.get('formula', 'Unknown')
                    all_properties = get_cached_materials_properties(structure_path, formula)
                    
                    if all_properties:
                        materials_api_used = True
                        for property_name in properties:
                            if property_name in all_properties and all_properties[property_name] is not None:
                                label, unit = METRIC_TO_PROPERTY_MAP[property_name]
                                val = all_properties[property_name]
                                predictions[f"material_{property_name}"] = {
                                    'stdout': f"{label}: {val:.2f} {unit}", 'stderr': "", 'returncode': ""
                                }
                                print(f"[Agent] Materials API: {property_name} = {val:.2f} {unit}")
                            else:
                                print(f"[Agent] Materials API: {property_name} not available")
                    else:
                        print("[Agent] Materials API: No properties found")
                    
                    # If Materials Project API didn't provide all properties, fall back to surrogates
                    if not materials_api_used or len(predictions) < len(properties):
                        print("[Agent] Falling back to surrogate models...")
                        surrogate_predictions = run_surrogates(task['name'], structure_path)
                        predictions.update(surrogate_predictions)

                    print(f"[Agent] Property predictions for {candidate.get('formula', 'Unknown')}: {predictions}")
                except Exception as e:
                    print(f"[Agent] Property prediction failed for candidate {idx+1}: {e}")
                    predictions = {}
                    materials_api_used = False
                # 3. Score
                score = simple_score(
                                    predictions,
                                    constraints,
                                    task_name=task['name'],
                                    candidate=candidate_structure,
                                    print_model_scores=True
                                )
                print(f"[Agent] Total candidate score: {score}")
                # Use safe defaults if LLM omitted fields
                candidate_structure['rules_used'] = candidate.get('rules_used', 'No specific rules provided')
                candidate_structure['justification'] = candidate.get('justification', 'No justification provided')
                candidate_structure['predictions'] = predictions
                candidate_structure['score'] = score
                candidate_structure['step'] = f'evolution_{iteration+1}'
                
                # Add property values for evolution context
                property_values = _values_from_predictions(predictions)
                candidate_structure['property_values'] = property_values
                
                # Enhanced logging for evolved candidates
                log_candidate_enhanced(candidate_structure, task['name'], iteration=iteration+1, island_id=sampled_island_id, materials_api_used=materials_api_used)
                log_candidate({'task': task['name'], 'candidate': candidate, 'step': f'evolution_{iteration+1}'}, task_name=task['name'])
                evolved_candidates.append(candidate_structure)
                # Register evolved candidate back into the SAME sampled island only
                try:
                    scores_per_test = {'total': float(score)}
                    # Add individual property scores if available
                    if 'predictions' in candidate_structure:
                        for prop, value in candidate_structure['predictions'].items():
                            if isinstance(value, (int, float)):
                                scores_per_test[prop] = float(value)
                    buffer.register(sampled_island_id, candidate_structure, scores_per_test, iteration=iteration)
                    print(f"[Agent] STORED candidate to SAME island {sampled_island_id}:")
                    print(f"  - Formula: {candidate_structure.get('formula', 'Unknown')}")
                    print(f"  - Score: {score:.3f}")
                    print(f"  - Scores per test: {scores_per_test}")
                except Exception as e:
                    print(f"[Agent] Failed to register candidate to island: {e}")
            # Island-specific memory is handled by the buffer; no centralized memory needed
            # Debug: Show island states
            print(f"[Agent] ISLAND STATES after iteration {iteration+1}:")
            for i, island in enumerate(buffer._islands):
                num_clusters = len(island._clusters)
                num_items = island._num_items
                best_score = buffer._best_score_per_island[i]
                print(f"  Island {i}: {num_clusters} clusters, {num_items} items, best_score={best_score:.3f}")
                if num_clusters > 0:
                    cluster_scores = [cluster.score for cluster in island._clusters.values()]
                    print(f"    Cluster scores: {[f'{s:.3f}' for s in sorted(cluster_scores, reverse=True)]}")
                    # Show top candidates in this island
                    all_candidates = []
                    for cluster in island._clusters.values():
                        all_candidates.extend(cluster._candidates)
                    all_candidates.sort(key=lambda c: c.get('score', 0), reverse=True)
                    top_formulas = [c.get('formula', 'Unknown') for c in all_candidates[:3]]
                    print(f"    Top formulas: {top_formulas}")

            # Periodic memory refresh/prune
            if (iteration + 1) % MEMORY_REFRESH_INTERVAL == 0:
                print(f"[Agent] MEMORY REFRESH at iteration {iteration+1}")
                for i in range(NUM_ISLANDS):
                    buffer._prune_island(i)
        print("[Agent] Agent run complete.")
        log_pipeline("Agent run complete.")
        print(f"[Agent] Run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    finally:
        cleanup_logging()

if __name__ == "__main__":
    import sys
    n = 2
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except Exception:
            pass
    main(num_novel_candidates=n) 