#!/usr/bin/env python3
"""
Analyze Stability of Valid Candidates in MatterGen Results

This script processes all candidate files in the scoring_output directories
and analyzes the stability properties of VALID candidates, focusing on:
- First checks validity using task constraints
- For valid candidates, calculates energy above hull using CIF files
- Formation energy (thermodynamic stability)
- Energy above hull (phase stability)
- Materials API usage for stability validation

A candidate is considered stable if it meets stability criteria.
"""

import os
import json
import glob
import subprocess
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add the material directory to the path to import modules (if needed)
# This can be configured via environment variable MATERIAL_DIR if required
material_dir = os.environ.get('MATERIAL_DIR')
if material_dir:
    sys.path.append(material_dir)

# Task constraints from scoring.py
TASK_CONSTRAINTS = {
    "Hard, Stiff Ceramics": {
        "numeric": [
            ("bulk_modulus", "in", (100.0, 300.0)),
            ("shear_modulus", "in", (60.0, 200.0)),
        ]
    },
    "Photovoltaic Absorbers": {
        "numeric": [
            ("band_gap", "in", (0.7, 2.0)),
            ("formation_energy", "<=", 0.0),
        ],
        "categorical": {"earth_abundant": True, "non_toxic": True},
    },
    "Solid-State Electrolytes": {
        "numeric": [
            ("formation_energy", "<=", -1.0),
            ("energy_above_hull", "<=", 2.0),
            ("band_gap", ">=", 2.0),
        ],
        "categorical": {"requires_any_element": [["Li"], ["Na"], ["K"], ["Mg"], ["Ca"], ["Al"]]},
    },
    "Stable Wide-Bandgap Semiconductors": {
        "numeric": [
            ("band_gap", ">=", 2.5),
            ("formation_energy", "<=", -1.0),
            ("energy_above_hull", "<=", 2.0),
        ]
    },
    "SAW/BAW Acoustic Substrates": {
        "numeric": [
            ("shear_modulus", "in", (25.0, 150.0)),
            ("dielectric_constant", "in", (3.7, 95.0)),
        ]
    },
    "Structural Materials for Aerospace": {
        "numeric": [
            ("bulk_modulus", "in", (100.0, 300.0)),
            ("shear_modulus", "in", (60.0, 200.0)),
            ("formation_energy", "<", 0.0),
        ]
    },
    "High-k Dielectrics": {
        "numeric": [
            ("dielectric_constant", "in", (10.0, 90.0)),
            ("band_gap", "in", (2.5, 6.5)),
        ]
    },
    "Piezo Energy Harvesters": {
        "numeric": [
            ("piezo_max_dij", ">=", 8.0),
            ("piezo_max_dielectric", "in", (10.0, 8000.0)),
        ]
    },
    "Hard Coating Materials": {
        "numeric": [
            ("bulk_modulus", "in", (200.0, 500.0)),
            ("shear_modulus", "in", (100.0, 300.0)),
        ]
    },
    "Acousto-optic Hybrids": {
        "numeric": [
            ("piezo_max_dij", "in", (3.0, 9.0)),
            ("piezo_max_dielectric", "in", (8.0, 85.0)),
        ]
    },
    "Electrically Insulating Dielectrics": {
        "numeric": [
            ("band_gap", ">=", 2.5),
            ("dielectric_constant", ">=", 8.0),
        ]
    },
    "Transparent Conductors": {
        "numeric": [
            ("band_gap", ">", 1.5),
            ("electrical_conductivity", "in", (500.0, 30000.0)),
        ]
    },
    "Low_Density_Structural_Aerospace": {
        "numeric": [
            ("density", "<=", 3.5),
            ("shear_modulus", "in", (65.0, 195.0)),
        ]
    },
    "Toxic_Free_Perovskite_Oxide": {
        "numeric": [
            ("band_gap", ">=", 2.0),
            ("bulk_modulus", "in", (90.0, 135.0)),
        ],
        "categorical": {"exclude_elements": ["Pb", "Cd", "Hg", "Tl", "Be", "As", "Sb", "Se", "U", "Th"]},
    },
}

# Stability constraints - focusing on thermodynamic and phase stability
STABILITY_CONSTRAINTS = {
    "formation_energy": {
        "stable": ("<=", -1.0),           # Stable: formation_energy <= -1.0 eV/atom
        "marginally_stable": ("<=", 0.0), # Marginally stable: formation_energy <= 0.0 eV/atom
        "unstable": (">", 0.0)            # Unstable: formation_energy > 0.0 eV/atom
    },
    "energy_above_hull": {
        "stable": ("<=", 0.1),       # Stable: energy_above_hull <= 0.1 eV/atom
        "marginally_stable": ("<=", 0.5), # Marginally stable: energy_above_hull <= 0.5 eV/atom
        "unstable": (">", 0.5)            # Unstable: energy_above_hull > 0.5 eV/atom
    }
}

def check_numeric_constraint(value: float, operator: str, threshold: Any) -> bool:
    """Check if a numeric value satisfies a constraint."""
    if value is None:
        return False
    
    if operator == ">=":
        return value >= threshold
    elif operator == "<=":
        return value <= threshold
    elif operator == ">":
        return value > threshold
    elif operator == "<":
        return value < threshold
    elif operator == "in":
        return threshold[0] <= value <= threshold[1]
    else:
        raise ValueError(f"Unknown operator: {operator}")

def check_categorical_constraint(candidate: Dict, constraint_name: str, constraint_value: Any) -> bool:
    """Check if a candidate satisfies a categorical constraint."""
    if constraint_name == "earth_abundant":
        earth_abundant_elements = {"H", "Li", "B", "C", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "La", "Ce", "Nd"}
        
        formula = candidate.get("compound_formula", "")
        if not formula:
            return False
        
        # Extract elements from formula
        import re
        elements = set(re.findall(r'[A-Z][a-z]?', formula))
        
        for element in elements:
            if element not in earth_abundant_elements:
                return False
        return True
        
    elif constraint_name == "non_toxic":
        toxic_elements = {"Pb", "Cd", "Hg", "Tl", "Be", "As", "Sb", "Se", "U", "Th"}
        
        formula = candidate.get("compound_formula", "")
        if not formula:
            return False
        
        # Extract elements from formula
        import re
        elements = set(re.findall(r'[A-Z][a-z]?', formula))
        
        for element in elements:
            if element in toxic_elements:
                return False
        return True
        
    elif constraint_name == "requires_any_element":
        formula = candidate.get("compound_formula", "")
        if not formula:
            return False
        
        # Extract elements from formula
        import re
        elements = set(re.findall(r'[A-Z][a-z]?', formula))
        
        required_elements = constraint_value
        for element_group in required_elements:
            if any(element in elements for element in element_group):
                return True
        return False
    
    elif constraint_name == "exclude_elements":
        formula = candidate.get("compound_formula", "")
        if not formula:
            return False
        
        # Extract elements from formula
        import re
        elements = set(re.findall(r'[A-Z][a-z]?', formula))
        
        exclude_elements = constraint_value
        for element in exclude_elements:
            if element in elements:
                return False
        return True
    else:
        print(f"Warning: Unknown categorical constraint: {constraint_name}")
        return True

def is_candidate_valid(candidate: Dict) -> bool:
    """Check if a candidate is valid based on its task constraints."""
    task_name = candidate.get("task_name")
    if not task_name:
        return False
    
    # Handle special cases for task names
    special_cases = {
        "Toxic_Free_Perovskite_Oxide": "Toxic_Free_Perovskite_Oxide",
        "Low_Density_Structural_Aerospace": "Low_Density_Structural_Aerospace"
    }
    
    if task_name not in TASK_CONSTRAINTS:
        # Try to map common variations
        task_name_mapped = task_name.replace(" ", "_").replace("-", "_").replace("/", "_")
        if task_name_mapped in TASK_CONSTRAINTS:
            task_name = task_name_mapped
        elif task_name in special_cases:
            task_name = special_cases[task_name]
        else:
            return False
    
    constraints = TASK_CONSTRAINTS[task_name]
    property_values = candidate.get("property_values", {})
    
    # Check numeric constraints
    if "numeric" in constraints:
        for prop_name, operator, threshold in constraints["numeric"]:
            if prop_name not in property_values:
                return False
            
            value = property_values[prop_name]
            if not check_numeric_constraint(value, operator, threshold):
                return False
    
    # Check categorical constraints
    if "categorical" in constraints:
        for constraint_name, constraint_value in constraints["categorical"].items():
            if not check_categorical_constraint(candidate, constraint_name, constraint_value):
                return False
    
    return True

def calculate_energy_above_hull_materials_api(cif_file_path: str) -> float:
    """Calculate energy above hull using Materials Project API."""
    try:
        import agent.property_extractor
        if hasattr(agent.property_extractor, 'get_energy_above_hull'):
            func = getattr(agent.property_extractor, 'get_energy_above_hull')
            val = func(cif_file_path)
            if val is not None:
                return float(val)
        return None
    except Exception as e:
        print(f"    ❌ Materials API energy above hull calculation failed: {e}")
        return None

def calculate_energy_above_hull_alignn(cif_file_path: str) -> float:
    """Calculate energy above hull using ALIGNN model via jv_ehull_alignn with mat_sci conda environment."""
    try:
        # Use the correct ALIGNN command with mat_sci conda environment
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alignn_dir = os.path.join(script_dir, "..", "..", "surrogate_models", "alignn")
        
        # Convert to absolute path for the CIF file
        abs_cif_path = os.path.abspath(cif_file_path)
        
        result = subprocess.run([
            'bash', '-c', 
            f'cd {alignn_dir} && '
            f'export CUDA_VISIBLE_DEVICES="" && '
            f'conda run -n mat_sci python alignn/pretrained.py --model_name jv_ehull_alignn --file_format cif --file_path {abs_cif_path}'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            # Parse the output to extract energy above hull
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'Energy above hull:' in line or 'energy_above_hull:' in line or 'ehull:' in line.lower():
                    try:
                        # Extract the numeric value
                        import re
                        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", line)
                        if numbers:
                            value = float(numbers[0])
                            return value
                    except (IndexError, ValueError):
                        continue
        return None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"    ❌ ALIGNN energy above hull calculation failed: {e}")
        return None

def calculate_energy_above_hull(cif_file_path: str, materials_api_used: bool = False) -> float:
    """Calculate energy above hull using Materials API or ALIGNN model."""
    if materials_api_used:
        # Try Materials API first
        result = calculate_energy_above_hull_materials_api(cif_file_path)
        if result is not None:
            return result
        # Fall back to ALIGNN if Materials API fails
        result = calculate_energy_above_hull_alignn(cif_file_path)
        return result
    else:
        # Use ALIGNN model
        result = calculate_energy_above_hull_alignn(cif_file_path)
        return result

def get_stability_level(formation_energy: float, energy_above_hull: float = None) -> str:
    """Determine stability level based on formation energy and energy above hull."""
    # If we have energy above hull, use it for classification
    if energy_above_hull is not None:
        if energy_above_hull <= 0.1:
            eh_level = "stable"
        elif energy_above_hull <= 0.5:
            eh_level = "marginally_stable"
        else:
            eh_level = "unstable"
        
        # If we also have formation energy, combine both criteria
        if formation_energy is not None:
            if formation_energy <= -1.0:
                fe_level = "stable"
            elif formation_energy <= 0.0:
                fe_level = "marginally_stable"
            else:
                fe_level = "unstable"
            
            # Use the more conservative (less stable) classification
            if fe_level == "unstable" or eh_level == "unstable":
                return "unstable"
            elif fe_level == "marginally_stable" or eh_level == "marginally_stable":
                return "marginally_stable"
            else:
                return "stable"
        else:
            # Only energy above hull available
            return eh_level
    
    # Only formation energy available
    if formation_energy is not None:
        if formation_energy <= -1.0:
            return "stable"
        elif formation_energy <= 0.0:
            return "marginally_stable"
        else:
            return "unstable"
    
    # Neither available
    return "unknown"

def analyze_candidate_stability(candidate: Dict, cif_dir: str, quiet: bool = True) -> Dict[str, Any]:
    """Analyze stability properties of a single candidate."""
    if not quiet:
        print(f"\n{'='*60}")
        print(f"ANALYZING CANDIDATE: {candidate.get('compound_formula', 'Unknown')}")
        print(f"Iteration: {candidate.get('iteration', 'Unknown')}")
        print(f"Task: {candidate.get('task_name', 'Unknown')}")
        print(f"{'='*60}")
    
    property_values = candidate.get("property_values", {})
    
    # First check if candidate is valid
    is_valid = is_candidate_valid(candidate)
    
    formation_energy = property_values.get("formation_energy")
    energy_above_hull = property_values.get("energy_above_hull")
    materials_api_used = candidate.get("materials_api_used", False)
    
    # If valid and has CIF file, calculate energy above hull
    calculated_energy_above_hull = None
    if is_valid and candidate.get("cif_file"):
        cif_filename = candidate["cif_file"]
        # CIF files are in the specified directory (example or generated_cif)
        cif_file_path = os.path.join(cif_dir, cif_filename)
        
        if os.path.exists(cif_file_path):
            calculated_energy_above_hull = calculate_energy_above_hull(cif_file_path, materials_api_used)
        else:
            if not quiet:
                print(f"Warning: CIF file not found: {cif_file_path}")
    
    # Use calculated energy above hull if available, otherwise use existing value
    final_energy_above_hull = calculated_energy_above_hull if calculated_energy_above_hull is not None else energy_above_hull
    
    # Determine stability level
    if is_valid:
        stability_level = get_stability_level(formation_energy, final_energy_above_hull)
    else:
        stability_level = "invalid"
    
    stability_analysis = {
        "iteration": candidate.get("iteration"),
        "compound_formula": candidate.get("compound_formula"),
        "task_name": candidate.get("task_name"),
        "is_valid": is_valid,
        "formation_energy": formation_energy,
        "energy_above_hull": final_energy_above_hull,
        "calculated_energy_above_hull": calculated_energy_above_hull,
        "materials_api_used": materials_api_used,
        "stability_level": stability_level,
        "has_formation_energy": formation_energy is not None,
        "has_energy_above_hull": final_energy_above_hull is not None,
        "surrogate_used": candidate.get("surrogate_used", False),
        "score": candidate.get("score", 0.0),
        "cif_file": candidate.get("cif_file")
    }
    
    return stability_analysis

def process_candidate_file(file_path: str, cif_dir: str, max_samples: int = None, quiet: bool = False, cif_file_pattern: str = None) -> Tuple[int, List[Dict]]:
    """Process a single candidate file and return (total, stability_analyses)."""
    total_candidates = 0
    stability_analyses = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Stop after max_samples for testing (if specified)
                if max_samples is not None and total_candidates >= max_samples:
                    break
                
                try:
                    candidate = json.loads(line)
                    total_candidates += 1
                    
                    # If CIF file pattern is provided and candidate doesn't have cif_file, construct it from iteration
                    if cif_file_pattern and not candidate.get("cif_file"):
                        iteration = candidate.get("iteration")
                        if iteration is not None:
                            candidate["cif_file"] = cif_file_pattern.format(iteration)
                    
                    stability_analysis = analyze_candidate_stability(candidate, cif_dir, quiet)
                    stability_analyses.append(stability_analysis)
                        
                except json.JSONDecodeError as e:
                    if not quiet:
                        print(f"Warning: Could not parse JSON in {file_path}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return 0, []
    except (OSError, IOError) as e:
        print(f"Error processing {file_path}: {e}")
        return 0, []
    
    return total_candidates, stability_analyses

def main():
    """Main function to analyze stability of all candidate files."""
    parser = argparse.ArgumentParser(description='Analyze stability of valid candidates for MatterGen')
    parser.add_argument('--task', '-t', type=str, help='Specific task name to analyze (e.g., "Transparent Conductors")')
    parser.add_argument('--max-samples', '-n', type=int, help='Maximum number of samples to process per task')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output verbosity')
    parser.add_argument('--output-dir', type=str, help='Specific scoring_output directory to process (default: latest)')
    
    args = parser.parse_args()
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try validity_output first, then fall back to generated_cif
    example_cif_dir = os.path.join(script_dir, "example")
    generated_cif_dir = os.path.join(script_dir, "generated_cif")
    
    # Determine which CIF directory to use
    if os.path.exists(example_cif_dir):
        cif_dir = example_cif_dir
        cif_file_pattern = "gen_{}.cif"  # Format: gen_0.cif, gen_1.cif, etc.
        print(f"Using example directory for CIF files: {cif_dir}")
    elif os.path.exists(generated_cif_dir):
        cif_dir = generated_cif_dir
        cif_file_pattern = None  # Use filename directly
        print(f"Using generated_cif directory for CIF files: {cif_dir}")
    else:
        print(f"Error: Neither example nor generated_cif directory found")
        print(f"  Looked for: {example_cif_dir}")
        print(f"  Looked for: {generated_cif_dir}")
        return
    
    # Find validity_output directories first, then fall back to scoring_output
    if args.output_dir:
        output_dirs = [args.output_dir]
    else:
        validity_output_base = os.path.join(script_dir, "validity_output")
        scoring_output_base = script_dir
        
        # Look for validity_output/property_output_* directories
        validity_dirs = sorted(glob.glob(os.path.join(validity_output_base, "property_output_*")), reverse=True)
        # Look for scoring_output_* directories
        scoring_dirs = sorted(glob.glob(os.path.join(scoring_output_base, "scoring_output_*")), reverse=True)
        
        output_dirs = validity_dirs + scoring_dirs
    
    if not output_dirs:
        print(f"Error: No validity_output or scoring_output directories found")
        return
    
    # Use the latest directory if multiple exist
    output_dir = output_dirs[0]
    print(f"Processing output directory: {output_dir}")
    
    # Create stability output directory
    stability_output_dir = os.path.join(script_dir, "stability_output")
    os.makedirs(stability_output_dir, exist_ok=True)
    
    # Find all candidate files (JSONL in validity_output, .log in scoring_output)
    candidate_files = glob.glob(os.path.join(output_dir, "results_*.jsonl"))
    if not candidate_files:
        candidate_files = glob.glob(os.path.join(output_dir, "candidates_*.log"))
    
    # Filter by task if specified
    if args.task:
        # Handle different task name formats
        task_patterns = [
            args.task,
            args.task.replace(" ", "_"),
            args.task.replace(" ", "-"),
            args.task.replace(" ", ""),
            args.task.replace("/", "_")
        ]
        
        filtered_files = []
        for file_path in candidate_files:
            filename = os.path.basename(file_path)
            if any(pattern.replace(" ", "_").replace("/", "_") in filename for pattern in task_patterns):
                filtered_files.append(file_path)
        
        candidate_files = filtered_files
        
        if not candidate_files:
            print(f"No candidate files found for task: {args.task}")
            print("Available task files:", [os.path.basename(f) for f in glob.glob(os.path.join(scoring_output_dir, "candidates_*.log"))])
            return
    
    if not candidate_files:
        print("No candidate files found!")
        return
    
    # Process files quietly (only show summary at the end)
    
    # Process each file
    all_stability_analyses = []
    task_summary = defaultdict(lambda: {
        "total": 0, 
        "valid": 0,
        "invalid": 0,
        "stable": 0, 
        "marginally_stable": 0, 
        "unstable": 0, 
        "unknown": 0,
        "has_formation_energy": 0,
        "has_energy_above_hull": 0,
        "calculated_energy_above_hull": 0,
        "materials_api_used": 0,
        "surrogate_used": 0
    })
    
    # Counters for energy above hull calculations
    total_ehull_calculations = 0
    successful_ehull_calculations = 0
    
    for file_idx, file_path in enumerate(sorted(candidate_files)):
        # Extract task name from file path
        filename = os.path.basename(file_path)
        # Handle both results_*.jsonl and candidates_*.log formats
        if filename.startswith("results_"):
            task_name = filename.replace("results_", "").replace(".jsonl", "").replace("_", " ")
        else:
            task_name = filename.replace("candidates_", "").replace(".log", "").replace("_", " ")
        
        print(f"Processing file {file_idx+1}/{len(candidate_files)}: {filename}")
        
        # Handle special cases where underscores should not be replaced with spaces
        special_cases = {
            "SAW BAW Acoustic Substrates": "SAW/BAW Acoustic Substrates",
            "High k Dielectrics": "High-k Dielectrics",
            "Piezo Energy Harvesters": "Piezo Energy Harvesters",
            "Acousto optic Hybrids": "Acousto-optic Hybrids",
            "Electrically Insulating Dielectrics": "Electrically Insulating Dielectrics",
            "Hard, Stiff Ceramics": "Hard, Stiff Ceramics",
            "Hard Coating Materials": "Hard Coating Materials",
            "Structural Materials for Aerospace": "Structural Materials for Aerospace",
            "Photovoltaic Absorbers": "Photovoltaic Absorbers",
            "Solid State Electrolytes": "Solid-State Electrolytes",
            "Stable Wide Bandgap Semiconductors": "Stable Wide-Bandgap Semiconductors",
            "Toxic Free Perovskite Oxide": "Toxic_Free_Perovskite_Oxide",
            "Toxic_Free_Perovskite_Oxide": "Toxic_Free_Perovskite_Oxide",
            "Transparent Conductors": "Transparent Conductors",
            "Low Density Structural Aerospace": "Low_Density_Structural_Aerospace",
            "Low_Density_Structural_Aerospace": "Low_Density_Structural_Aerospace"
        }
        
        if task_name in special_cases:
            task_name = special_cases[task_name]
        
        # Process the file (quiet mode by default - only show summary)
        max_samples = args.max_samples if args.max_samples else None
        total, stability_analyses = process_candidate_file(file_path, cif_dir, max_samples=max_samples, quiet=True, cif_file_pattern=cif_file_pattern)
        
        if total > 0:
            all_stability_analyses.extend(stability_analyses)
            
            # Update task summary
            task_summary[task_name]["total"] += total
            for analysis in stability_analyses:
                stability_level = analysis["stability_level"]
                
                # Only count stability levels for valid candidates
                if analysis["is_valid"]:
                    task_summary[task_name][stability_level] += 1
                    task_summary[task_name]["valid"] += 1
                else:
                    task_summary[task_name]["invalid"] += 1
                    
                if analysis["has_formation_energy"]:
                    task_summary[task_name]["has_formation_energy"] += 1
                if analysis["has_energy_above_hull"]:
                    task_summary[task_name]["has_energy_above_hull"] += 1
                if analysis["calculated_energy_above_hull"] is not None:
                    task_summary[task_name]["calculated_energy_above_hull"] += 1
                    successful_ehull_calculations += 1
                if analysis["materials_api_used"]:
                    task_summary[task_name]["materials_api_used"] += 1
                if analysis["surrogate_used"]:
                    task_summary[task_name]["surrogate_used"] += 1
                
                # Count total energy above hull calculations attempted
                if analysis["is_valid"] and analysis.get("cif_file"):
                    total_ehull_calculations += 1
    
    # Calculate overall summary statistics
    total_candidates = len(all_stability_analyses)
    if total_candidates == 0:
        print("No candidates found to analyze!")
        return
    
    valid = sum(1 for a in all_stability_analyses if a["is_valid"])
    invalid = sum(1 for a in all_stability_analyses if not a["is_valid"])
    stable = sum(1 for a in all_stability_analyses if a["stability_level"] == "stable")
    marginally_stable = sum(1 for a in all_stability_analyses if a["stability_level"] == "marginally_stable")
    unstable = sum(1 for a in all_stability_analyses if a["stability_level"] == "unstable")
    unknown = sum(1 for a in all_stability_analyses if a["stability_level"] == "unknown")
    
    has_formation_energy = sum(1 for a in all_stability_analyses if a["has_formation_energy"])
    has_energy_above_hull = sum(1 for a in all_stability_analyses if a["has_energy_above_hull"])
    calculated_energy_above_hull = sum(1 for a in all_stability_analyses if a["calculated_energy_above_hull"] is not None)
    materials_api_used = sum(1 for a in all_stability_analyses if a["materials_api_used"])
    surrogate_used = sum(1 for a in all_stability_analyses if a["surrogate_used"])
    
    # Create summary statistics dictionary
    summary_stats = {
        "total_candidates": total_candidates,
        "valid": valid,
        "invalid": invalid,
        "stable": stable,
        "marginally_stable": marginally_stable,
        "unstable": unstable,
        "unknown": unknown,
        "has_formation_energy": has_formation_energy,
        "has_energy_above_hull": has_energy_above_hull,
        "calculated_energy_above_hull": calculated_energy_above_hull,
        "materials_api_used": materials_api_used,
        "surrogate_used": surrogate_used,
        "total_ehull_calculations": total_ehull_calculations,
        "successful_ehull_calculations": successful_ehull_calculations,
        "task_breakdown": {}
    }
    
    # Add task-specific statistics
    for task_name, stats in task_summary.items():
        if stats["total"] > 0:
            summary_stats["task_breakdown"][task_name] = stats
    
    # Save summary statistics to JSON file
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_output_file = os.path.join(stability_output_dir, f'stability_summary_{timestamp}.json')
    
    with open(summary_output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Print only the summary statistics
    print("=" * 80)
    print("📊 STABILITY ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total candidates analyzed: {total_candidates}")
    print(f"Valid: {valid} ({valid/total_candidates*100:.1f}%)")
    print(f"Invalid: {invalid} ({invalid/total_candidates*100:.1f}%)")
    print(f"Stable: {stable} ({stable/total_candidates*100:.1f}%)")
    print(f"Marginally stable: {marginally_stable} ({marginally_stable/total_candidates*100:.1f}%)")
    print(f"Unstable: {unstable} ({unstable/total_candidates*100:.1f}%)")
    print(f"Unknown: {unknown} ({unknown/total_candidates*100:.1f}%)")
    print()
    print(f"Has formation energy: {has_formation_energy}/{total_candidates} ({has_formation_energy/total_candidates*100:.1f}%)")
    print(f"Has energy above hull: {has_energy_above_hull}/{total_candidates} ({has_energy_above_hull/total_candidates*100:.1f}%)")
    print(f"Calculated energy above hull: {calculated_energy_above_hull}/{total_candidates} ({calculated_energy_above_hull/total_candidates*100:.1f}%)")
    if total_ehull_calculations > 0:
        print(f"Energy above hull calculation success rate: {successful_ehull_calculations}/{total_ehull_calculations} ({successful_ehull_calculations/total_ehull_calculations*100:.1f}%)")
    print(f"Materials API used: {materials_api_used}/{total_candidates} ({materials_api_used/total_candidates*100:.1f}%)")
    print(f"Surrogate used: {surrogate_used}/{total_candidates} ({surrogate_used/total_candidates*100:.1f}%)")
    print()
    
    # Print task breakdown
    print("📋 TASK BREAKDOWN")
    print("=" * 80)
    for task_name, stats in sorted(summary_stats["task_breakdown"].items()):
        total = stats["total"]
        print(f"\n{task_name}:")
        print(f"  Total candidates: {total}")
        print(f"  Valid: {stats['valid']} ({stats['valid']/total*100:.1f}%)")
        print(f"  Invalid: {stats['invalid']} ({stats['invalid']/total*100:.1f}%)")
        print(f"  Stable: {stats['stable']} ({stats['stable']/total*100:.1f}%)")
        print(f"  Marginally stable: {stats['marginally_stable']} ({stats['marginally_stable']/total*100:.1f}%)")
        print(f"  Unstable: {stats['unstable']} ({stats['unstable']/total*100:.1f}%)")
        print(f"  Unknown: {stats['unknown']} ({stats['unknown']/total*100:.1f}%)")
    
    print(f"\n💾 Summary saved to: {summary_output_file}")

if __name__ == "__main__":
    main()

