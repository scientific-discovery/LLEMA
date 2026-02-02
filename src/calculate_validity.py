#!/usr/bin/env python3
"""
CIF File Property Calculation Script for MatterGen
Processes CIF files and calculates material properties.
"""

import os
import json
import sys
import re
import argparse
from datetime import datetime
from typing import Dict, Optional, List
import glob

# Add the material directory to the path to import modules
# Since this file is in the material folder, add the directory containing this file to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import property calculation modules
try:
    from agent.property_extractor import MaterialsProjectPropertyAPI
except ImportError as e:
    print(f"Warning: Could not import agent modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


# ====================== PROPERTY CALCULATION ======================

# Map task names to property names
TASK_TO_PROPERTY_MAP = {
    "Hard, Stiff Ceramics": ["bulk_modulus", "shear_modulus"],
    "Photovoltaic Absorbers": ["band_gap", "formation_energy"],
    "Solid-State Electrolytes": ["formation_energy", "band_gap"],
    "Stable Wide-Bandgap Semiconductors": ["band_gap", "formation_energy"],
    "SAW/BAW Acoustic Substrates": ["shear_modulus", "dielectric_constant"],
    "Structural Materials for Aerospace": ["bulk_modulus", "shear_modulus", "formation_energy"],
    "High-k Dielectrics": ["dielectric_constant", "band_gap"],
    "Piezo Energy Harvesters": ["piezo_max_dij", "piezo_max_dielectric"],
    "Hard Coating Materials": ["bulk_modulus", "shear_modulus"],
    "Acousto-optic Hybrids": ["piezo_max_dij", "piezo_max_dielectric"],
    "Electrically Insulating Dielectrics": ["band_gap", "dielectric_constant"],
    "Transparent Conductors": ["band_gap", "electrical_conductivity"],
    "Low_Density_Structural_Aerospace": ["density", "shear_modulus"],
    "Toxic_Free_Perovskite_Oxide": ["band_gap", "bulk_modulus"]
}

METRIC_TO_PROPERTY_MAP = {
    "band_gap": ("Band Gap", "eV"),
    "formation_energy": ("Formation Energy", "eV/atom"),
    "energy_above_hull": ("Energy Above Hull", "eV/atom"),
    "density": ("Density", "g/cm³"),
    "dielectric_constant": ("Dielectric Constant", ""),
    "bulk_modulus": ("Bulk modulus", "GPa"),
    "shear_modulus": ("Shear modulus", "GPa"),
    "piezo_max_dij": ("Piezoelectric d coefficient", "pC/N"),
    "piezo_max_dielectric": ("Piezoelectric dielectric constant", ""),
    "seebeck_coefficient": ("Seebeck coefficient", "μV·K⁻¹"),
    "thermal_conductivity": ("Thermal conductivity", "W·m⁻¹·K⁻¹"),
    "electrical_conductivity": ("Electrical conductivity", "S/m"),
}

# Constraints & Objectives (from main.py)
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
            ("band_gap", ">=", 2.0),
        ],
        "categorical": {"requires_any_element": [["Li"], ["Na"], ["K"], ["Mg"], ["Ca"], ["Al"]]},
    },
    "Stable Wide-Bandgap Semiconductors": {
        "numeric": [
            ("band_gap", ">=", 2.5),
            ("formation_energy", "<=", -1.0),
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

# Global target overrides for value-equals-target style objectives
TARGET_OVERRIDES = {
    "energy_above_hull": (0.0, 2.0),
}

TASK_OBJECTIVES = {
    "Hard, Stiff Ceramics": {"higher": ["bulk_modulus", "shear_modulus"]},
    "Photovoltaic Absorbers": {"lower": ["formation_energy"], "higher": ["band_gap"]},
    "Solid-State Electrolytes": {"lower": ["formation_energy"], "higher": ["band_gap"]},
    "Stable Wide-Bandgap Semiconductors": {"lower": ["formation_energy"], "higher": ["band_gap"]},
    "SAW/BAW Acoustic Substrates": {"higher": ["shear_modulus", "dielectric_constant"]},
    "Structural Materials for Aerospace": {"lower": ["formation_energy"], "higher": ["bulk_modulus", "shear_modulus"]},
    "High-k Dielectrics": {"higher": ["dielectric_constant", "band_gap"]},
    "Piezo Energy Harvesters": {"higher": ["piezo_max_dij", "piezo_max_dielectric"]},
    "Hard Coating Materials": {"higher": ["bulk_modulus", "shear_modulus"]},
    "Acousto-optic Hybrids": {"higher": ["piezo_max_dij", "piezo_max_dielectric"]},
    "Electrically Insulating Dielectrics": {"higher": ["band_gap", "dielectric_constant"]},
    "Transparent Conductors": {"higher": ["band_gap", "electrical_conductivity"]},
    "Low_Density_Structural_Aerospace": {"lower": ["density"], "higher": ["shear_modulus"]},
    "Toxic_Free_Perovskite_Oxide": {"higher": ["band_gap", "bulk_modulus"]},
}

# Property name handling
PROPERTY_SET = {
    "formation_energy","energy_above_hull","band_gap","thermal_conductivity",
    "seebeck_coefficient","dielectric_constant","bulk_modulus",
    "shear_modulus","electrical_conductivity","density","piezo_max_dij","piezo_max_dielectric"
}

ALIASES = {
    "formation_energy":"formation_energy","energy_above_hull":"energy_above_hull","band_gap":"band_gap",
    "thermal_conductivity":"thermal_conductivity","seebeck_coefficient":"seebeck_coefficient",
    "dielectric_constant":"dielectric_constant",
    "bulk_modulus":"bulk_modulus","shear_modulus":"shear_modulus",
    "electrical_conductivity":"electrical_conductivity","density":"density",
    "piezo_max_dij":"piezo_max_dij","piezo_max_dielectric":"piezo_max_dielectric",
    "temp_piezo_max_dij":"piezo_max_dij","temp_piezo_max_dielectric":"piezo_max_dielectric",
    "formation energy":"formation_energy","e_form":"formation_energy","ef":"formation_energy",
    "energy above hull":"energy_above_hull","e_above_hull":"energy_above_hull","e above hull":"energy_above_hull",
    "band gap":"band_gap","bandgap":"band_gap","e_g":"band_gap",
    "kappa":"thermal_conductivity",
    "seebeck":"seebeck_coefficient","s":"seebeck_coefficient",
    "dielectric":"dielectric_constant","κ":"dielectric_constant","k":"dielectric_constant",
    "sigma":"electrical_conductivity",
    "k_bulk":"bulk_modulus","g_shear":"shear_modulus",
    "piezo_dij":"piezo_max_dij","piezo_d":"piezo_max_dij","dij":"piezo_max_dij",
    "piezo_dielectric":"piezo_max_dielectric","piezo_k":"piezo_max_dielectric",
}

def _extract_value_from_predictions(predictions: dict, property_name: str) -> Optional[float]:
    """Extract a property value from surrogate predictions dictionary"""
    if not predictions:
        return None
    
    # Try to find the value in predictions
    for model_name, res in predictions.items():
        if property_name in model_name.lower() or property_name.replace("_", " ") in model_name.lower():
            if isinstance(res, dict):
                # Try structured fields first
                for k in ("value", "pred", "prediction", "y", "score"):
                    if k in res and isinstance(res[k], (int, float)):
                        return float(res[k])
                # Try to parse from stdout
                stdout = res.get("stdout", "")
                if stdout:
                    m = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(stdout))
                    if m:
                        return float(m[0])
    return None

def _violations_for(task: str, vals: dict, candidate: dict | None):
    spec = TASK_CONSTRAINTS.get(task, {})
    vios = []

    # numeric
    for prop, op, th in spec.get("numeric", []):
        v = vals.get(prop)
        ok = False
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
            if any(e in sp for e in {"Hg","Cd","Pb","Tl","As","Be","Se"}):
                vios.append(("composition","non_toxic",None,sp))
        if cat.get("earth_abundant"):
            if any(e in sp for e in {"Au","Ag","Pt","Pd","Rh","Ir","Os","Ru","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Sc","Y","In","Ga","Ge","Te","Ta","Hf","Re"}):
                vios.append(("composition","earth_abundant",None,sp))
        exclude_elements = cat.get("exclude_elements", [])
        if exclude_elements:
            if any(e in sp for e in exclude_elements):
                vios.append(("composition","exclude_elements",exclude_elements,sp))

    return vios

def get_all_unique_properties():
    """Get all unique properties required across all tasks"""
    all_properties = set()
    for task_properties in TASK_TO_PROPERTY_MAP.values():
        all_properties.update(task_properties)
    return sorted(list(all_properties))

def count_constraints_for_task(task_name: str) -> int:
    """Count total constraints for a task"""
    total = len(TASK_CONSTRAINTS.get(task_name, {}).get("numeric", []))
    if TASK_CONSTRAINTS.get(task_name, {}).get("categorical"):
        cat_constraints = TASK_CONSTRAINTS[task_name]["categorical"]
        if cat_constraints.get("requires_any_element"):
            total += 1
        if cat_constraints.get("non_toxic"):
            total += 1
        if cat_constraints.get("earth_abundant"):
            total += 1
        if cat_constraints.get("exclude_elements"):
            total += 1
    return total

def calculate_single_property(property_name: str, cif_path: str) -> Optional[float]:
    """
    Calculate a single property directly using the appropriate model.
    This avoids the overhead of calculating multiple properties per task.
    Falls back to Materials Project API for seebeck_coefficient.
    """
    # Ensure CIF path is absolute to avoid path resolution issues
    cif_path = os.path.abspath(cif_path)
    try:
        # Special handling for seebeck_coefficient - use Materials Project API directly
        if property_name == "seebeck_coefficient":
            from agent.property_extractor import get_seebeck_n_type_coefficient
            return get_seebeck_n_type_coefficient(cif_path)
        
        # Special handling for thermal_conductivity - use Materials Project API directly
        if property_name == "thermal_conductivity":
            from agent.property_extractor import get_thermal_conductivity
            return get_thermal_conductivity(cif_path)
        
        
        # Import the surrogate runner to access model mappings
        from agent.surrogate_runner import PROPERTY_TO_MODEL_MAP, WORKING_MODELS
        
        # Get the appropriate model for this property
        if property_name not in PROPERTY_TO_MODEL_MAP:
            print(f"[Property Calculation] No model mapping for {property_name}")
            return None
        
        models = PROPERTY_TO_MODEL_MAP[property_name]
        working_models = [m for m in models if m in WORKING_MODELS]
        
        if not working_models:
            print(f"[Property Calculation] No working models for {property_name}")
            return None
        
        # Create a temporary single-property task for the surrogate runner
        temp_task = f"temp_{property_name}"
        
        # Temporarily modify the surrogate runner's task mapping
        import agent.surrogate_runner as sr
        original_mapping = sr.TASK_TO_PROPERTY_MAP.copy()
        sr.TASK_TO_PROPERTY_MAP[temp_task] = [property_name]
        
        try:
            # Run surrogates for just this property
            surrogate_predictions = sr.run_surrogates(temp_task, cif_path)
            return _extract_value_from_predictions(surrogate_predictions, property_name)
            
        finally:
            # Restore original mapping
            sr.TASK_TO_PROPERTY_MAP.clear()
            sr.TASK_TO_PROPERTY_MAP.update(original_mapping)
            
    except Exception as e:
        print(f"[Property Calculation] Error calculating {property_name}: {e}")
        return None

def calculate_all_properties(cif_path: str, cif_data: Dict, tasks: List[str] = None) -> Dict:
    """
    Calculate properties for a candidate.
    If tasks is provided, only calculates properties needed for those tasks.
    Returns a dictionary with property values and metadata.
    """
    # Ensure CIF path is absolute to avoid path resolution issues
    cif_path = os.path.abspath(cif_path)
    print(f"[Property Calculation] Calculating properties for {os.path.basename(cif_path)}")
    
    # Initialize Materials Project API
    api = MaterialsProjectPropertyAPI()
    
    # Get properties needed for specified tasks, or all properties if no tasks specified
    if tasks:
        all_unique_properties = set()
        for task in tasks:
            if task in TASK_TO_PROPERTY_MAP:
                all_unique_properties.update(TASK_TO_PROPERTY_MAP[task])
        all_unique_properties = sorted(list(all_unique_properties))
    else:
        all_unique_properties = get_all_unique_properties()
    print(f"[Property Calculation] Properties to calculate: {all_unique_properties}")
    
    # Initialize results
    property_values = {}
    materials_api_used = False
    surrogate_used = False
    
    # Try Materials Project API first for all properties
    print("[Property Calculation] Getting properties from Materials Project API...")
    all_properties = api.get_all_5_properties(cif_path, cif_data.get('formula'))
    
    if all_properties:
        materials_api_used = True
        for property_name in all_unique_properties:
            if property_name in all_properties and all_properties[property_name] is not None:
                property_values[property_name] = all_properties[property_name]
                print(f"[Property Calculation] Materials API: {property_name} = {all_properties[property_name]:.4f}")
            else:
                print(f"[Property Calculation] Materials API: {property_name} not available")
    
    # For properties not available from Materials Project API, use surrogates
    missing_properties = [prop for prop in all_unique_properties if prop not in property_values]
    
    if missing_properties:
        print(f"[Property Calculation] Using surrogates for missing properties: {missing_properties}")
        surrogate_used = True
        
        # Calculate each missing property only once using direct model calls
        for prop in missing_properties:
            if prop in property_values:
                continue  # Already calculated
                
            print(f"[Property Calculation] Calculating {prop} directly")
            
            # Calculate this property directly using the appropriate model
            prop_value = calculate_single_property(prop, cif_path)
            
            if prop_value is not None:
                property_values[prop] = prop_value
                print(f"[Property Calculation] Direct calculation: {prop} = {prop_value:.4f}")
            else:
                print(f"[Property Calculation] Direct calculation: {prop} not calculated")
    
    # Create final property dictionary with metadata
    result = {
        'property_values': property_values,
        'materials_api_used': materials_api_used,
        'surrogate_used': surrogate_used,
        'total_properties_calculated': len(property_values),
        'missing_properties': [prop for prop in all_unique_properties if prop not in property_values]
    }
    
    print(f"[Property Calculation] Completed: {len(property_values)}/{len(all_unique_properties)} properties calculated")
    if result['missing_properties']:
        print(f"[Property Calculation] Missing properties: {result['missing_properties']}")
    
    return result

# ====================== CIF PARSING ======================

def parse_cif_file(cif_path: str) -> Optional[Dict]:
    """Parse a CIF file and extract structural information"""
    # Ensure CIF path is absolute to avoid path resolution issues
    cif_path = os.path.abspath(cif_path)
    if not os.path.exists(cif_path):
        print(f"CIF file not found: {cif_path}")
        return None
    
    try:
        with open(cif_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cif_data = {}
        
        # Extract formula from _chemical_formula_structural (preferred)
        formula_match = re.search(r'_chemical_formula_structural\s+([^\s\n]+)', content)
        if formula_match:
            cif_data['formula'] = formula_match.group(1).strip()
        else:
            # Fallback to _chemical_formula_sum
            formula_match = re.search(r'_chemical_formula_sum\s+[\'"]?([^\'"]+)[\'"]?', content)
            if formula_match:
                cif_data['formula'] = formula_match.group(1).strip()
        
        # Extract iteration number from filename (mattergen pattern: gen_XXX.cif)
        filename = os.path.basename(cif_path)
        iteration_match = re.search(r'gen_(\d+)\.cif', filename)
        if iteration_match:
            cif_data['iteration'] = int(iteration_match.group(1))
        else:
            cif_data['iteration'] = 0
        
        # Extract lattice parameters
        lattice_params = {}
        for param in ['_cell_length_a', '_cell_length_b', '_cell_length_c', 
                     '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']:
            match = re.search(f'{param}\\s+([0-9.-]+)', content)
            if match:
                lattice_params[param] = float(match.group(1))
        
        if len(lattice_params) == 6:
            # Convert to lattice vectors (simplified - assumes orthogonal for now)
            a, b, c = lattice_params['_cell_length_a'], lattice_params['_cell_length_b'], lattice_params['_cell_length_c']
            # Note: alpha, beta, gamma are available but not used in this simplified version
            # alpha, beta, gamma = lattice_params['_cell_angle_alpha'], lattice_params['_cell_angle_beta'], lattice_params['_cell_angle_gamma']
            
            # Simple orthogonal lattice for now (can be enhanced for non-orthogonal)
            cif_data['lattice'] = [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
        
        # Extract atomic positions
        species = []
        coords = []
        
        # Find the atomic positions section and determine column order
        lines = content.split('\n')
        in_atom_section = False
        column_indices = {}  # Map field names to column indices
        
        for i, line in enumerate(lines):
            if '_atom_site_type_symbol' in line:
                in_atom_section = True
                # Parse column headers to determine positions
                for j in range(i, min(i + 10, len(lines))):
                    header_line = lines[j].strip()
                    if header_line.startswith('_atom_site_'):
                        field_name = header_line.split()[0]
                        column_indices[field_name] = len(column_indices)
                    elif header_line and not header_line.startswith('_'):
                        # Data row starts here
                        break
                continue
            elif in_atom_section and line.strip() and not line.startswith('_'):
                parts = line.split()
                if len(parts) >= 4:
                    # Determine indices based on what we found in headers
                    type_idx = column_indices.get('_atom_site_type_symbol', 0)
                    x_idx = column_indices.get('_atom_site_fract_x', 3)
                    y_idx = column_indices.get('_atom_site_fract_y', 4)
                    z_idx = column_indices.get('_atom_site_fract_z', 5)
                    
                    # Fallback: if headers weren't parsed, use standard CIF format
                    # Standard format: type_symbol, label, multiplicity, x, y, z, occupancy
                    if not column_indices:
                        type_idx, x_idx, y_idx, z_idx = 0, 3, 4, 5
                    
                    if (type_idx < len(parts) and x_idx < len(parts) and 
                        y_idx < len(parts) and z_idx < len(parts)):
                        species.append(parts[type_idx])
                        try:
                            coords.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
                        except (ValueError, IndexError):
                            continue
            elif in_atom_section and line.startswith('_'):
                break
        
        if species and coords:
            cif_data['species'] = species
            cif_data['coords'] = coords
        
        return cif_data
        
    except (ValueError, IndexError, KeyError) as e:
        print(f"Error parsing CIF file {cif_path}: {e}")
        return None

# ====================== CONSTRAINT CHECKING ======================

def check_constraints_for_task(cif_data: Dict, property_calculation_result: Dict, task_name: str) -> Dict:
    """
    Check constraints for a specific task.
    Returns a dictionary with iteration, compound_formula, property_values (including categorical constraints), 
    successful_constraints, failed_constraints, and materials_api_used.
    """
    property_values = property_calculation_result['property_values']
    
    # Get properties needed for this task
    task_properties = TASK_TO_PROPERTY_MAP.get(task_name, [])
    task_property_values = {prop: property_values.get(prop) for prop in task_properties}
    
    # Check violations
    vios = _violations_for(task_name, task_property_values, cif_data)
    total_constraints = count_constraints_for_task(task_name)
    
    failed = len(vios)
    successful = total_constraints - failed
    
    # Check categorical constraints and include results in property_values
    spec = TASK_CONSTRAINTS.get(task_name, {})
    if spec.get("categorical"):
        sp = set()
        if cif_data:
            if isinstance(cif_data.get("species"), list):
                sp = {str(x) for x in cif_data["species"]}
            elif isinstance(cif_data.get("formula"), str):
                sp = set(re.findall(r"[A-Z][a-z]?", cif_data["formula"]))
        
        cat = spec["categorical"]
        
        # Check earth_abundant
        if cat.get("earth_abundant"):
            non_abundant = {"Au","Ag","Pt","Pd","Rh","Ir","Os","Ru","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Sc","Y","In","Ga","Ge","Te","Ta","Hf","Re"}
            has_non_abundant = any(e in sp for e in non_abundant)
            task_property_values["earth_abundant"] = not has_non_abundant
        
        # Check non_toxic
        if cat.get("non_toxic"):
            toxic_elements = {"Hg","Cd","Pb","Tl","As","Be","Se"}
            has_toxic = any(e in sp for e in toxic_elements)
            task_property_values["non_toxic"] = not has_toxic
        
        # Check requires_any_element
        if cat.get("requires_any_element"):
            need_any = cat.get("requires_any_element", [])
            ok_any = any(any(el in sp for el in group) for group in need_any)
            task_property_values["requires_any_element"] = ok_any
        
        # Check exclude_elements
        if cat.get("exclude_elements"):
            exclude_elements = cat.get("exclude_elements", [])
            has_excluded = any(e in sp for e in exclude_elements)
            task_property_values["exclude_elements"] = not has_excluded
    
    # Create result dictionary
    result = {
        'iteration': cif_data.get('iteration', 0),
        'compound_formula': cif_data.get('formula', 'Unknown'),
        'property_values': task_property_values,  # Includes both numeric properties and categorical constraints
        'task_name': task_name,
        'successful_constraints': successful,
        'failed_constraints': failed,
        'materials_api_used': property_calculation_result['materials_api_used']
    }
    
    return result

def get_available_tasks() -> List[str]:
    """Get list of all available tasks"""
    return [
        "Hard, Stiff Ceramics",
        "Photovoltaic Absorbers",
        "Solid-State Electrolytes",
        "Stable Wide-Bandgap Semiconductors",
        "SAW/BAW Acoustic Substrates",
        "Structural Materials for Aerospace",
        "High-k Dielectrics",
        "Piezo Energy Harvesters",
        "Hard Coating Materials",
        "Acousto-optic Hybrids",
        "Electrically Insulating Dielectrics",
        "Transparent Conductors",
        "Low_Density_Structural_Aerospace",
        "Toxic_Free_Perovskite_Oxide"
    ]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Calculate properties and check constraints for CIF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CIF files for all tasks
  python scoring.py --tasks all
  
  # Process for a specific task
  python scoring.py --tasks "Photovoltaic Absorbers"
  
  # Process for multiple specific tasks
  python scoring.py --tasks "Photovoltaic Absorbers" "Transparent Conductors"
  
  # Process with custom CIF directory
  python scoring.py --tasks all --cif-dir /path/to/cif/files
        """
    )
    
    parser.add_argument(
        '--tasks',
        nargs='+',
        required=True,
        help='Task(s) to process. Use "all" to process all tasks, or specify task names.'
    )
    
    parser.add_argument(
        '--cif-dir',
        default="example",
        help='Directory containing CIF files to process (default: %(default)s)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for results (default: auto-generated with timestamp)'
    )
    
    return parser.parse_args()

def main():
    """
    Main function to process all CIF files, calculate properties, and check constraints for specified tasks.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Get available tasks
    all_available_tasks = get_available_tasks()
    
    # Determine which tasks to process
    if len(args.tasks) == 1 and args.tasks[0].lower() == 'all':
        tasks_to_process = all_available_tasks
    else:
        # Validate provided tasks
        invalid_tasks = [task for task in args.tasks if task not in all_available_tasks]
        if invalid_tasks:
            print(f"Error: Invalid task(s): {invalid_tasks}")
            print("Available tasks:")
            for i, task in enumerate(all_available_tasks, 1):
                print(f"  {i:2d}. {task}")
            sys.exit(1)
        tasks_to_process = args.tasks
    
    # Configuration
    cif_directory = os.path.abspath(args.cif_dir)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Default output directory relative to material folder
        output_dir = f"./validity_output/property_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[Main] Starting CIF file property calculation and constraint checking")
    print(f"[Main] CIF directory: {cif_directory}")
    print(f"[Main] Tasks to process: {', '.join(tasks_to_process)}")
    print(f"[Main] Output directory: {output_dir}")
    print("="*80)
    
    # Find all CIF files and convert to absolute paths
    cif_files = glob.glob(os.path.join(cif_directory, "*.cif"))
    # Convert all paths to absolute paths to avoid issues with surrogate runners
    cif_files = [os.path.abspath(cif_file) for cif_file in cif_files]
    print(f"[Main] Found {len(cif_files)} CIF files to process")
    
    if not cif_files:
        print("[Main] No CIF files found!")
        return
    
    # Process each task separately
    task_results = {task: [] for task in tasks_to_process}
    
    # Process each CIF file
    for i, cif_file in enumerate(cif_files):
        print(f"\n{'='*80}")
        print(f"[Main] PROCESSING CIF FILE {i+1}/{len(cif_files)}: {os.path.basename(cif_file)}")
        print(f"{'='*80}")
        
        # Parse CIF file
        cif_data = parse_cif_file(cif_file)
        if not cif_data:
            print(f"[Main] Failed to parse CIF file: {cif_file}")
            continue
        
        print(f"[Main] Parsed formula: {cif_data.get('formula', 'Unknown')} (iteration: {cif_data.get('iteration', 0)})")
        
        # Calculate all properties for this candidate (once per file, for all tasks)
        try:
            property_calculation_result = calculate_all_properties(cif_file, cif_data, tasks_to_process)
            
            if not property_calculation_result['property_values']:
                print(f"[Main] No properties calculated for {cif_file}, but will still check constraints...")
            
            # Check constraints for each task (even if no properties were calculated)
            for task_name in tasks_to_process:
                try:
                    result = check_constraints_for_task(cif_data, property_calculation_result, task_name)
                    task_results[task_name].append(result)
                    
                    # Write result to task-specific file
                    task_safe_name = task_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('/', '_')
                    task_output_file = os.path.join(output_dir, f'results_{task_safe_name}.jsonl')
                    
                    with open(task_output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result) + '\n')
                    
                    print(f"[Main] {task_name}: {result['successful_constraints']} successful, {result['failed_constraints']} failed")
                    
                except (ValueError, KeyError, OSError) as e:
                    print(f"[Main] Error processing {cif_file} for {task_name}: {e}")
                    continue
                
        except (ValueError, KeyError, OSError) as e:
            print(f"[Main] Error processing {cif_file}: {e}")
            continue
    
    # Summary
    print("\n" + "="*80)
    print("[Main] Processing complete!")
    print(f"[Main] Processed {len(cif_files)} CIF files")
    print(f"[Main] Results saved to task-specific files in: {output_dir}/")
    print("\n[Main] Summary by task:")
    
    for task_name in tasks_to_process:
        results = task_results[task_name]
        if results:
            total_successful = sum(r['successful_constraints'] for r in results)
            total_failed = sum(r['failed_constraints'] for r in results)
            task_safe_name = task_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('/', '_')
            task_output_file = os.path.join(output_dir, f'results_{task_safe_name}.jsonl')
            print(f"  {task_name}: {len(results)} candidates, {total_successful} successful, {total_failed} failed constraints")
            print(f"    Output: results_{task_safe_name}.jsonl")
        else:
            print(f"  {task_name}: No valid results")

if __name__ == "__main__":
    main()
