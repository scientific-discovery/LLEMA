"""
CIF file validation and correction utilities for ensuring proper stoichiometry
and structure integrity before passing to ALIGNN.
"""

import re
from collections import Counter
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import os

def parse_formula(formula: str) -> dict:
    """
    Parse chemical formula into element counts.
    Example: "Li2MnPO4" -> {"Li": 2, "Mn": 1, "P": 1, "O": 4}
    """
    # Pattern to match element symbols and their counts
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    element_counts = {}
    for element, count in matches:
        count = int(count) if count else 1
        element_counts[element] = count
    
    return element_counts

def count_atoms_in_structure(species: list) -> dict:
    """
    Count atoms in the structure species list.
    """
    return Counter(species)

def validate_stoichiometry(candidate: dict) -> tuple[bool, str, dict]:
    """
    Validate that the formula matches the actual atom counts in the structure.
    
    Returns:
        (is_valid, error_message, corrected_candidate)
    """
    if not all(k in candidate for k in ['formula', 'species', 'coords']):
        return False, "Missing required fields: formula, species, coords", candidate
    
    formula = candidate['formula']
    species = candidate['species']
    coords = candidate['coords']
    
    # Check if species and coords have same length
    if len(species) != len(coords):
        return False, f"Species count ({len(species)}) doesn't match coords count ({len(coords)})", candidate
    
    # Parse formula
    try:
        expected_counts = parse_formula(formula)
    except Exception as e:
        return False, f"Failed to parse formula '{formula}': {e}", candidate
    
    # Count actual atoms
    actual_counts = count_atoms_in_structure(species)
    
    # Check if counts match
    if expected_counts != actual_counts:
        error_msg = f"Stoichiometry mismatch: formula '{formula}' expects {expected_counts}, but structure has {actual_counts}"
        return False, error_msg, candidate
    
    return True, "", candidate

def fix_stoichiometry(candidate: dict) -> dict:
    """
    Attempt to fix stoichiometry by adjusting the formula to match the structure.
    """
    if not all(k in candidate for k in ['formula', 'species', 'coords']):
        return candidate
    
    species = candidate['species']
    actual_counts = count_atoms_in_structure(species)
    
    # Generate corrected formula
    corrected_formula = ""
    for element, count in sorted(actual_counts.items()):
        if count == 1:
            corrected_formula += element
        else:
            corrected_formula += f"{element}{count}"
    
    # Create corrected candidate
    corrected_candidate = candidate.copy()
    corrected_candidate['formula'] = corrected_formula
    
    return corrected_candidate

def validate_and_fix_cif(candidate: dict) -> tuple[bool, str, dict]:
    """
    Validate and attempt to fix CIF structure issues.
    
    Returns:
        (is_valid, message, corrected_candidate)
    """
    # First validate stoichiometry
    is_valid, error_msg, candidate = validate_stoichiometry(candidate)
    
    if not is_valid:
        # Try to fix by correcting the formula
        corrected_candidate = fix_stoichiometry(candidate)
        is_valid_fixed, _, _ = validate_stoichiometry(corrected_candidate)
        
        if is_valid_fixed:
            return True, f"Fixed stoichiometry: {error_msg} -> corrected formula to {corrected_candidate['formula']}", corrected_candidate
        else:
            return False, f"Cannot fix stoichiometry: {error_msg}", candidate
    
    return True, "Structure is valid", candidate

def build_validated_cif(candidate: dict, output_path: str) -> tuple[bool, str]:
    """
    Build CIF file with validation and error handling.
    
    Returns:
        (success, message)
    """
    try:
        # Validate and fix the candidate
        is_valid, message, corrected_candidate = validate_and_fix_cif(candidate)
        
        if not is_valid:
            return False, f"Validation failed: {message}"
        
        # Build the structure
        structure = Structure(
            lattice=corrected_candidate['lattice'],
            species=corrected_candidate['species'],
            coords=corrected_candidate['coords']
        )
        
        # Write CIF file
        structure.to(filename=output_path)
        
        return True, f"Successfully built CIF: {message}"
        
    except Exception as e:
        return False, f"Failed to build CIF: {e}"

def test_cif_validation():
    """Test the validation functions with example data."""
    
    # Test case 1: Valid structure
    valid_candidate = {
        'formula': 'TiO2',
        'lattice': [[4.6, 0.0, 0.0], [0.0, 4.6, 0.0], [0.0, 0.0, 3.0]],
        'species': ['Ti', 'O', 'O'],
        'coords': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
    }
    
    is_valid, msg, corrected = validate_and_fix_cif(valid_candidate)
    print(f"Valid case: {is_valid}, {msg}")
    
    # Test case 2: Invalid stoichiometry (like the LLM generated)
    invalid_candidate = {
        'formula': 'Li2MnPO4',  # Expects 4 O atoms
        'lattice': [[6.65, 0.0, 0.0], [0.0, 6.65, 0.0], [0.0, 0.0, 4.93]],
        'species': ['Li', 'Li', 'Mn', 'P', 'O', 'O'],  # Only 2 O atoms
        'coords': [[0.28, 0.33, 0.25], [0.72, 0.67, 0.75], [0.43, 0.43, 0.0], 
                  [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.33, 0.67, 0.0]]
    }
    
    is_valid, msg, corrected = validate_and_fix_cif(invalid_candidate)
    print(f"Invalid case: {is_valid}, {msg}")
    print(f"Corrected formula: {corrected['formula']}")

if __name__ == "__main__":
    test_cif_validation()
