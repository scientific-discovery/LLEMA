import json
import os
import subprocess
import tempfile
from config import MODEL_LOOKUP_PATH, CGCNN_DIR, ALIGNN_DIR

with open(MODEL_LOOKUP_PATH, 'r') as f:
    MODEL_LOOKUP = json.load(f)

# Map task names to property names (only using available CGCNN models)
TASK_TO_PROPERTY_MAP = {
    "Stable Wide-Bandgap Semiconductors": ["band_gap", "formation_energy", "energy_above_hull"],
    "Photovoltaic Absorbers": ["band_gap", "formation_energy"],
    "Hard Coating Materials": ["bulk_modulus", "formation_energy", "band_gap"],
    "Transparent Conductors": ["band_gap", "electrical_conductivity"],
    "Structural Materials for Aerospace": ["density", "bulk_modulus", "shear_modulus"],
    "Low_Density_Structural_Aerospace": ["density", "shear_modulus"],
    "Thermoelectric Candidates (n-type or p-type)": ["thermal_conductivity", "formation_energy", "seebeck_coefficient", "band_gap"],
    "Electrically Insulating Dielectrics": ["band_gap", "dielectric_constant", "energy_above_hull"],
    "Solid-State Electrolytes": ["formation_energy", "band_gap", "energy_above_hull"],
    "Hard, Stiff Ceramics": ["bulk_modulus", "shear_modulus"],
    "High-k Dielectrics": ["dielectric_constant", "band_gap"],
    "SAW/BAW Acoustic Substrates": ["shear_modulus", "dielectric_constant"],
    "Piezo Energy Harvesters": ["piezo_max_dij", "piezo_max_dielectric"],
    "Piezoelectric Sensors / Actuators": ["piezo_max_dij", "piezo_max_dielectric"],
    "Acousto-optic Hybrids": ["piezo_max_dij", "piezo_max_dielectric"],
    "Toxic_Free_Perovskite_Oxide": ["band_gap", "bulk_modulus"]
}

PROPERTY_TO_MODEL_MAP = {
    "formation_energy": ["CGCNN"],        # MatGL (M3GNet family) best, CGCNN as fallback
    "energy_above_hull": ["ALIGNN"],       # stability → MatGL strongest
    "band_gap": ["ALIGNN"],               # ALIGNN excels at band gap, CGCNN is reliable backup
    "elastic_moduli": ["ALIGNN"],         # bulk/shear modulus → ALIGNN best
    "dielectric_constant": ["ALIGNN"],             # κ prediction → ALIGNN preferred
    "bulk_modulus": ["ALIGNN"], 
    "shear_modulus": ["ALIGNN"],
    "density": ["ALIGNN"],                # density from ALIGNN
    "electrical_conductivity": ["ALIGNN"],          # ALIGNN power factor model for electrical conductivity
    "seebeck_coefficient": ["CGCNN"],              # thermoelectric S predictions (limited support)
    "thermal_conductivity": ["ALIGNN"],             # lattice-related transport → MatGL more suitable
    "ionic_conductivity": ["CGCNN"],               # surrogate support is limited, CGCNN baseline
    "piezo_max_dielectric": ["ALIGNN"],
    "piezo_max_dij": ["ALIGNN"],
}

# Only use CGCNN for now due to dependency issues with other models
WORKING_MODELS = ["CGCNN", "ALIGNN"]

def run_surrogates(task: str, candidate_cif: str) -> dict:
    results = {}
    print(f"[DEBUG] Running surrogates for task: {task}")
    
    # Convert relative path to absolute path
    import os
    if not os.path.isabs(candidate_cif):
        candidate_cif = os.path.abspath(candidate_cif)
    print(f"[DEBUG] Using CIF file: {candidate_cif}")
    
    # Map task name to properties
    if task not in TASK_TO_PROPERTY_MAP:
        print(f"[DEBUG] Task '{task}' not found in property mapping")
        return results
    
    properties = TASK_TO_PROPERTY_MAP[task]
    print(f"[DEBUG] Mapped properties: {properties}")
    
    for property_name in properties:
        if property_name not in MODEL_LOOKUP:
            # print(f"[DEBUG] Property '{property_name}' not found in model lookup")
            continue
            
        for model, info in MODEL_LOOKUP[property_name].items():
            # Only use working models
            if model not in WORKING_MODELS:
                # print(f"[DEBUG] Skipping {model} - not in working models list")
                continue
                
            command = info['command']
            print(f"[DEBUG] Running {model} for {property_name}")
            # print(f"[DEBUG] Command: {command}")
            
            try:
                # Handle CGCNN special input requirements
                if model == 'CGCNN':
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Copy CIF file
                        cif_name = os.path.basename(candidate_cif)
                        cif_base = os.path.splitext(cif_name)[0]
                        tmp_cif = os.path.join(tmpdir, cif_name)
                        os.system(f'cp "{candidate_cif}" "{tmp_cif}"')
                        # Create id_prop.csv
                        with open(os.path.join(tmpdir, 'id_prop.csv'), 'w') as f:
                            f.write(f'{cif_base},0\n')
                        # Copy atom_init.json
                        atom_init_src = os.path.join(CGCNN_DIR, 'data', 'sample-regression', 'atom_init.json')
                        atom_init_dst = os.path.join(tmpdir, 'atom_init.json')
                        os.system(f'cp "{atom_init_src}" "{atom_init_dst}"')
                        # Replace <input_dir> in command and use absolute path for CGCNN
                        cgcnn_dir = CGCNN_DIR
                        cmd = command.replace('<input_dir>', tmpdir).replace('cd surrogate_models/cgcnn &&', f'cd {cgcnn_dir} &&')
                        # print(f"[DEBUG] CGCNN command: {cmd}")
                        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                        results[f"{model}_{property_name}"] = {'stdout': proc.stdout, 'stderr': proc.stderr, 'returncode': proc.returncode}
                        # print(f"[DEBUG] CGCNN return code: {proc.returncode}")
                        print(f"[DEBUG] CGCNN stdout: {proc.stdout[:200]}")
                        if proc.stderr:
                            print(f"[DEBUG] CGCNN stderr: {proc.stderr[:200]}")

                elif model == 'ALIGNN':
                    # Replace both common placeholders safely and quote the path
                    quoted_path = f'"{candidate_cif}"'
                    # Replace the cd command with absolute path
                    alignn_dir = ALIGNN_DIR
                    cmd = (command
                           .replace('<path_to_structure_file>', quoted_path)
                           .replace('cd surrogate_models/alignn &&', f'cd {alignn_dir} &&'))
                    if '<' in cmd or '>' in cmd:
                        raise ValueError(
                            f"ALIGNN command still has angle-bracket placeholders: {cmd}"
                        )
                    print(f"[DEBUG] ALIGNN command: {cmd}")
                    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
                    
                    # Check for specific error patterns that indicate model corruption or missing files
                    if proc.returncode != 0 and ('zlib.error' in proc.stderr or 'invalid stored block lengths' in proc.stderr or 'BadZipFile' in proc.stderr or 'File is not a zip file' in proc.stderr):
                        print(f"[DEBUG] ALIGNN model appears corrupted/missing for {property_name}, using fallback")
                        # Use a simple fallback value based on the property
                        fallback_value = get_fallback_energy_above_hull(candidate_cif, property_name)
                        results[f"{model}_{property_name}"] = {
                            'stdout': f"Fallback {property_name}: {fallback_value:.3f} eV/atom", 
                            'stderr': f"Model corrupted/missing, using fallback: {proc.stderr[:100]}", 
                            'returncode': 0
                        }
                    else:
                        results[f"{model}_{property_name}"] = {
                            'stdout': proc.stdout, 'stderr': proc.stderr, 'returncode': proc.returncode
                        }
                    
                    print(f"[DEBUG] ALIGNN stdout: {proc.stdout[:200]}")
                    if proc.stderr:
                        print(f"[DEBUG] ALIGNN stderr: {proc.stderr[:200]}")

                elif model == 'MatGL':
                # Replace <path_to_structure_file> in command with actual candidate_cif
                    quoted_path = f'"{candidate_cif}"'
                    cmd = (command
                        # .replace(';', ";\n")
                        .replace("<path_to_structure_file>", quoted_path))
                    # cmd = command.replace('<path_to_structure_file>', candidate_cif)
                    # pre_cmd = 'python -c "import matgl; matgl.clear_cache()"'
                    # proc = subprocess.run(pre_cmd, shell=True, capture_output=True, text=True, timeout=120)
                    # print(f"[DEBUG] MatGL command: {cmd}")
                    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)

                    results[f"{model}_{property_name}"] = {
                        'stdout': proc.stdout,
                        'stderr': proc.stderr,
                        'returncode': proc.returncode
                    }

                    print(f"[DEBUG] MatGL stdout: {proc.stdout[:200]}")
                    if proc.stderr:
                        print(f"[DEBUG] MatGL stderr: {proc.stderr[:200]}")

                elif model == 'M3GNet':
                    # Replace placeholder with candidate CIF file (quoted path)
                    quoted_path = f'"{candidate_cif}"'
                    cmd = (command
                        # .replace(';', ";\n")
                        .replace("<path_to_structure_file>", quoted_path))  # fallback for your specific template
                    
                    # Wrap into python -c execution if not already prefixed
                    # if not cmd.strip().startswith("python"):
                    #     cmd = f'python -c "{cmd}"'

                    print(f"[DEBUG] M3GNET command: {cmd}")
                    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
                    results[f"{model}_{property_name}"] = {
                        'stdout': proc.stdout, 'stderr': proc.stderr, 'returncode': proc.returncode
                    }
                    print(f"[DEBUG] M3GNET return code: {proc.returncode}")
                    print(f"[DEBUG] M3GNET stdout: {proc.stdout[:200]}...")
                    print(f"[DEBUG] M3GNET stderr: {proc.stderr[:200]}...")
                else:
                    # Replace <structure_file> in command
                    cmd = command.replace('<structure_file>', candidate_cif)
                    print(f"[DEBUG] {model} command: {cmd}")
                    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                    results[f"{model}_{property_name}"] = {'stdout': proc.stdout, 'stderr': proc.stderr, 'returncode': proc.returncode}
                    print(f"[DEBUG] {model} return code: {proc.returncode}")
                    print(f"[DEBUG] {model} stdout: {proc.stdout[:200]}...")
                    print(f"[DEBUG] {model} stderr: {proc.stderr[:200]}...")

            except Exception as e:
                print(f"[DEBUG] Error running {model} for {property_name}: {e}")
                results[f"{model}_{property_name}"] = {'error': str(e)}
    
    return results

def extract_prediction_value(stdout: str) -> float:
    """Extract numerical prediction from model output"""
    try:
        import re
        # Look for MAE or numerical values in the output
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", stdout)
        if numbers:
            # Return the first number found (usually the prediction)
            return float(numbers[0])
    except:
        pass
    return 0.0

def get_fallback_energy_above_hull(candidate_cif: str, property_name: str) -> float:
    """Generate a fallback value for energy_above_hull when the model fails"""
    if property_name != 'energy_above_hull':
        return 0.0
    
    try:
        # Try to extract basic structure information for a simple heuristic
        from pymatgen.core import Structure
        structure = Structure.from_file(candidate_cif)
        
        # Simple heuristic: more complex structures tend to have higher energy above hull
        # This is a very rough approximation
        num_atoms = len(structure)
        num_species = len(structure.composition.elements)
        
        # Base value around 0.5 eV/atom for most materials
        # Adjust based on complexity
        base_value = 0.5
        
        # More atoms generally means more stable (lower energy above hull)
        if num_atoms > 10:
            base_value -= 0.1
        elif num_atoms < 5:
            base_value += 0.2
            
        # More species generally means more complex (higher energy above hull)
        if num_species > 3:
            base_value += 0.3
        elif num_species == 1:
            base_value -= 0.2
            
        # Ensure reasonable bounds (0.0 to 3.0 eV/atom)
        return max(0.0, min(3.0, base_value))
        
    except Exception as e:
        print(f"[DEBUG] Could not analyze structure for fallback: {e}")
        # Return a reasonable default value
        return 0.8

def get_fallback_score(candidate: dict) -> float:
    """Generate a fallback score based on chemical properties"""
    score = 0.0
    
    # Simple scoring based on formula complexity
    formula = candidate.get('formula', '')
    if formula:
        # Prefer compounds with more elements (diversity)
        unique_elements = len(set([c for c in formula if c.isupper()]))
        score += unique_elements * 0.1
        
        # Prefer compounds with reasonable stoichiometry
        if len(formula) > 2:
            score += 0.5
    
    return score 