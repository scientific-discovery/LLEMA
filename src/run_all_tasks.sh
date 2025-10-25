#!/bin/bash

# Bash script to run all LLEMA tasks using run_task.py
# Each task runs individually with run_task.py

set -e  # Exit on any error

## Activate conda environment 'mat_sci'
if command -v conda >/dev/null 2>&1; then
    eval "$(/home/reddy/miniconda3/bin/conda shell.bash hook)"
    conda activate mat_sci
else
    echo "conda not found in PATH. Please install or load conda and retry." >&2
    exit 1
fi

# Ensure Materials Project API key is available for all subtasks
export MATERIALS_PROJECT_API_KEY="${MATERIALS_PROJECT_API_KEY:-YOUR_MATERIALS_PROJECT_API_KEY_HERE}"

echo "Starting execution of all LLEMA benchmark tasks"
echo "=========================================="

# Optional: set model if not provided by environment

# Run each task individually
echo "Running thermoelectric_candidates..."
python3 run_task.py thermoelectric_candidates

echo "Running electrically_insulating_dielectrics..."
python3 run_task.py electrically_insulating_dielectrics

echo "Running transparent_conductors..."
python3 run_task.py transparent_conductors

echo "Running low_density_structural_aerospace..."
python3 run_task.py low_density_structural_aerospace

echo "Running toxic_free_perovskite_oxide..."
python3 run_task.py toxic_free_perovskite_oxide

echo "Running structural_materials_for_aerospace..."
python3 run_task.py structural_materials_for_aerospace

echo "Running high_k_dielectrics..."
python3 run_task.py high_k_dielectrics

echo "Running piezo_energy_harvesters..."
python3 run_task.py piezo_energy_harvesters

echo "Running hard_coating_materials..."
python3 run_task.py hard_coating_materials

echo "Running acousto_optic_hybrids..."
python3 run_task.py acousto_optic_hybrids

echo "Running hard_stiff_ceramics..."
python3 run_task.py hard_stiff_ceramics

echo "Running saw_baw_acoustic_substrates..."
python3 run_task.py saw_baw_acoustic_substrates

echo "Running solid_state_electrolytes..."
python3 run_task.py solid_state_electrolytes

echo "Running stable_wide_bandgap_semiconductors..."
python3 run_task.py stable_wide_bandgap_semiconductors

echo "Running photovoltaic_absorbers..."
python3 run_task.py photovoltaic_absorbers
