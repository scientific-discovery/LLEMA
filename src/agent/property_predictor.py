import os
import json
from datetime import datetime
from typing import Dict, Tuple

# Local imports from agent
from surrogate_runner import run_surrogates
from structure_builder import build_cif
from main import (
    TASK_TO_PROPERTY_MAP,
    METRIC_TO_PROPERTY_MAP,
    _values_from_predictions,
    get_cached_materials_properties,
)


def predict_properties_for_candidate(candidate: Dict, task_name: str, out_dir: str) -> Tuple[Dict, Dict, bool, str]:
    """
    Build CIF for candidate, query Materials Project API for all mapped properties,
    fall back to surrogate models for missing properties, and return:
      - predictions: dict of model_name -> {stdout, stderr, returncode}
      - property_values: canonical {property: float}
      - materials_api_used: bool
      - cif_path: path to the written CIF file
    Mirrors the logic used in main.py.
    """
    os.makedirs(out_dir, exist_ok=True)

    formula = candidate.get('formula', 'Unknown')
    safe_formula = ''.join(ch if ch.isalnum() else '_' for ch in formula) or 'candidate'
    cif_path = os.path.join(out_dir, f"{safe_formula}.cif")

    # 1) Build CIF
    build_cif(candidate, cif_path)

    # 2) Materials API first
    predictions: Dict[str, Dict] = {}
    materials_api_used = False
    properties = TASK_TO_PROPERTY_MAP.get(task_name, [])

    all_properties = get_cached_materials_properties(cif_path, formula)
    if all_properties:
        materials_api_used = True
        for property_name in properties:
            if property_name in all_properties and all_properties[property_name] is not None:
                label, unit = METRIC_TO_PROPERTY_MAP[property_name]
                val = all_properties[property_name]
                predictions[f"material_{property_name}"] = {
                    'stdout': f"{label}: {val:.2f} {unit}",
                    'stderr': "",
                    'returncode': "",
                }

    # 3) Surrogates for gaps
    if not materials_api_used or len(predictions) < len(properties):
        surrogate_predictions = run_surrogates(task_name, cif_path)
        predictions.update(surrogate_predictions)

    # 4) Canonical property values
    property_values = _values_from_predictions(predictions)

    return predictions, property_values, materials_api_used, cif_path


