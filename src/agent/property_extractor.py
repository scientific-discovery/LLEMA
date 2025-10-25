#!/usr/bin/env python3
"""
Materials Project Property Functions
Individual functions for each of the 5 properties with CIF-based polymorph validation

Key fixes:
- Removed 'fields=' usage that caused HTTP 400 on /materials/summary.
- Use '_all_fields=true' again for summary/elasticity/dielectric endpoints.
- Kept robust extraction for dielectric (electronic preferred) and bulk modulus (VRH preferred),
  with many key-name fallbacks and tensor → scalar reduction when needed.
"""

import os
import requests
import re
from typing import Dict, List, Optional

# Optional: use NumPy if available to compute VRH from elastic tensor (rarely needed)
try:
    import numpy as np  # type: ignore
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False

# API Configuration
API_KEY = os.environ.get('MATERIALS_PROJECT_API_KEY', '')
BASE_URL = "https://api.materialsproject.org"  # keep as-is for your setup

class MaterialsProjectPropertyAPI:
    def __init__(self, api_key=API_KEY):
        self.api_key = api_key
        self.headers = {"X-API-KEY": self.api_key}
        self.base_url = BASE_URL
    
    # ---------------- CIF parsing ----------------

    def parse_cif_file(self, cif_path: str) -> Optional[Dict]:
        """Parse a CIF file and extract structural information"""
        if not os.path.exists(cif_path):
            print(f"CIF file not found: {cif_path}")
            return None
        
        try:
            with open(cif_path, 'r') as f:
                content = f.read()
            
            cif_data = {}
            
            # Extract formula
            formula_match = re.search(r'_chemical_formula_sum\s+[\'"]?([^\'"]+)[\'"]?', content)
            if formula_match:
                raw_formula = formula_match.group(1).strip()
                cif_data['formula'] = self._normalize_formula(raw_formula)
            
            # Extract lattice parameters
            a_match = re.search(r'_cell_length_a\s+([\d.]+)', content)
            b_match = re.search(r'_cell_length_b\s+([\d.]+)', content)
            c_match = re.search(r'_cell_length_c\s+([\d.]+)', content)
            
            if a_match and b_match and c_match:
                cif_data['lattice_a'] = float(a_match.group(1))
                cif_data['lattice_b'] = float(b_match.group(1))
                cif_data['lattice_c'] = float(c_match.group(1))
            
            # Extract angles
            alpha_match = re.search(r'_cell_angle_alpha\s+([\d.]+)', content)
            beta_match = re.search(r'_cell_angle_beta\s+([\d.]+)', content)
            gamma_match = re.search(r'_cell_angle_gamma\s+([\d.]+)', content)
            
            if alpha_match and beta_match and gamma_match:
                cif_data['alpha'] = float(alpha_match.group(1))
                cif_data['beta'] = float(beta_match.group(1))
                cif_data['gamma'] = float(gamma_match.group(1))
            
            # Extract volume
            volume_match = re.search(r'_cell_volume\s+([\d.]+)', content)
            if volume_match:
                cif_data['volume'] = float(volume_match.group(1))
            
            # Extract space group
            sg_match = re.search(r'_symmetry_space_group_name_H-M\s+[\'"]?([^\'"]+)[\'"]?', content)
            if sg_match:
                cif_data['space_group'] = sg_match.group(1).strip()
            
            # Extract Z
            z_match = re.search(r'_cell_formula_units_Z\s+(\d+)', content)
            if z_match:
                cif_data['z'] = int(z_match.group(1))
            
            return cif_data
            
        except Exception as e:
            print(f"Error parsing CIF file {cif_path}: {e}")
            return None
    
    def _normalize_formula(self, raw_formula: str) -> str:
        """Convert formula from 'Ba1 Ti1 O3' format to 'BaTiO3' format"""
        normalized = re.sub(r'(\w)1\b', r'\1', raw_formula)
        normalized = normalized.replace(' ', '')
        return normalized

    # ---------------- Summary (no 'fields' to avoid 400) ----------------

    def get_all_polymorphs(self, formula: str) -> List[Dict]:
        """Get all polymorphs for a given formula from Materials Project."""
        url = f"{self.base_url}/materials/summary"
        params = {
            "formula": formula,
            "_all_fields": "true"  # safe in your environment + gives 'structure'
        }
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=30)
            if r.status_code == 200:
                return (r.json() or {}).get("data", []) or []
            else:
                print(f"Summary query failed: HTTP {r.status_code} - {r.text[:200]}")
        except Exception as e:
            print(f"Error getting polymorphs for {formula}: {e}")
        return []

    # ---------------- CIF/structure matching ----------------

    def calculate_structural_similarity(self, cif_data: Dict, mp_polymorph: Dict) -> float:
        """Calculate structural similarity between CIF and MP polymorph"""
        if not cif_data or not mp_polymorph.get('structure'):
            return 0.0
        
        mp_structure = mp_polymorph['structure']
        mp_lattice = mp_structure.get('lattice', {})
        
        similarity_score = 0.0
        total_checks = 0
        
        # Compare lattice parameters (10% tolerance)
        tolerance = 0.1
        
        if 'lattice_a' in cif_data and mp_lattice.get('a'):
            total_checks += 1
            if abs(cif_data['lattice_a'] - mp_lattice['a']) / mp_lattice['a'] <= tolerance:
                similarity_score += 1
        
        if 'lattice_b' in cif_data and mp_lattice.get('b'):
            total_checks += 1
            if abs(cif_data['lattice_b'] - mp_lattice['b']) / mp_lattice['b'] <= tolerance:
                similarity_score += 1
        
        if 'lattice_c' in cif_data and mp_lattice.get('c'):
            total_checks += 1
            if abs(cif_data['lattice_c'] - mp_lattice['c']) / mp_lattice['c'] <= tolerance:
                similarity_score += 1
        
        # Compare angles (5° tolerance)
        angle_tolerance = 5.0
        
        if 'alpha' in cif_data and mp_lattice.get('alpha'):
            total_checks += 1
            if abs(cif_data['alpha'] - mp_lattice['alpha']) <= angle_tolerance:
                similarity_score += 1
        
        if 'beta' in cif_data and mp_lattice.get('beta'):
            total_checks += 1
            if abs(cif_data['beta'] - mp_lattice['beta']) <= angle_tolerance:
                similarity_score += 1
        
        if 'gamma' in cif_data and mp_lattice.get('gamma'):
            total_checks += 1
            if abs(cif_data['gamma'] - mp_lattice['gamma']) <= angle_tolerance:
                similarity_score += 1
        
        # Compare volume (10% tolerance)
        if 'volume' in cif_data and mp_lattice.get('volume'):
            total_checks += 1
            if abs(cif_data['volume'] - mp_lattice['volume']) / mp_lattice['volume'] <= tolerance:
                similarity_score += 1
        
        # Compare space group (exact match)
        if 'space_group' in cif_data and mp_polymorph.get('symmetry', {}).get('symbol'):
            total_checks += 1
            if cif_data['space_group'] == mp_polymorph['symmetry']['symbol']:
                similarity_score += 1
        
        return similarity_score / total_checks if total_checks > 0 else 0.0
    
    def find_best_polymorph_with_cif(self, cif_path: str, formula: str = None) -> Optional[Dict]:
        """Find the best polymorph using CIF-based structural matching"""
        cif_data = self.parse_cif_file(cif_path)
        if not cif_data:
            return None
        
        if not formula:
            formula = cif_data.get('formula')
            if not formula:
                return None
        
        polymorphs = self.get_all_polymorphs(formula)
        if not polymorphs:
            return None
        
        # keep only those with structure
        polys = [p for p in polymorphs if p.get('structure') and p['structure'].get('lattice')]
        if not polys:
            return None
        
        scored = []
        for p in polys:
            scored.append((p, self.calculate_structural_similarity(cif_data, p)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None

    # ---------------- Dielectric ----------------

    def get_dielectric_data(self, material_id: str) -> Dict:
        """
        Get dielectric data for a material.
        Use '_all_fields=true' to avoid param incompatibility in your environment.
        """
        url = f"{self.base_url}/materials/dielectric"
        params = {"material_ids": material_id, "_all_fields": "true"}
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=30)
            if r.status_code == 200:
                data = (r.json() or {}).get("data", [])
                if data:
                    return data[0] or {}
            else:
                print(f"Dielectric query failed: HTTP {r.status_code} - {r.text[:200]}")
        except Exception as e:
            print(f"Error getting dielectric data for {material_id}: {e}")
        return {}

    def _extract_scalar_dielectric(self, dielectric: Dict) -> Optional[float]:
        """
        Return a single scalar dielectric constant.
        Preference:
          1) e_electronic (high-frequency)
          2) e_total (electronic + ionic)
          3) legacy keys/tensors averaged on the diagonal
        """

        def diag_avg(m):
            # scalar
            if isinstance(m, (int, float)):
                return float(m)
            # 3-element vector
            if isinstance(m, (list, tuple)) and len(m) == 3 and all(isinstance(x, (int, float)) for x in m):
                return float(sum(m) / 3.0)
            # 3x3 tensor
            if isinstance(m, (list, tuple)) and len(m) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in m):
                try:
                    d = [m[0][0], m[1][1], m[2][2]]
                    if all(isinstance(x, (int, float)) for x in d):
                        return float(sum(d) / 3.0)
                except Exception:
                    pass
            # dict with typical summary keys
            if isinstance(m, dict):
                for k in ("poly", "average", "avg", "mean", "scalar"):
                    v = m.get(k)
                    if isinstance(v, (int, float)):
                        return float(v)
            return None

        if not dielectric:
            return None

        # 1) Preferred: e_electronic
        val = diag_avg(dielectric.get("e_electronic"))
        if isinstance(val, float):
            return val

        # 2) Fallback: e_total
        val = diag_avg(dielectric.get("e_total"))
        if isinstance(val, float):
            return val

        # 3) Legacy/common aliases (rare; keep for robustness)
        for key in ("k_electronic", "poly_electronic", "e_electronic_poly", "electronic_poly", "e_electronic_scalar"):
            v = dielectric.get(key)
            vv = diag_avg(v)
            if isinstance(vv, float):
                return vv

        for key in ("k_total", "poly_total", "e_total_poly", "total_poly", "e_total_scalar"):
            v = dielectric.get(key)
            vv = diag_avg(v)
            if isinstance(vv, float):
                return vv

        # 4) Tensor-ish fallbacks
        for key in ("electronic_tensor", "epsilon_infinity"):
            v = dielectric.get(key)
            vv = diag_avg(v)
            if isinstance(vv, float):
                return vv

        for key in ("total_tensor",):
            v = dielectric.get(key)
            vv = diag_avg(v)
            if isinstance(vv, float):
                return vv

        return None

    # ---------------- Elasticity / Bulk modulus ----------------

    def get_elastic_data(self, material_id: str) -> Dict:
        """
        Get elastic (mechanical) data for a material (bulk/shear moduli).
        """
        url = f"{self.base_url}/materials/elasticity"
        params = {"material_ids": material_id, "_all_fields": "true"}
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=30)
            if r.status_code == 200:
                data = (r.json() or {}).get("data", [])
                if data:
                    return data[0] or {}
            else:
                print(f"Elasticity query failed: HTTP {r.status_code} - {r.text[:200]}")
        except Exception as e:
            print(f"Error getting elasticity data for {material_id}: {e}")
        return {}

    @staticmethod
    def _extract_bulk_from_elastic_doc(elastic: Dict) -> Optional[float]:
        """
        Extract a scalar bulk modulus (GPa) from an elasticity document.
        Preference: VRH average, then Voigt, then Reuss. Also supports nested dicts.
        May compute from elastic_tensor if needed and numpy is available.
        """
        if not elastic:
            return None

        # 1) direct scalar keys (common variants)
        for k in ("k_vrh", "K_VRH", "Kvrh"):
            v = elastic.get(k)
            if isinstance(v, (int, float)):
                return float(v)

        # 2) nested: bulk_modulus = {vrh/voigt/reuss} (case-insensitive)
        bulk = elastic.get("bulk_modulus")
        if isinstance(bulk, dict):
            for key in ("vrh", "VRH", "Vrh"):
                v = bulk.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
            for key in ("voigt", "Voigt", "reuss", "Reuss"):
                v = bulk.get(key)
                if isinstance(v, (int, float)):
                    return float(v)

        # 3) other direct keys
        for k in ("k_voigt", "K_Voigt", "Kvoigt", "Kvoight"):
            v = elastic.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        for k in ("k_reuss", "K_Reuss", "Kreuss"):
            v = elastic.get(k)
            if isinstance(v, (int, float)):
                return float(v)

        # 4) compute VRH from 6x6 elastic tensor if present
        cij = elastic.get("elastic_tensor")
        if _HAVE_NUMPY and isinstance(cij, (list, tuple)):
            try:
                C = np.array(cij, dtype=float)
                if C.ndim == 2 and C.shape == (6, 6):
                    K_V = (C[0,0] + C[1,1] + C[2,2] + 2.0*(C[0,1] + C[0,2] + C[1,2])) / 9.0
                    S = np.linalg.inv(C)
                    denom = (S[0,0] + S[1,1] + S[2,2] + 2.0*(S[0,1] + S[0,2] + S[1,2]))
                    if denom != 0:
                        K_R = 1.0 / denom
                        return float(0.5*(K_V + K_R))
                    else:
                        return float(K_V)
            except Exception:
                pass

        return None

    def find_best_polymorph_with_cif_and_dielectric(self, cif_path: str, formula: str = None):
        """
        Like find_best_polymorph_with_cif, but returns the first CIF-matching polymorph
        that actually has dielectric data. Returns (best_polymorph, dielectric_doc) or (None, {}).
        """
        cif_data = self.parse_cif_file(cif_path)
        if not cif_data:
            return None, {}

        if not formula:
            formula = cif_data.get('formula')
            if not formula:
                return None, {}

        # Try to prefilter to entries that have dielectric data
        url = f"{self.base_url}/materials/summary"
        try:
            r = requests.get(
                url,
                headers=self.headers,
                params={"formula": formula, "_all_fields": "true", "has_props": "dielectric"},
                timeout=30,
            )
            data = (r.json() or {}).get("data", []) if r.status_code == 200 else []
        except Exception:
            data = []

        if not data:
            data = self.get_all_polymorphs(formula)

        polys = [p for p in data if p.get('structure') and p['structure'].get('lattice')]
        if not polys:
            return None, {}

        # Rank by CIF similarity
        scored = [(p, self.calculate_structural_similarity(cif_data, p)) for p in polys]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Walk down candidates until one has dielectric
        for p, _score in scored:
            mpid = p.get("material_id")
            if not mpid:
                continue
            d = self.get_dielectric_data(mpid)
            if d:
                return p, d

        return None, {}


    def find_best_polymorph_with_cif_and_elasticity(self, cif_path: str, formula: str = None):
        """
        Like find_best_polymorph_with_cif, but returns the first CIF-matching polymorph
        that actually has elasticity data. Returns (best_polymorph, elastic_doc) or (None, {}).
        """
        cif_data = self.parse_cif_file(cif_path)
        if not cif_data:
            return None, {}

        if not formula:
            formula = cif_data.get('formula')
            if not formula:
                return None, {}

        # Fast path: try to only pull summaries that have elasticity (if supported)
        # If your deployment ignores this param, we’ll just fall back to all polymorphs.
        url = f"{self.base_url}/materials/summary"
        try:
            r = requests.get(
                url,
                headers=self.headers,
                params={"formula": formula, "_all_fields": "true", "has_props": "elasticity"},
                timeout=30,
            )
            data = (r.json() or {}).get("data", []) if r.status_code == 200 else []
        except Exception:
            data = []

        if not data:
            data = self.get_all_polymorphs(formula)

        polys = [p for p in data if p.get('structure') and p['structure'].get('lattice')]
        if not polys:
            return None, {}

        # Rank by CIF similarity
        scored = [(p, self.calculate_structural_similarity(cif_data, p)) for p in polys]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Walk down candidates until one has elasticity
        for p, _score in scored:
            mpid = p.get("material_id")
            if not mpid:
                continue
            elastic = self.get_elastic_data(mpid)
            if elastic:  # found one with data
                return p, elastic

        # none had elasticity
        return None, {}

    @staticmethod
    def _extract_shear_vrh_from_elastic_doc(elastic: Dict) -> Optional[float]:
        """
        Extract the VRH shear modulus G (GPa) from an elasticity document.
        Preference: direct VRH → nested dict VRH → Voigt/Reuss average → compute from elastic tensor.
        """
        if not elastic:
            return None

        # 1) direct scalar keys (common variants)
        for k in ("g_vrh", "G_VRH", "Gvrh"):
            v = elastic.get(k)
            if isinstance(v, (int, float)):
                return float(v)

        # 2) nested: shear_modulus = {vrh/voigt/reuss}
        sm = elastic.get("shear_modulus")
        if isinstance(sm, dict):
            for key in ("vrh", "VRH", "Vrh"):
                v = sm.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
            # If only Voigt/Reuss exist, average them
            gv = sm.get("voigt")
            gr = sm.get("reuss")
            if isinstance(gv, (int, float)) and isinstance(gr, (int, float)):
                return float(0.5 * (gv + gr))

        # 3) other direct keys
        gv = elastic.get("g_voigt") or elastic.get("G_Voigt") or elastic.get("Gvoigt")
        gr = elastic.get("g_reuss") or elastic.get("G_Reuss") or elastic.get("Greuss")
        if isinstance(gv, (int, float)) and isinstance(gr, (int, float)):
            return float(0.5 * (gv + gr))
        if isinstance(gv, (int, float)):
            return float(gv)
        if isinstance(gr, (int, float)):
            return float(gr)

        # 4) compute from elastic tensor if present
        cij = elastic.get("elastic_tensor")
        if _HAVE_NUMPY and isinstance(cij, (list, tuple)):
            try:
                C = np.array(cij, dtype=float)
                if C.ndim == 2 and C.shape == (6, 6):
                    # Voigt shear modulus:
                    # G_V = (C11 + C22 + C33 - C12 - C13 - C23 + 3*(C44 + C55 + C66)) / 15
                    G_V = (C[0,0] + C[1,1] + C[2,2]
                           - C[0,1] - C[0,2] - C[1,2]
                           + 3.0*(C[3,3] + C[4,4] + C[5,5])) / 15.0
                    # Reuss shear modulus:
                    # G_R = 15 / (4*(S11+S22+S33) - 4*(S12+S13+S23) + 3*(S44+S55+S66))
                    S = np.linalg.inv(C)
                    denom = (4.0*(S[0,0] + S[1,1] + S[2,2])
                             - 4.0*(S[0,1] + S[0,2] + S[1,2])
                             + 3.0*(S[3,3] + S[4,4] + S[5,5]))
                    if denom != 0:
                        G_R = 15.0 / denom
                        return float(0.5*(G_V + G_R))
                    else:
                        return float(G_V)
            except Exception:
                pass

        return None

    def get_thermoelectric_data(self, material_id: str) -> Dict:
        """
        Get thermoelectric data for a material from Materials Project API.
        This includes Seebeck coefficient, electrical conductivity, and thermal conductivity.
        Note: This may not be available for all materials as thermoelectric data is limited.
        """
        url = f"{self.base_url}/materials/thermo"
        params = {"material_ids": material_id, "_all_fields": "true"}
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=30)
            if r.status_code == 200:
                data = (r.json() or {}).get("data", [])
                if data:
                    return data[0] or {}
            else:
                print(f"Thermo query failed: HTTP {r.status_code} - {r.text[:200]}")
        except Exception as e:
            print(f"Error getting thermo data for {material_id}: {e}")
        return {}

    def get_seebeck_data(self, material_id: str) -> Dict:
        """
        Get Seebeck coefficient data for a material from Materials Project API.
        Note: This may not be available for all materials as thermoelectric data is limited.
        """
        return self.get_thermoelectric_data(material_id)

    def _extract_seebeck_n_type_coefficient(self, thermo_data: Dict) -> Optional[float]:
        """
        Extract n-type Seebeck coefficient from thermo data.
        Look specifically for n-type Seebeck coefficient values in μV/K.
        """
        if not thermo_data:
            return None
        
        # Look for Seebeck coefficient in various possible keys, prioritizing n-type
        seebeck_keys = [
            'seebeck_coefficient_n', 'seebeck_n', 's_n', 'thermopower_n',
            'seebeck_coefficient', 'seebeck', 'thermopower'
        ]
        
        for key in seebeck_keys:
            value = thermo_data.get(key)
            if value is not None:
                if isinstance(value, (int, float)):
                    # Convert to μV/K if needed (Materials Project might use different units)
                    return float(value) if value > 1 else float(value) * 1e6
                elif isinstance(value, dict):
                    # Look for n-type specific values first
                    n_value = value.get('n', value.get('n_type', value.get('negative')))
                    if n_value is not None and isinstance(n_value, (int, float)):
                        return float(n_value) if n_value > 1 else float(n_value) * 1e6
                    # If no n-type specific value, use general value
                    general_value = value.get('value', value.get('avg', value.get('average')))
                    if general_value is not None and isinstance(general_value, (int, float)):
                        return float(general_value) if general_value > 1 else float(general_value) * 1e6
        
        return None

    def _extract_thermal_conductivity(self, thermo_data: Dict) -> Optional[float]:
        """
        Extract thermal conductivity from thermo data.
        Look for thermal conductivity values in W/(m·K).
        """
        if not thermo_data:
            return None
        
        # Look for thermal conductivity in various possible keys
        thermal_conductivity_keys = [
            'thermal_conductivity', 'k_thermal', 'k_lattice', 'lattice_thermal_conductivity',
            'thermal_conductivity_lattice', 'k', 'thermal_k'
        ]
        
        for key in thermal_conductivity_keys:
            value = thermo_data.get(key)
            if value is not None:
                if isinstance(value, (int, float)):
                    # Assume value is already in W/(m·K)
                    return float(value)
                elif isinstance(value, dict):
                    # Look for specific values
                    specific_value = value.get('value', value.get('avg', value.get('average')))
                    if specific_value is not None and isinstance(specific_value, (int, float)):
                        return float(specific_value)
        
        return None

    def get_electrical_conductivity_data(self, material_id: str) -> Dict:
        """
        Get electrical conductivity data for a material from Materials Project API.
        Note: This may not be available for all materials as transport data is limited.
        """
        url = f"{self.base_url}/materials/electronic_structure"
        params = {"material_ids": material_id, "_all_fields": "true"}
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=30)
            if r.status_code == 200:
                data = (r.json() or {}).get("data", [])
                if data:
                    return data[0] or {}
            else:
                print(f"Electronic structure query failed: HTTP {r.status_code} - {r.text[:200]}")
        except Exception as e:
            print(f"Error getting electronic structure data for {material_id}: {e}")
        return {}

    def _extract_electrical_conductivity(self, electronic_data: Dict) -> Optional[float]:
        """
        Extract electrical conductivity from electronic structure data.
        Look for conductivity values in S/m.
        """
        if not electronic_data:
            return None
        
        # Look for electrical conductivity in various possible keys
        conductivity_keys = [
            'electrical_conductivity', 'conductivity', 'sigma', 'sigma_e',
            'electrical_resistivity', 'resistivity', 'rho'
        ]
        
        for key in conductivity_keys:
            value = electronic_data.get(key)
            if value is not None:
                if isinstance(value, (int, float)):
                    # If it's resistivity, convert to conductivity (σ = 1/ρ)
                    if 'resistivity' in key or 'rho' in key:
                        return 1.0 / float(value) if value != 0 else None
                    else:
                        # Assume it's already conductivity in S/m
                        return float(value)
                elif isinstance(value, dict):
                    # Look for n-type specific values
                    n_value = value.get('n', value.get('n_type', value.get('negative')))
                    if n_value is not None and isinstance(n_value, (int, float)):
                        if 'resistivity' in key or 'rho' in key:
                            return 1.0 / float(n_value) if n_value != 0 else None
                        else:
                            return float(n_value)
        
        return None

    # ---------------- High-level property aggregation ----------------

    def get_all_5_properties(self, cif_path: str, formula: str = None) -> Dict:
        """Get all 5 available properties and filter out incomplete polymorphs"""
        best_polymorph = self.find_best_polymorph_with_cif(cif_path, formula)
        if not best_polymorph:
            return {}
        
        material_id = best_polymorph.get('material_id')
        
        properties = {
            'material_id': material_id,
            'formula': best_polymorph.get('formula_pretty'),
            'band_gap': best_polymorph.get('band_gap'),
            'formation_energy': best_polymorph.get('formation_energy_per_atom'),
            'energy_above_hull': best_polymorph.get('energy_above_hull'),
            'density': best_polymorph.get('density'),
            'is_stable': best_polymorph.get('is_stable'),
            'space_group': best_polymorph.get('symmetry', {}).get('symbol'),
            'crystal_system': best_polymorph.get('symmetry', {}).get('crystal_system')
        }
        
        # Dielectric: prefer a scalar electronic; fallback to total or tensor avg
        dielectric = self.get_dielectric_data(material_id)
        properties['dielectric_constant'] = self._extract_scalar_dielectric(dielectric)
        
        # Thermoelectric properties: Seebeck n-type coefficient, electrical conductivity, and thermal conductivity
        thermo_data = self.get_thermoelectric_data(material_id)
        properties['seebeck_n_type_coefficient'] = self._extract_seebeck_n_type_coefficient(thermo_data)
        properties['thermal_conductivity'] = self._extract_thermal_conductivity(thermo_data)
        
        electrical_conductivity_data = self.get_electrical_conductivity_data(material_id)
        properties['electrical_conductivity'] = self._extract_electrical_conductivity(electrical_conductivity_data)

        return properties
    
    def filter_complete_polymorphs(self, properties: Dict) -> bool:
        """Check if all 5 available properties are present and not None"""
        required_properties = [
            'band_gap', 'formation_energy', 'energy_above_hull', 
            'density', 'dielectric_constant'
        ]
        for prop in required_properties:
            if properties.get(prop) is None:
                return False
        return True
    
    def has_required_property(self, properties: Dict, property_name: str) -> bool:
        """Check if a specific property is present and not None"""
        return properties.get(property_name) is not None


# ---------------- Global API instance ----------------

api = MaterialsProjectPropertyAPI()

# ---------------- Individual property functions ----------------

def get_band_gap(cif_path: str, formula: str = None) -> Optional[float]:
    """Get band gap for a material using CIF-based polymorph matching"""
    properties = api.get_all_5_properties(cif_path, formula)
    if api.has_required_property(properties, 'band_gap'):
        return properties.get('band_gap')
    return None

def get_formation_energy(cif_path: str, formula: str = None) -> Optional[float]:
    """Get formation energy for a material using CIF-based polymorph matching"""
    properties = api.get_all_5_properties(cif_path, formula)
    if api.has_required_property(properties, 'formation_energy'):
        return properties.get('formation_energy')
    return None

def get_energy_above_hull(cif_path: str, formula: str = None) -> Optional[float]:
    """Get energy above hull for a material using CIF-based polymorph matching"""
    properties = api.get_all_5_properties(cif_path, formula)
    if api.has_required_property(properties, 'energy_above_hull'):
        return properties.get('energy_above_hull')
    return None

def get_density(cif_path: str, formula: str = None) -> Optional[float]:
    """Get density for a material using CIF-based polymorph matching"""
    properties = api.get_all_5_properties(cif_path, formula)
    if api.has_required_property(properties, 'density'):
        return properties.get('density')
    return None

def get_dielectric_constant(cif_path: str, formula: str = None) -> Optional[float]:
    """
    Return a scalar dielectric constant using CIF-based polymorph matching.
    Prefer e_electronic; fallback to e_total/tensors.
    """
    # First try: find a polymorph that actually has dielectric data
    best, dielectric = api.find_best_polymorph_with_cif_and_dielectric(cif_path, formula)
    if best and dielectric:
        k = api._extract_scalar_dielectric(dielectric)
        if isinstance(k, (int, float)):
            return float(k)

    # Fallback: use the best summary match, then try dielectric once
    props = api.get_all_5_properties(cif_path, formula)
    if not props:
        return None

    # If _extract_scalar_dielectric already found something in get_all_5_properties, return it
    k_cached = props.get('dielectric_constant')
    if isinstance(k_cached, (int, float)):
        return float(k_cached)

    mpid = props.get('material_id')
    if not mpid:
        return None

    dielectric2 = api.get_dielectric_data(mpid)
    return api._extract_scalar_dielectric(dielectric2)

def get_shear_modulus_vrh(cif_path: str, formula: str = None) -> Optional[float]:
    """
    Get shear modulus G (VRH, GPa) using CIF-based polymorph matching + MP elasticity data.
    Prefers VRH; falls back to Voigt/Reuss; tensor-based compute as last resort.
    """
    # Prefer a polymorph that *actually* has elasticity data
    best, elastic = api.find_best_polymorph_with_cif_and_elasticity(cif_path, formula)
    if best and elastic:
        g = api._extract_shear_vrh_from_elastic_doc(elastic)
        if isinstance(g, (int, float)):
            return float(g)

    # Fallback: try summary (some deployments expose a scalar 'shear_modulus' on /summary)
    best_summary = api.find_best_polymorph_with_cif(cif_path, formula)
    if not best_summary:
        return None

    # Summary may already hold a scalar VRH shear modulus
    v = best_summary.get("shear_modulus")
    if isinstance(v, (int, float)):
        return float(v)
    # Or a dict with 'vrh'
    if isinstance(v, dict):
        for k in ("vrh", "VRH", "Vrh"):
            vv = v.get(k)
            if isinstance(vv, (int, float)):
                return float(vv)
        gv = v.get("voigt"); gr = v.get("reuss")
        if isinstance(gv, (int, float)) and isinstance(gr, (int, float)):
            return float(0.5*(gv+gr))

    # Last resort: pull elasticity once for this mp-id
    mpid = best_summary.get("material_id")
    if not mpid:
        return None
    elastic2 = api.get_elastic_data(mpid)
    return api._extract_shear_vrh_from_elastic_doc(elastic2)


def get_bulk_modulus(cif_path: str, formula: str = None) -> Optional[float]:
    """
    Get bulk modulus (GPa) using CIF-based polymorph matching + MP elasticity data.
    Prefers VRH average; falls back to Voigt/Reuss; tensor-based compute as last resort.
    """
    # Prefer a polymorph that *actually* has elasticity data
    best, elastic = api.find_best_polymorph_with_cif_and_elasticity(cif_path, formula)
    if best and elastic:
        k = api._extract_bulk_from_elastic_doc(elastic)
        if isinstance(k, (int, float)):
            return float(k)

    # Fallback: if none of the matched polymorphs have elasticity, try summary hints
    # (some deployments stash bulk in summary)
    best_summary = api.find_best_polymorph_with_cif(cif_path, formula)
    if not best_summary:
        return None

    # Sometimes summary might already hold bulk fields
    for key in ("k_vrh", "K_VRH", "Kvrh"):
        v = best_summary.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    bulk = best_summary.get("bulk_modulus")
    if isinstance(bulk, dict):
        for k in ("vrh", "VRH", "Vrh", "voigt", "Voigt", "reuss", "Reuss"):
            v = bulk.get(k)
            if isinstance(v, (int, float)):
                return float(v)

    # As a last resort, try elasticity again for this single mpid
    mpid = best_summary.get("material_id")
    if not mpid:
        return None
    elastic2 = api.get_elastic_data(mpid)
    return api._extract_bulk_from_elastic_doc(elastic2)

def get_thermal_conductivity(cif_path: str, formula: str = None) -> Optional[float]:
    """
    Get thermal conductivity (W/(m·K)) using CIF-based polymorph matching + MP thermoelectric data.
    """
    best_polymorph = api.find_best_polymorph_with_cif(cif_path, formula)
    if not best_polymorph:
        return None
    
    material_id = best_polymorph.get('material_id')
    if not material_id:
        return None
    
    thermo_data = api.get_thermoelectric_data(material_id)
    return api._extract_thermal_conductivity(thermo_data)

def get_seebeck_n_type_coefficient(cif_path: str, formula: str = None) -> Optional[float]:
    """
    Get n-type Seebeck coefficient (μV/K) using CIF-based polymorph matching + MP thermoelectric data.
    """
    best_polymorph = api.find_best_polymorph_with_cif(cif_path, formula)
    if not best_polymorph:
        return None
    
    material_id = best_polymorph.get('material_id')
    if not material_id:
        return None
    
    thermo_data = api.get_thermoelectric_data(material_id)
    return api._extract_seebeck_n_type_coefficient(thermo_data)

def get_all_properties(cif_path: str, formula: str = None) -> Dict:
    """
    Get all available properties for a material using CIF-based polymorph matching
    Returns dict with available properties (may be partial)
    """
    properties = api.get_all_5_properties(cif_path, formula)
    if properties:
        try:
            bulk = get_bulk_modulus(cif_path, formula)
        except Exception:
            bulk = None
        try:
            thermal_cond = get_thermal_conductivity(cif_path, formula)
        except Exception:
            thermal_cond = None
        try:
            seebeck_n = get_seebeck_n_type_coefficient(cif_path, formula)
        except Exception:
            seebeck_n = None
        return {
            'band_gap': properties.get('band_gap'),
            'formation_energy': properties.get('formation_energy'),
            'energy_above_hull': properties.get('energy_above_hull'),
            'density': properties.get('density'),
            'dielectric_constant': properties.get('dielectric_constant'),
            'bulk_modulus': bulk,
            'thermal_conductivity': thermal_cond,
            'seebeck_n_type_coefficient': seebeck_n
        }
    return {}

def get_property_with_metadata(cif_path: str, formula: str = None) -> Dict:
    """
    Get all properties with metadata (material_id, formula, etc.)
    Returns empty dict if no polymorph found
    """
    properties = api.get_all_5_properties(cif_path, formula)
    if properties:
        return properties
    return {}

# ---------------- Example usage and testing ----------------

if __name__ == "__main__":
    print("Materials Project Property Functions")
    print("Individual functions for each of the 5 available properties")
    print("Uses CIF-based polymorph validation")
    # print()
    
    # Test with the initial base CIF file from the Aerospace run (Al2O3)
    batio3_cif = os.path.join(os.path.dirname(__file__), "..", "runs", "run_Aerospace_20250921_091028", "Structural_Materials_for_Aerospace", "candidate_init_base.cif")
    if os.path.exists(batio3_cif):
        print("Testing individual property functions with Al2O3:")
        print(f"CIF file: {batio3_cif}")
        print()
        
        print("Individual property functions:")
        print(f"  Band Gap: {get_band_gap(batio3_cif)} eV")
        print(f"  Formation Energy: {get_formation_energy(batio3_cif)} eV/atom")
        print(f"  Energy Above Hull: {get_energy_above_hull(batio3_cif)} eV/atom")
        print(f"  Density: {get_density(batio3_cif)} g/cm³")
        print(f"  Dielectric Constant: {get_dielectric_constant(batio3_cif)}")
        print(f"  Bulk Modulus: {get_bulk_modulus(batio3_cif)} GPa")
        print(f"  Shear Modulus: {get_shear_modulus_vrh(batio3_cif)} GPa")
        print(f"  Thermal Conductivity: {get_thermal_conductivity(batio3_cif)} W/(m·K)")
        print(f"  Seebeck N-type Coefficient: {get_seebeck_n_type_coefficient(batio3_cif)} μV/K")
        print()
        
        print("All properties function:")
        all_props = get_all_properties(batio3_cif)
        if all_props:
            for prop, value in all_props.items():
                print(f"  {prop}: {value}")
        else:
            print("  No complete polymorph found")
        print()
        
        print("Properties with metadata:")
        props_with_meta = get_property_with_metadata(batio3_cif)
        if props_with_meta:
            print(f"  Material ID: {props_with_meta.get('material_id')}")
            print(f"  Formula: {props_with_meta.get('formula')}")
            print(f"  Space Group: {props_with_meta.get('space_group')}")
            print(f"  Crystal System: {props_with_meta.get('crystal_system')}")
        else:
            print("  No complete polymorph found")
    else:
        print(f"❌ CIF file not found: {batio3_cif}")