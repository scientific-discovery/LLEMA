# Surrogate Models

This README provides a detailed overview of the surrogate models integrated into **LLEMA**, focusing on **ALIGNN** and **CGCNN** .  

## ALIGNN

**ALIGNN (Atomistic Line Graph Neural Network)** is used as a surrogate model for predicting materials properties.  
It integrates pretrained models available on the **JARVIS-DFT dataset** hosted on Figshare.

**Pretrained model source:**  
[ALIGNN models on JARVIS-DFT dataset (Figshare)](https://figshare.com/articles/dataset/ALIGNN_models_on_JARVIS-DFT_dataset/17005681/6)

---

### Directory Structure

Pretrained ALIGNN models are stored at:
`/home/reddy/llema/llema/src/surrogate_models/alignn/alignn/`

**Saved files:**

- jv_ehull_alignn.zip
- jv_bulk_modulus_kv_alignn.zip
- jv_dfpt_piezo_max_dielectric_alignn.zip
- jv_dfpt_piezo_max_dij_alignn.zip
- jv_ehull_alignn.zip
- jv_epsx_alignn.zip
- jv_formation_energy_peratom_alignn.zip
- jv_n-powerfact_alignn.zip
- jv_n-Seebeck_alignn.zip
- jv_optb88vdw_bandgap_alignn.zip
- jv_shear_modulus_gv_alignn.zip

### Modifications to `pretrained.py` for LLEMA

The main script uses a generic output format: `"Predicted value: <model_name> <file_path> <values>"` instead of model-specific formatted outputs. The `pretrained.py` file has been modified to include property-specific formatted print statements for better readability when using the CLI. The changes include:

- **Enhanced output formatting**: Added property-specific print statements that display formatted, unit-aware output for multiple properties:
  - Formation Energy: `Formation Energy: X.XXX eV/atom`
  - Band Gap: `Band Gap: X.XXX eV`
  - Energy Above Hull: `Energy above hull: X.XXX eV`
  - Bulk Modulus: `Bulk modulus: X.XXX GPa`
  - Shear Modulus: `Shear modulus: X.XXX GPa`
  - Dielectric Constant: `Dielectric constant: X.XXX`
  - Piezoelectric Properties: `Max piezo dielectric (κ): X.XXX` and `Max piezo d_ij: X.XXX pC/N`

 - Add property-specific print statements for human-readable, unit-aware outputs:
     ```python
    if model_name == "jv_formation_energy_peratom_alignn":
        print(f'Formation Energy: {out_data[0]:.3f} eV/atom')
    elif model_name == 'jv_optb88vdw_bandgap_alignn':
        print(f'Band Gap: {out_data[0]:.3f} eV')
    elif model_name == 'jv_ehull_alignn':
        print(f'Energy above hull: {out_data[0]:.3f} eV')
    elif model_name == 'jv_bulk_modulus_kv_alignn':
        print(f'Bulk modulus: {out_data[0]:.3f} GPa')
    elif model_name == 'jv_shear_modulus_gv_alignn':
        print(f'Shear modulus: {out_data[0]:.3f} GPa')
    elif model_name == 'jv_epsx_alignn':
        print(f'Dielectric constant: {out_data[0]:.3f}')
    elif model_name == 'jv_dfpt_piezo_max_dielectric_alignn':
        print(f'Max piezo dielectric (κ): {out_data[0]:.3f}')
    elif model_name == 'jv_dfpt_piezo_max_dij_alignn':
        print(f'Max piezo d_ij: {out_data[0]:.3f} pC/N')
     ```
    instead of 
    ```python
    print("Predicted value:", model_name, file_path, out_data)
    ```

    Comment out the lines:
    ```python
    print("Using chk file", tmp, "from ", chks)
    print("Path", os.path.abspath(path))
    print("Config", os.path.abspath(cfg))
    ``` 
---

### CGCNN

**Edit `predict.py`**
  - Comment out verbose model-loading print statements.
  - Update regression output section to include: **property-specific results**:
    Replace
    ```python
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
    ```
    With
    ```python
    if "band-gap" in args.modelpath:
        print(f"Band Gap: {mae_errors.avg:.3f} eV")
    elif "formation-energy" in args.modelpath:
        print(f"Formation Energy: {mae_errors.avg:.3f} eV/A")
    ```
    
---
