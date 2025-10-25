from pymatgen.core import Structure

def build_cif(candidate: dict, output_path: str) -> None:
    # This is a placeholder. Actual implementation will depend on candidate fields.
    # Example assumes candidate has 'lattice', 'species', and 'coords'.
    structure = Structure(
        lattice=candidate['lattice'],
        species=candidate['species'],
        coords=candidate['coords']
    )
    structure.to(filename=output_path) 