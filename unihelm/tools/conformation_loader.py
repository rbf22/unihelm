"""
UniHelm Conformation Loader
----------------------

This module handles loading monomer data from YAML files that follow the
v2.0 format, where geometry is separated into a constant part (bonds, angles)
and a variable part (dihedrals within conformations).
"""

import os
import yaml
import random

# Assuming this script is in unihelm/tools, the monomers are in ../monomers
MONOMERS_DIR = os.path.join(os.path.dirname(__file__), "../monomers")

def load_monomer_data(monomer_id, monomer_type):
    """
    Loads the full YAML data for a specific monomer.
    """
    path = os.path.join(MONOMERS_DIR, monomer_type, f"{monomer_id}.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_conformation_by_id(monomer_data, conformation_id):
    """
    Finds and returns a specific conformation block by its ID.
    """
    for conformation in monomer_data.get("conformations", []):
        if conformation.get("id") == conformation_id:
            return conformation
    raise ValueError(f"Conformation '{conformation_id}' not found in {monomer_data['monomer_id']}")

def get_complete_geometry(monomer_data, conformation_id):
    """
    Constructs a complete geometry set (bonds, angles, dihedrals) for a
    specific conformation.
    """
    base_geometry = monomer_data.get("geometry", {})
    conformation = get_conformation_by_id(monomer_data, conformation_id)

    complete_geometry = {
        "bonds": base_geometry.get("bonds", []),
        "angles": base_geometry.get("angles", []),
        "dihedrals": conformation.get("dihedrals", [])
    }
    return complete_geometry

def get_random_conformation_geometry(monomer_data):
    """
    Selects a random conformation and returns its complete geometry set.
    """
    conformations = monomer_data.get("conformations", [])
    if not conformations:
        raise ValueError(f"No conformations found for {monomer_data['monomer_id']}")

    random_conformation = random.choice(conformations)
    return get_complete_geometry(monomer_data, random_conformation['id'])

if __name__ == '__main__':
    # Example usage
    ala_data = load_monomer_data("ALA", "PEPTIDE")

    # Get the geometry for the first conformation
    conformation1_geom = get_complete_geometry(ala_data, "ala_conformation_1")
    print("--- Geometry for ALA Conformation 1 ---")
    print(f"Bonds: {len(conformation1_geom['bonds'])}")
    print(f"Angles: {len(conformation1_geom['angles'])}")
    print(f"Dihedrals: {len(conformation1_geom['dihedrals'])}")
    print(f"First dihedral: {conformation1_geom['dihedrals'][0]}")

    # Get the geometry for a random conformation
    random_geom = get_random_conformation_geometry(ala_data)
    print("\n--- Geometry for a random ALA conformation ---")
    print(f"First dihedral: {random_geom['dihedrals'][0]}")
