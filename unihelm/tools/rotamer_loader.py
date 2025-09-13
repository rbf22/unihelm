"""
UniHelm Rotamer Loader
----------------------

This module handles loading monomer data from YAML files that follow the
v2.0 format, where geometry is separated into a constant part (bonds, angles)
and a variable part (dihedrals within rotamers).
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

def get_rotamer_by_id(monomer_data, rotamer_id):
    """
    Finds and returns a specific rotamer block by its ID.
    """
    for rotamer in monomer_data.get("rotamers", []):
        if rotamer.get("id") == rotamer_id:
            return rotamer
    raise ValueError(f"Rotamer '{rotamer_id}' not found in {monomer_data['monomer_id']}")

def get_complete_geometry(monomer_data, rotamer_id):
    """
    Constructs a complete geometry set (bonds, angles, dihedrals) for a
    specific rotamer.
    """
    base_geometry = monomer_data.get("geometry", {})
    rotamer = get_rotamer_by_id(monomer_data, rotamer_id)

    complete_geometry = {
        "bonds": base_geometry.get("bonds", []),
        "angles": base_geometry.get("angles", []),
        "dihedrals": rotamer.get("dihedrals", [])
    }
    return complete_geometry

def get_random_rotamer_geometry(monomer_data):
    """
    Selects a random rotamer and returns its complete geometry set.
    """
    rotamers = monomer_data.get("rotamers", [])
    if not rotamers:
        raise ValueError(f"No rotamers found for {monomer_data['monomer_id']}")

    random_rotamer = random.choice(rotamers)
    return get_complete_geometry(monomer_data, random_rotamer['id'])

if __name__ == '__main__':
    # Example usage
    ala_data = load_monomer_data("ALA", "PEPTIDE")

    # Get the geometry for the first rotamer
    rotamer1_geom = get_complete_geometry(ala_data, "ala_rotamer_1")
    print("--- Geometry for ALA Rotamer 1 ---")
    print(f"Bonds: {len(rotamer1_geom['bonds'])}")
    print(f"Angles: {len(rotamer1_geom['angles'])}")
    print(f"Dihedrals: {len(rotamer1_geom['dihedrals'])}")
    print(f"First dihedral: {rotamer1_geom['dihedrals'][0]}")

    # Get the geometry for a random rotamer
    random_geom = get_random_rotamer_geometry(ala_data)
    print("\n--- Geometry for a random ALA rotamer ---")
    print(f"First dihedral: {random_geom['dihedrals'][0]}")
