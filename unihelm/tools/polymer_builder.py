"""
UniHelm Polymer Builder
-----------------------

This tool builds a 3D structure for a full polymer chain by connecting
individual monomers, each built with a specific rotamer geometry.

The connection logic uses rigid-body transformations (rotations and translations)
to place each new monomer relative to the growing chain.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import math
import yaml
import os

# We will need the other tools
try:
    from . import ic_builder
    from . import rotamer_loader
except ImportError:
    import ic_builder
    import rotamer_loader

# --- Vector and Matrix Math Helpers ---

def get_rotation_matrix(axis, angle_rad):
    """
    Returns the rotation matrix for a given axis and angle.
    (Rodrigues' rotation formula)
    """
    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle_rad / 2.0)
    b, c, d = -axis * math.sin(angle_rad / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])

def apply_transform(mol, rotation_matrix, translation_vector):
    """
    Applies a rotation and translation to all atoms in a molecule's conformer.
    """
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        new_pos = np.dot(rotation_matrix, pos) + translation_vector
        conf.SetAtomPosition(i, new_pos)

# --- Main Polymer Build Function (to be implemented) ---

def build_polymer(sequence_with_rotamers):
    """
    Builds a full polymer 3D structure from a sequence of pre-loaded
    monomer and rotamer data.
    """
    if not sequence_with_rotamers:
        return None

    # Step 1: Build each monomer as a separate RDKit Mol object
    monomer_mols = []
    for entry in sequence_with_rotamers:
        monomer_data = entry['monomer_data']
        rotamer_data = entry['rotamer_data']

        geometry = {
            "bonds": monomer_data['geometry']['bonds'],
            "angles": monomer_data['geometry']['angles'],
            "dihedrals": rotamer_data['dihedrals']
        }

        mol = ic_builder.build_molecule_from_ics(
            monomer_data['smiles'],
            monomer_data['atom_names'],
            geometry
        )
        if mol:
            mol.SetProp("monomer_data", yaml.dump(monomer_data))
            monomer_mols.append(mol)

    if not monomer_mols:
        return None

    # Step 2: Iteratively connect the monomers
    polymer = monomer_mols[0]

    # Load standard connection templates
    with open(os.path.join(os.path.dirname(__file__), "../connections/standard_connections.yaml"), 'r') as f:
        std_conns_data = yaml.safe_load(f)
    std_conns = std_conns_data.get("standard_connections", {})

    for i in range(1, len(monomer_mols)):
        prev_mol = polymer
        curr_mol = monomer_mols[i]

        prev_data = yaml.safe_load(prev_mol.GetProp("monomer_data"))
        curr_data = yaml.safe_load(curr_mol.GetProp("monomer_data"))

        # Determine connection labels based on polymer type
        if prev_data['polymer_type'] == 'PEPTIDE':
            prev_label, curr_label = 'C_term', 'N_term'
        elif prev_data['polymer_type'] in ['RNA', 'DNA']:
            prev_label, curr_label = '3_prime', '5_prime'
        else:
            raise ValueError(f"Unsupported polymer type: {prev_data['polymer_type']}")

        prev_conn_info = next(c for c in prev_data['connections'] if prev_label in c['label'])
        curr_conn_info = next(c for c in curr_data['connections'] if curr_label in c['label'])

        conn_geom = std_conns.get(prev_conn_info.get("use_standard"), {})
        bond_len = conn_geom['bond']['length']
        angle = conn_geom['angle']['value']
        dihedral = conn_geom['dihedral']['default']

        prev_conf = prev_mol.GetConformer()
        prev_atom_names = prev_data['atom_names']

        # Define the three reference atoms from the previous monomer
        c_prev_name = prev_conn_info['connect_atom']
        c_prev_idx = prev_atom_names[c_prev_name]
        c_prev_coord = np.array(prev_conf.GetAtomPosition(c_prev_idx))

        b_prev_name = conn_geom['angle']['atoms'][0].replace('_prev','')
        b_prev_idx = prev_atom_names[b_prev_name]
        b_prev_coord = np.array(prev_conf.GetAtomPosition(b_prev_idx))

        a_prev_name = conn_geom['dihedral']['atoms'][0].replace('_prev','')
        a_prev_idx = prev_atom_names[a_prev_name]
        a_prev_coord = np.array(prev_conf.GetAtomPosition(a_prev_idx))

        # Use NeRF to calculate the position of the new monomer's first atom
        n_curr_name = curr_conn_info['connect_atom']
        n_curr_target_coord = ic_builder.place_atom(
            a_prev_coord, b_prev_coord, c_prev_coord,
            bond_len, math.radians(angle), math.radians(dihedral)
        )

        # --- Align the current monomer to this new position ---
        curr_conf = curr_mol.GetConformer()
        curr_atom_names = curr_data['atom_names']
        n_curr_idx = curr_atom_names[n_curr_name]
        n_curr_original_coord = np.array(curr_conf.GetAtomPosition(n_curr_idx))

        # Translate the current monomer to place its N-atom at the target
        translation = n_curr_target_coord - n_curr_original_coord
        apply_transform(curr_mol, np.identity(3), translation)

        # --- Now combine the chemically correct and geometrically placed parts ---
        rw_prev = Chem.RWMol(prev_mol)
        atoms_to_remove_prev = {prev_atom_names[name] for name in prev_conn_info.get('remove_atoms_on_connect', [])}
        for idx in sorted(list(atoms_to_remove_prev), reverse=True):
            rw_prev.RemoveAtom(idx)

        rw_curr = Chem.RWMol(curr_mol)
        atoms_to_remove_curr = {curr_atom_names[name] for name in curr_conn_info.get('remove_atoms_on_connect', [])}
        for idx in sorted(list(atoms_to_remove_curr), reverse=True):
            rw_curr.RemoveAtom(idx)

        clean_prev_mol = rw_prev.GetMol()
        clean_curr_mol = rw_curr.GetMol()

        # We assume indices of connection atoms don't change after removing terminal atoms
        prev_connect_atom_idx = prev_atom_names[prev_conn_info['connect_atom']]
        curr_connect_atom_idx = curr_atom_names[curr_conn_info['connect_atom']]

        offset = clean_prev_mol.GetNumAtoms()
        combined_mol = Chem.CombineMols(clean_prev_mol, clean_curr_mol)
        rw_polymer = Chem.RWMol(combined_mol)

        rw_polymer.AddBond(prev_connect_atom_idx, curr_connect_atom_idx + offset, Chem.BondType.SINGLE)
        polymer = rw_polymer.GetMol()
        polymer.SetProp("monomer_data", yaml.dump(prev_data))

    # A full implementation would need to carefully manage atom names and properties
    # across the combined molecule.
    return polymer

if __name__ == '__main__':
    # Example of how it would be used
    print("This script is a library and is not meant to be run directly yet.")
