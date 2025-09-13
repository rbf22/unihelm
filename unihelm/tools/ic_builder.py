"""
UniHelm Internal Coordinate (IC) Builder (v2)
---------------------------------------------
This module builds a 3D molecular structure for a single monomer from a
complete set of internal coordinates (bonds, angles, dihedrals).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import math

def _find_in_geom(geom_list, atom_names):
    """Helper to find a geometry definition (bond, angle, etc.) for a set of atoms."""
    target_set = set(atom_names)
    for item in geom_list:
        if set(item['atoms']) == target_set:
            return item
    return None

def place_atom(a_coord, b_coord, c_coord, bond_length, bond_angle_rad, dihedral_angle_rad):
    """
    Places a new atom (D) given the coordinates of three previous atoms (A, B, C)
    and the internal coordinates that define the relationship between them.
    """
    # Vector from C to B
    bc_vec = b_coord - c_coord
    bc_vec /= np.linalg.norm(bc_vec)

    # Vector from B to A
    ab_vec = a_coord - b_coord
    ab_vec /= np.linalg.norm(ab_vec)

    # Normal vector to the plane A-B-C
    n_vec = np.cross(ab_vec, bc_vec)
    n_vec /= np.linalg.norm(n_vec)

    # Cross product for the second axis of the local reference frame
    nbc_vec = np.cross(n_vec, bc_vec)

    # Build the rotation matrix to align the new atom
    M = np.array([bc_vec, nbc_vec, n_vec]).T

    # Coordinates of the new atom in the local reference frame of C
    d_local = np.array([
        -bond_length * np.cos(bond_angle_rad),
        bond_length * np.sin(bond_angle_rad) * np.cos(dihedral_angle_rad),
        bond_length * np.sin(bond_angle_rad) * np.sin(dihedral_angle_rad)
    ])

    # Transform local coordinates to global coordinates
    d_global = c_coord + M.dot(d_local)
    return d_global

def build_molecule_from_ics(smiles, atom_names_map, geometry):
    """
    Updates the 3D coordinates of a molecule (created from SMILES) based on a
    complete geometry definition.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer()

    atom_name_to_idx = {str(k): int(v) for k, v in atom_names_map.items()}
    idx_to_name = {v: k for k, v in atom_name_to_idx.items()}

    atom_coords = {}

    adj = {name: [] for name in atom_name_to_idx.keys()}
    for bond in geometry['bonds']:
        a1, a2 = bond['atoms']
        adj[a1].append(a2)
        adj[a2].append(a1)

    placed_atoms = set()
    q = []

    ref_b_name, ref_c_name = geometry['bonds'][0]['atoms']
    ref_a_name = next(neighbor for neighbor in adj[ref_b_name] if neighbor != ref_c_name)

    atom_coords[ref_a_name] = np.array([0.0, 0.0, 0.0])
    placed_atoms.add(ref_a_name)
    q.extend(adj[ref_a_name])

    bond_len = _find_in_geom(geometry['bonds'], [ref_a_name, ref_b_name])['length']
    atom_coords[ref_b_name] = np.array([bond_len, 0.0, 0.0])
    placed_atoms.add(ref_b_name)
    q.extend(adj[ref_b_name])

    bond_len_bc = _find_in_geom(geometry['bonds'], [ref_b_name, ref_c_name])['length']
    angle_abc = _find_in_geom(geometry['angles'], [ref_a_name, ref_b_name, ref_c_name])['angle']
    angle_rad = math.radians(180.0 - angle_abc)
    coords_c = np.array([
        atom_coords[ref_b_name][0] - bond_len_bc * np.cos(angle_rad),
        atom_coords[ref_b_name][1] + bond_len_bc * np.sin(angle_rad),
        0.0
    ])
    atom_coords[ref_c_name] = coords_c
    placed_atoms.add(ref_c_name)
    q.extend(adj[ref_c_name])

    visited_in_q = set([ref_a_name, ref_b_name, ref_c_name])

    while q:
        d_name = q.pop(0)
        if d_name in placed_atoms or d_name in visited_in_q:
            continue
        visited_in_q.add(d_name)

        c_name, b_name, a_name = None, None, None
        for neighbor in adj[d_name]:
            if neighbor in placed_atoms: c_name = neighbor; break
        if not c_name: continue
        for neighbor in adj[c_name]:
            if neighbor in placed_atoms and neighbor != d_name: b_name = neighbor; break
        if not b_name: continue
        for neighbor in adj[b_name]:
            if neighbor in placed_atoms and neighbor != c_name: a_name = neighbor; break
        if not a_name: continue

        bond = _find_in_geom(geometry['bonds'], [c_name, d_name])
        angle = _find_in_geom(geometry['angles'], [b_name, c_name, d_name])
        dihedral = _find_in_geom(geometry['dihedrals'], [a_name, b_name, c_name, d_name])

        if not all([bond, angle, dihedral]):
            continue

        new_coords = place_atom(
            atom_coords[a_name], atom_coords[b_name], atom_coords[c_name],
            bond['length'], math.radians(angle['angle']), math.radians(dihedral['dihedral'])
        )
        atom_coords[d_name] = new_coords
        placed_atoms.add(d_name)

        for neighbor in adj[d_name]:
            if neighbor not in placed_atoms:
                q.append(neighbor)

    for name, coords in atom_coords.items():
        conf.SetAtomPosition(atom_name_to_idx[name], coords)

    return mol
