#!/usr/bin/env python3
"""
UniHelm Builder v3.1 - Fully Parametric IC Builder with Debug Logging
--------------------------------------------------------------------
Applies bond length, bond angle, and dihedral from standard connection templates.
Removes caps, adds real chemical bonds, positions new monomers with exact geometry.
Logs all placement steps for verification.
"""

import os, yaml, math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

MONOMERS_DIR = os.path.join(os.path.dirname(__file__), "../monomers")
CONNECTIONS_FILE = os.path.join(os.path.dirname(__file__), "../connections/standard_connections.yaml")

DEBUG = True  # Set to False to suppress debug output

def log(msg):
    if DEBUG:
        print(msg)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_monomer(monomer_id, monomer_type):
    path = os.path.join(MONOMERS_DIR, monomer_type, f"{monomer_id}.yaml")
    return load_yaml(path)

def load_standard_connections():
    data = load_yaml(CONNECTIONS_FILE)
    return data.get("standard_connections", {})

def merge_standard(conn, std_conns):
    if "use_standard" in conn:
        tmpl = std_conns[conn["use_standard"]]
        merged = dict(conn)
        for k, v in tmpl.items():
            if k not in merged:
                merged[k] = v
        return merged
    return conn

def rotation_matrix_from_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, 1.0):
        return np.identity(3)
    if np.isclose(c, -1.0):
        return -np.identity(3)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.identity(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix

def apply_rotation(conf, rot_mat):
    for i in range(conf.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        conf.SetAtomPosition(i, rot_mat @ pos)

def apply_translation(conf, translation):
    for i in range(conf.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        conf.SetAtomPosition(i, pos + translation)

def build_sequence(seq_def):
    std_conns = load_standard_connections()
    positioned_mol = None
    positioned_atom_names_map = {}
    prev_monomer_yaml = None

    for i, entry in enumerate(seq_def["sequence"]):
        m_id, m_type = entry["monomer"], entry["type"]
        log(f"\n[Monomer {i+1}] {m_id} ({m_type})")

        current_monomer_yaml = load_monomer(m_id, m_type)
        current_atom_names = current_monomer_yaml.get("atom_names", {})
        current_semantic_map = current_monomer_yaml.get("semantic_map", {})

        mol = Chem.MolFromSmiles(current_monomer_yaml["smiles"])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        rw_mol = Chem.RWMol(mol)

        if i == 0:
            positioned_mol = rw_mol
            for name, data in current_atom_names.items():
                positioned_atom_names_map[f"{m_id}:{name}"] = str(data['idx'])
        else:
            prev_monomer_id = prev_monomer_yaml['monomer_id']
            prev_semantic_map = prev_monomer_yaml.get("semantic_map", {})
            reversed_prev_semantic_map = {v: k for k, v in prev_semantic_map.items()}
            reversed_current_semantic_map = {v: k for k, v in current_semantic_map.items()}

            prev_conns = [merge_standard(c, std_conns) for c in prev_monomer_yaml.get("connections", [])]
            curr_conns = [merge_standard(c, std_conns) for c in current_monomer_yaml.get("connections", [])]

            prev_cterm = next((c for c in prev_conns if "C_term" in c.get("label", "")), None) or next((c for c in prev_conns if c.get("id") == "R2"), None)
            curr_nterm = next((c for c in curr_conns if "N_term" in c.get("label", "")), None) or next((c for c in curr_conns if c.get("id") == "R1"), None)

            if not prev_cterm or not curr_nterm:
                raise ValueError(f"Could not find connection points between {prev_monomer_id} and {m_id}")

            log(f"[Connection] Linking {prev_monomer_id}->{m_id}")
            log(f"   Prev connect_atom: {prev_cterm['connect_atom']}")
            log(f"   Curr connect_atom: {curr_nterm['connect_atom']}")

            # --- Atom Removal ---
            atoms_to_remove_prev = set()
            for name in prev_cterm.get("remove_atoms_on_connect", []):
                generic_name = prev_semantic_map.get(name, name)
                atoms_to_remove_prev.add(int(positioned_atom_names_map[f"{prev_monomer_id}:{generic_name}"]))

            atoms_to_remove_curr = set()
            for name in curr_nterm.get("remove_atoms_on_connect", []):
                generic_name = current_semantic_map.get(name, name)
                atoms_to_remove_curr.add(current_atom_names[generic_name]['idx'])

            new_positioned_mol = Chem.RWMol(positioned_mol)
            for idx in sorted(list(atoms_to_remove_prev), reverse=True):
                new_positioned_mol.RemoveAtom(idx)

            new_rw_mol = Chem.RWMol(rw_mol)
            for idx in sorted(list(atoms_to_remove_curr), reverse=True):
                new_rw_mol.RemoveAtom(idx)

            positioned_mol = new_positioned_mol.GetMol()
            rw_mol = new_rw_mol.GetMol()

            # --- Geometry Placement ---
            prev_connect_generic = prev_semantic_map.get(prev_cterm['connect_atom'], prev_cterm['connect_atom'])
            prev_idx = int(positioned_atom_names_map[f"{prev_monomer_id}:{prev_connect_generic}"])

            # Use the semantic map to get the generic name for the current monomer
            curr_connect_semantic = curr_nterm['connect_atom']
            curr_connect_generic = current_semantic_map.get(curr_connect_semantic)
            if not curr_connect_generic:
                raise ValueError(f"Semantic name '{curr_connect_semantic}' not found in semantic_map for {m_id}")
            curr_idx = current_atom_names[curr_connect_generic]['idx']

            positioned_conf = positioned_mol.GetConformer()
            prev_coord = np.array(positioned_conf.GetAtomPosition(prev_idx))
            prev_neigh_idx = [n.GetIdx() for n in positioned_mol.GetAtomWithIdx(prev_idx).GetNeighbors()][0]
            prev_neigh_coord = np.array(positioned_conf.GetAtomPosition(prev_neigh_idx))

            curr_conf = rw_mol.GetConformer()
            curr_coord = np.array(curr_conf.GetAtomPosition(curr_idx))
            curr_neigh_idx = [n.GetIdx() for n in rw_mol.GetAtomWithIdx(curr_idx).GetNeighbors()][0]
            curr_neigh_coord = np.array(curr_conf.GetAtomPosition(curr_neigh_idx))

            bond_len = prev_cterm.get('bond', {}).get('length', 1.33)
            log(f"   Bond length: {bond_len}")

            # ... (rest of geometry logic is the same)

            apply_translation(curr_conf, -curr_coord)
            bond_vec = curr_neigh_coord - curr_coord
            R_align = rotation_matrix_from_vectors(bond_vec, np.array([-1.0, 0, 0]))
            apply_rotation(curr_conf, R_align)

            if "angle" in prev_cterm:
                angle_rad = math.radians(prev_cterm["angle"]["value"])
                v1 = prev_neigh_coord - prev_coord
                v1 /= np.linalg.norm(v1)
                tilt_axis = np.cross(np.array([1.0, 0, 0]), v1)
                if np.linalg.norm(tilt_axis) > 1e-8:
                    tilt_axis /= np.linalg.norm(tilt_axis)
                    cos_a = math.cos(math.pi - angle_rad)
                    sin_a = math.sin(math.pi - angle_rad)
                    K = np.array([[0, -tilt_axis[2], tilt_axis[1]], [tilt_axis[2], 0, -tilt_axis[0]], [-tilt_axis[1], tilt_axis[0], 0]])
                    R_tilt = np.identity(3) + sin_a * K + (1 - cos_a) * (K @ K)
                    apply_rotation(curr_conf, R_tilt)

            if "dihedral" in prev_cterm and "default" in prev_cterm["dihedral"]:
                dihedral_rad = math.radians(prev_cterm["dihedral"]["default"])
                axis = np.array([1.0, 0, 0])
                Kd = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                R_dih = np.identity(3) + math.sin(dihedral_rad) * Kd + (1 - math.cos(dihedral_rad)) * (Kd @ Kd)
                apply_rotation(curr_conf, R_dih)

            dir_vec = prev_coord - prev_neigh_coord
            dir_vec /= np.linalg.norm(dir_vec)
            apply_translation(curr_conf, prev_coord + dir_vec * bond_len)

            # --- Combine and Finalize ---
            offset = positioned_mol.GetNumAtoms()
            combo = Chem.CombineMols(positioned_mol, rw_mol)
            combo_rw = Chem.RWMol(combo)
            combo_rw.AddBond(prev_idx, offset + curr_idx, Chem.rdchem.BondType.SINGLE)

            if "dihedral" in prev_cterm:
                new_bond = combo_rw.GetBondBetweenAtoms(prev_idx, offset + curr_idx)
                if new_bond:
                    new_bond.SetBoolProp("is_rigid", True)
                    log(f"   [Rigidity] Marked bond between atoms {prev_idx} and {offset + curr_idx} as rigid.")

            new_positioned_atom_names_map = dict(positioned_atom_names_map)
            for name, data in current_atom_names.items():
                new_positioned_atom_names_map[f"{m_id}:{name}"] = str(int(data['idx']) + offset)

            positioned_mol = combo_rw
            positioned_atom_names_map = new_positioned_atom_names_map

        prev_monomer_yaml = current_monomer_yaml

    Chem.SanitizeMol(positioned_mol)
    return positioned_mol
