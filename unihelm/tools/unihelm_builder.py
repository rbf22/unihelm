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

def find_atom_index_by_name(atom_names_map, atom_name):
    for idx_str, name in atom_names_map.items():
        if name == atom_name:
            return int(idx_str)
    raise ValueError(f"Atom '{atom_name}' not found in atom_names_map")

def remove_atoms_by_names(rwmol, atom_names_map, names_to_remove):
    idxs = []
    for name in names_to_remove:
        try:
            idx = find_atom_index_by_name(atom_names_map, name)
            idxs.append(idx)
            log(f"   [Cap Removal] Removing atom {name} (idx {idx})")
        except ValueError:
            log(f"   [Cap Removal] Atom {name} not found - skipping")
    for idx in sorted(idxs, reverse=True):
        rwmol.RemoveAtom(idx)

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
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
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
    prev_yaml = None

    for i, entry in enumerate(seq_def["sequence"]):
        m_id, m_type = entry["monomer"], entry["type"]
        log(f"\n[Monomer {i+1}] {m_id} ({m_type})")
        monomer_yaml = load_monomer(m_id, m_type)
        atom_names_map = monomer_yaml.get("atom_names", {})
        mol = Chem.MolFromSmiles(monomer_yaml["smiles"])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        rw = Chem.RWMol(mol)

        if i == 0:
            positioned_mol = rw
            prev_yaml = monomer_yaml
        else:
            prev_conns = [merge_standard(c, std_conns) for c in prev_yaml.get("connections", [])]
            curr_conns = [merge_standard(c, std_conns) for c in monomer_yaml.get("connections", [])]

            prev_cterm = next(c for c in prev_conns if "C_term" in c["label"])
            curr_nterm = next(c for c in curr_conns if "N_term" in c["label"])

            log(f"[Connection] Linking {prev_yaml['monomer_id']}->{monomer_yaml['monomer_id']}")
            log(f"   Prev connect_atom: {prev_cterm['connect_atom']}")
            log(f"   Curr connect_atom: {curr_nterm['connect_atom']}")

            remove_atoms_by_names(positioned_mol, prev_yaml["atom_names"], prev_cterm.get("remove_atoms_on_connect", []))
            remove_atoms_by_names(rw, monomer_yaml["atom_names"], curr_nterm.get("remove_atoms_on_connect", []))

            positioned_conf = positioned_mol.GetConformer()
            prev_idx = find_atom_index_by_name(prev_yaml["atom_names"], prev_cterm["connect_atom"])
            prev_coord = np.array(positioned_conf.GetAtomPosition(prev_idx))
            prev_neigh_idx = [n.GetIdx() for n in positioned_mol.GetAtomWithIdx(prev_idx).GetNeighbors()][0]
            prev_neigh_coord = np.array(positioned_conf.GetAtomPosition(prev_neigh_idx))

            curr_conf = rw.GetConformer()
            curr_idx = find_atom_index_by_name(monomer_yaml["atom_names"], curr_nterm["connect_atom"])
            curr_coord = np.array(curr_conf.GetAtomPosition(curr_idx))
            curr_neigh_idx = [n.GetIdx() for n in rw.GetAtomWithIdx(curr_idx).GetNeighbors()][0]
            curr_neigh_coord = np.array(curr_conf.GetAtomPosition(curr_neigh_idx))

            log(f"   Bond length: {prev_cterm['bond']['length']}")
            log(f"   Bond angle: {prev_cterm.get('angle',{}).get('value','N/A')}")
            log(f"   Dihedral: {prev_cterm.get('dihedral',{}).get('default','N/A')}")

            # Move curr connect atom to origin
            apply_translation(curr_conf, -curr_coord)
            # Rotate bond vector to -X axis
            bond_vec = curr_neigh_coord - curr_coord
            R_align = rotation_matrix_from_vectors(bond_vec, np.array([-1.0, 0, 0]))
            apply_rotation(curr_conf, R_align)

            # Set bond angle
            if "angle" in prev_cterm:
                angle_rad = math.radians(prev_cterm["angle"]["value"])
                v1 = prev_neigh_coord - prev_coord
                v1 /= np.linalg.norm(v1)
                tilt_axis = np.cross(np.array([1.0, 0, 0]), v1)
                if np.linalg.norm(tilt_axis) > 1e-8:
                    tilt_axis /= np.linalg.norm(tilt_axis)
                    cos_a = math.cos(math.pi - angle_rad)
                    sin_a = math.sin(math.pi - angle_rad)
                    K = np.array([[0, -tilt_axis[2], tilt_axis[1]],
                                  [tilt_axis[2], 0, -tilt_axis[0]],
                                  [-tilt_axis[1], tilt_axis[0], 0]])
                    R_tilt = np.identity(3) + sin_a * K + (1 - cos_a) * (K @ K)
                    apply_rotation(curr_conf, R_tilt)

            # Apply dihedral if present
            if "dihedral" in prev_cterm and "default" in prev_cterm["dihedral"]:
                dihedral_rad = math.radians(prev_cterm["dihedral"]["default"])
                axis = np.array([1.0, 0, 0])
                Kd = np.array([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]])
                R_dih = np.identity(3) + math.sin(dihedral_rad) * Kd + (1 - math.cos(dihedral_rad)) * (Kd @ Kd)
                apply_rotation(curr_conf, R_dih)

            # Translate so bond length is correct
            bond_len = prev_cterm["bond"]["length"]
            dir_vec = prev_coord - prev_neigh_coord
            dir_vec /= np.linalg.norm(dir_vec)
            apply_translation(curr_conf, prev_coord + dir_vec * bond_len)

            # Merge into positioned_mol
            combo = Chem.CombineMols(positioned_mol, rw)
            combo = Chem.RWMol(combo)
            offset = positioned_mol.GetNumAtoms()

            combo.AddBond(prev_idx, offset + curr_idx, Chem.rdchem.BondType.SINGLE)

            positioned_mol = combo
            prev_yaml = monomer_yaml

    Chem.SanitizeMol(positioned_mol)
    return positioned_mol

def main(seq_file):
    seq_def = load_yaml(seq_file)
    mol = build_sequence(seq_def)
    Chem.MolToMolFile(mol, "output_ic_debug.mol")
    with open("output_ic_debug.pdb", "w") as f:
        f.write(Chem.MolToPDBBlock(mol))
    print("[OK] Fully parametric IC build with debug complete: output_ic_debug.mol / output_ic_debug.pdb")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build molecule from UniHelm sequence fully parametrically with debug")
    ap.add_argument("sequence_file", help="Path to sequence YAML")
    args = ap.parse_args()
    main(args.sequence_file)