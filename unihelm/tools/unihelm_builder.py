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
from rdkit.Chem.rdchem import AtomPDBResidueInfo

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

def find_atom_index_by_name(atom_names_map, atom_name, monomer_id=None):
    # If monomer_id is provided, the map is global (index -> monomer:name)
    if monomer_id:
        search_name = f"{monomer_id}:{atom_name}"
        for idx_str, name in atom_names_map.items():
            if name == search_name:
                return int(idx_str)
    # If no monomer_id, the map is local (index -> name)
    else:
        search_name = atom_name
        for idx_str, name in atom_names_map.items():
            if name == search_name:
                return int(idx_str)

    raise ValueError(f"Atom '{search_name}' not found in atom_names_map")

def remove_atoms_by_names(rwmol, atom_names_map, names_to_remove, monomer_id):
    """
    Removes specified atoms and any attached hydrogen atoms that would be left with no other bonds.
    """
    idxs_to_remove = set()
    for name in names_to_remove:
        try:
            idx = find_atom_index_by_name(atom_names_map, name, monomer_id)
            atom = rwmol.GetAtomWithIdx(idx)

            # Find hydrogens attached only to this atom
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1 and neighbor.GetDegree() == 1:
                    log(f"   [Cap Removal] Also removing attached H (idx {neighbor.GetIdx()})")
                    idxs_to_remove.add(neighbor.GetIdx())

            idxs_to_remove.add(idx)
            log(f"   [Cap Removal] Removing atom {monomer_id}:{name} (idx {idx})")
        except ValueError as e:
            log(f"   [Cap Removal] Atom {monomer_id}:{name} not found - skipping. Reason: {e}")

    # Remove all collected atoms, sorted descending to not mess up indices
    for idx in sorted(list(idxs_to_remove), reverse=True):
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
    positioned_atom_names_map = {}
    prev_monomer_yaml = None

    for i, entry in enumerate(seq_def["sequence"]):
        m_id, m_type = entry["monomer"], entry["type"]
        log(f"\n[Monomer {i+1}] {m_id} ({m_type})")

        current_monomer_yaml = load_monomer(m_id, m_type)
        # The YAML has name:index, so we invert it to get index:name for our internal use
        name_to_idx_map = current_monomer_yaml.get("atom_names", {})
        idx_to_name_map = {str(v): k for k, v in name_to_idx_map.items()}

        mol = Chem.MolFromSmiles(current_monomer_yaml["smiles"])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        rw_mol = Chem.RWMol(mol)

        # Set PDB info for each atom in the current monomer
        residue_name = m_id.ljust(3)
        residue_number = i + 1
        for atom in rw_mol.GetAtoms():
            atom_idx_str = str(atom.GetIdx())
            atom_name = idx_to_name_map.get(atom_idx_str, atom.GetSymbol())
            info = AtomPDBResidueInfo()
            info.SetName(f" {atom_name.ljust(3)}")
            info.SetResidueName(residue_name)
            info.SetResidueNumber(residue_number)
            info.SetIsHeteroAtom(False)
            atom.SetMonomerInfo(info)

        if i == 0:
            positioned_mol = rw_mol
            # Create the initial uniquely-named map from index -> monomer:atom_name
            for idx, name in idx_to_name_map.items():
                positioned_atom_names_map[idx] = f"{m_id}:{name}"
        else:
            prev_monomer_id = prev_monomer_yaml['monomer_id']
            prev_polymer_type = prev_monomer_yaml['polymer_type']
            prev_conns = [merge_standard(c, std_conns) for c in prev_monomer_yaml.get("connections", [])]
            curr_conns = [merge_standard(c, std_conns) for c in current_monomer_yaml.get("connections", [])]

            if prev_polymer_type == "PEPTIDE":
                prev_cterm = next((c for c in prev_conns if "C_term" in c["label"]), None)
                curr_nterm = next((c for c in curr_conns if "N_term" in c["label"]), None)
            elif prev_polymer_type == "RNA":
                prev_cterm = next((c for c in prev_conns if "3_prime" in c["label"]), None)
                curr_nterm = next((c for c in curr_conns if "5_prime" in c["label"]), None)
            else:
                raise ValueError(f"Unsupported polymer type for connection: {prev_polymer_type}")

            if not prev_cterm or not curr_nterm:
                raise ValueError(f"Could not find required connection points for linking {prev_monomer_id} to {m_id}")

            log(f"[Connection] Linking {prev_monomer_id}->{m_id}")
            log(f"   Prev connect_atom: {prev_cterm['connect_atom']}")
            log(f"   Curr connect_atom: {curr_nterm['connect_atom']}")

            remove_atoms_by_names(positioned_mol, positioned_atom_names_map, prev_cterm.get("remove_atoms_on_connect", []), prev_monomer_id)
            # For local cap removal, we must create a temporary map with prefixed names
            # so that find_atom_index_by_name can find them correctly.
            local_map_for_removal = {idx: f"{m_id}:{name}" for idx, name in idx_to_name_map.items()}
            remove_atoms_by_names(rw_mol, local_map_for_removal, curr_nterm.get("remove_atoms_on_connect", []), m_id)

            # Programmatically remove capping H from previous monomer's connection atom (e.g., O3')
            # This is more robust than relying on named H atoms in the monomer files.
            prev_idx_for_H_removal = find_atom_index_by_name(positioned_atom_names_map, prev_cterm["connect_atom"], prev_monomer_id)
            prev_atom_for_H_removal = positioned_mol.GetAtomWithIdx(prev_idx_for_H_removal)
            if prev_atom_for_H_removal.GetAtomicNum() == 8: # If it's an Oxygen...
                h_to_remove = -1
                for neighbor in prev_atom_for_H_removal.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        h_to_remove = neighbor.GetIdx()
                        break
                if h_to_remove != -1:
                    log(f"   [Cap Removal] Programmatically removing H (idx {h_to_remove}) from {prev_monomer_id}:{prev_cterm['connect_atom']}")
                    positioned_mol.RemoveAtom(h_to_remove)

            positioned_conf = positioned_mol.GetConformer()
            prev_idx = find_atom_index_by_name(positioned_atom_names_map, prev_cterm["connect_atom"], prev_monomer_id)
            prev_coord = np.array(positioned_conf.GetAtomPosition(prev_idx))
            prev_neigh_idx = [n.GetIdx() for n in positioned_mol.GetAtomWithIdx(prev_idx).GetNeighbors()][0]
            prev_neigh_coord = np.array(positioned_conf.GetAtomPosition(prev_neigh_idx))

            curr_conf = rw_mol.GetConformer()
            # For local search, pass the local index:name map and no monomer_id
            curr_idx = find_atom_index_by_name(idx_to_name_map, curr_nterm["connect_atom"])
            curr_coord = np.array(curr_conf.GetAtomPosition(curr_idx))
            curr_neigh_idx = [n.GetIdx() for n in rw_mol.GetAtomWithIdx(curr_idx).GetNeighbors()][0]
            curr_neigh_coord = np.array(curr_conf.GetAtomPosition(curr_neigh_idx))

            log(f"   Bond length: {prev_cterm['bond']['length']}")
            log(f"   Bond angle: {prev_cterm.get('angle',{}).get('value','N/A')}")
            log(f"   Dihedral: {prev_cterm.get('dihedral',{}).get('default','N/A')}")

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

            bond_len = prev_cterm["bond"]["length"]
            dir_vec = prev_coord - prev_neigh_coord
            dir_vec /= np.linalg.norm(dir_vec)
            apply_translation(curr_conf, prev_coord + dir_vec * bond_len)

            offset = positioned_mol.GetNumAtoms()
            combo = Chem.CombineMols(positioned_mol, rw_mol)
            combo_rw = Chem.RWMol(combo)
            combo_rw.AddBond(prev_idx, offset + curr_idx, Chem.rdchem.BondType.SINGLE)

            # Merge atom name maps with unique names
            new_positioned_atom_names_map = dict(positioned_atom_names_map)
            for idx, name in idx_to_name_map.items():
                new_positioned_atom_names_map[str(int(idx) + offset)] = f"{m_id}:{name}"

            positioned_mol = combo_rw
            positioned_atom_names_map = new_positioned_atom_names_map

        prev_monomer_yaml = current_monomer_yaml

    Chem.SanitizeMol(positioned_mol)
    return positioned_mol
