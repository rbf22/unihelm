"""
UniHelm Conformation Generator
-------------------------

This tool programmatically generates UniHelm v2.1 monomer definition files.
It performs a conformational search to find low-energy conformations and extracts
their internal coordinates.
"""

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os

def cluster_conformers(mol, results):
    """
    Clusters conformers based on RMSD and returns the lowest-energy
    conformer ID from each cluster.
    """
    if not results:
        return []

    print("Clustering conformers...")
    energies = np.array([res[1] for res in results])
    rms_matrix = AllChem.GetConformerRMSMatrix(mol, prealigned=False)

    threshold = 0.5 # Angstroms
    clusters = []
    covered_cids = set()

    sorted_cids = [results[i][0] for i in np.argsort(energies)]

    for cid in sorted_cids:
        if cid in covered_cids:
            continue

        new_cluster = [cid]
        covered_cids.add(cid)

        for other_cid in sorted_cids:
            if other_cid in covered_cids:
                continue

            rms = rms_matrix[cid * len(sorted_cids) + other_cid]
            if rms < threshold:
                new_cluster.append(other_cid)
                covered_cids.add(other_cid)
        clusters.append(new_cluster)

    print(f"Found {len(clusters)} clusters.")

    conformation_cids = []
    for cluster in clusters:
        min_energy = float('inf')
        best_cid = -1
        for cid in cluster:
            energy = next(res[1] for res in results if res[0] == cid)
            if energy < min_energy:
                min_energy = energy
                best_cid = cid
        if best_cid != -1:
            conformation_cids.append(best_cid)

    return conformation_cids, energies

def get_atom_name_map(mol):
    """
    Creates a deterministic map from atom index to a simple, unique name
    (e.g., C1, N1, H1) based on the canonical atom order.
    """
    name_map = {}
    elem_counts = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        elem_counts[symbol] = elem_counts.get(symbol, 0) + 1
        name_map[idx] = f"{symbol}{elem_counts[symbol]}"
    return name_map

def extract_bonds(mol, name_map):
    bonds = []
    conf = mol.GetConformer(0) # Assume constant bond lengths across conformations
    for bond in mol.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist = np.linalg.norm(conf.GetAtomPosition(idx1) - conf.GetAtomPosition(idx2))
        bonds.append({
            "atoms": sorted([name_map[idx1], name_map[idx2]]),
            "length": float(round(dist, 3)),
            "type": int(bond.GetBondType())
        })
    return bonds

def extract_angles(mol, name_map):
    angles = []
    for i in range(mol.GetNumAtoms()):
        for j in mol.GetAtomWithIdx(i).GetNeighbors():
            for k in mol.GetAtomWithIdx(j.GetIdx()).GetNeighbors():
                if i == k.GetIdx(): continue
                angle_deg = AllChem.GetAngleDeg(mol.GetConformer(0), i, j.GetIdx(), k.GetIdx())
                angles.append({
                    "atoms": [name_map[i], name_map[j.GetIdx()], name_map[k.GetIdx()]],
                    "angle": float(round(angle_deg, 1))
                })
    return angles

def extract_dihedrals(mol, name_map, conf_id):
    dihedrals = []
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for i_atom in mol.GetAtomWithIdx(j).GetNeighbors():
            i = i_atom.GetIdx()
            if i == k: continue
            for l_atom in mol.GetAtomWithIdx(k).GetNeighbors():
                l = l_atom.GetIdx()
                if l == j: continue
                dih_deg = AllChem.GetDihedralDeg(mol.GetConformer(conf_id), i, j, k, l)
                dihedrals.append({
                    "atoms": [name_map[i], name_map[j], name_map[k], name_map[l]],
                    "dihedral": round(dih_deg, 1)
                })
    return dihedrals

def generate_conformation_library(monomer_id, monomer_type, smiles_with_maps, num_confs=200, output_dir="."):
    """
    Main orchestration function.
    Generates a full monomer definition from a SMILES string with atom maps for connections.
    """
    mol = Chem.MolFromSmiles(smiles_with_maps)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles_with_maps}")

    connection_atoms = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() != 0:
            connection_atoms[atom.GetAtomMapNum()] = atom.GetIdx()

    # Canonicalize atom order
    new_order = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    mol = Chem.RenumberAtoms(mol, new_order)

    # After renumbering, the original indices are invalid. We need to map them.
    old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}

    updated_connection_atoms = {}
    for map_num, old_idx in connection_atoms.items():
        updated_connection_atoms[map_num] = old_to_new_map[old_idx]
    connection_atoms = updated_connection_atoms

    # Clear map numbers for clean SMILES generation
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(canonical_smiles)
    mol = Chem.AddHs(mol)

    print(f"Generating {num_confs} initial conformers for {canonical_smiles}...")
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=AllChem.ETKDGv3())
    print("Optimizing conformers with UFF...")
    results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)

    mol_with_results, successful_results = mol, [(cid, energy) for cid, energy in results if cid != -1]

    if not successful_results:
        print("No stable conformers found. Aborting.")
        return

    conformation_cids, energies = cluster_conformers(mol_with_results, successful_results)
    print(f"\nIdentified {len(conformation_cids)} unique conformations (lowest energy from each cluster).")

    idx_to_name_map = get_atom_name_map(mol_with_results)
    name_to_idx_map = {v: k for k, v in idx_to_name_map.items()}

    # Build the new nested atom_names structure
    atom_names_structure = {}
    for name, idx in name_to_idx_map.items():
        atom = mol_with_results.GetAtomWithIdx(idx)
        atom_names_structure[name] = {
            'idx': idx,
            'iupac': name,
            'mass': round(atom.GetMass(), 4),
            'charge': 0.0,
            'type': None
        }

    connections = []
    for map_num in sorted(connection_atoms.keys()):
        final_idx = connection_atoms[map_num]
        final_atom = mol_with_results.GetAtomWithIdx(final_idx)

        removed_atom_name = ""
        for neighbor in final_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'H':
                removed_atom_name = idx_to_name_map[neighbor.GetIdx()]
                break

        connections.append({
            'id': f'R{map_num}',
            'label': f'R{map_num}',
            'connect_atom': idx_to_name_map[final_idx],
            'remove_atoms_on_connect': [removed_atom_name] if removed_atom_name else []
        })

    base_geometry = {
        "bonds": extract_bonds(mol_with_results, idx_to_name_map),
        "angles": extract_angles(mol_with_results, idx_to_name_map)
    }

    conformations_list = []
    min_energy = energies.min() if energies.size > 0 else 0.0
    for i, cid in enumerate(conformation_cids):
        conf_energy = next((res[1] for res in results if res[0] == cid), 0.0)
        conformations_list.append({
            "id": f"{monomer_id.lower()}_conformation_{i+1}",
            "relative_energy_kcal_mol": float(round(conf_energy - min_energy, 4)),
            "dihedrals": [
                {"atoms": d["atoms"], "dihedral": float(d["dihedral"])}
                for d in extract_dihedrals(mol_with_results, idx_to_name_map, cid)
            ]
        })

    final_yaml_data = {
        "format": "UniHelm YAML",
        "version": "2.3", # Version bump for format change
        "monomer_id": monomer_id,
        "polymer_type": monomer_type,
        "smiles": canonical_smiles,
        "atom_names": atom_names_structure,
        "connections": connections,
        "geometry": base_geometry,
        "conformations": conformations_list
    }

    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{monomer_id}.yaml")

    print(f"\nWriting conformation library to '{output_filename}'...")
    with open(output_filename, 'w') as f:
        yaml.dump(final_yaml_data, f, sort_keys=False, indent=2)
    print("Done.")

if __name__ == '__main__':
    MONOMER_ID = "ALA"
    MONOMER_TYPE = "PEPTIDE"
    SMILES_WITH_MAPS = "[N:1][C@@H](C)[C:2](=O)O"
    output_directory = "generated_monomers"
    generate_conformation_library(MONOMER_ID, MONOMER_TYPE, SMILES_WITH_MAPS, output_dir=output_directory)
