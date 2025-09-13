"""
UniHelm Rotamer Generator
-------------------------

This tool programmatically generates UniHelm v2.1 monomer definition files.
It performs a conformational search to find low-energy rotamers and extracts
their internal coordinates.
"""

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def generate_conformers(smiles, num_confs):
    """
    Generates and optimizes a set of conformers for a molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    print(f"Generating {num_confs} initial conformers...")
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=AllChem.ETKDGv3())

    print("Optimizing conformers with UFF...")
    results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)

    return mol, [(cid, energy) for cid, energy in results if cid != -1]

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

    # Use a simple clustering algorithm for now.
    # A more advanced method like Butina clustering could be used here.
    # For now, we'll just find the lowest energy conformer for each cluster.
    # This is a simplification; a full implementation is more complex.

    # Let's assume a simple RMSD threshold for clustering.
    threshold = 0.5 # Angstroms
    clusters = []
    covered_cids = set()

    # Sort conformers by energy
    sorted_cids = [results[i][0] for i in np.argsort(energies)]

    for cid in sorted_cids:
        if cid in covered_cids:
            continue

        # Start a new cluster with this conformer
        new_cluster = [cid]
        covered_cids.add(cid)

        # Find other conformers within the threshold
        for other_cid in sorted_cids:
            if other_cid in covered_cids:
                continue

            # Get RMSD from the pre-computed matrix
            rms = rms_matrix[cid * len(sorted_cids) + other_cid]
            if rms < threshold:
                new_cluster.append(other_cid)
                covered_cids.add(other_cid)
        clusters.append(new_cluster)

    print(f"Found {len(clusters)} clusters.")

    # Get the lowest energy conformer from each cluster
    rotamer_cids = []
    for cluster in clusters:
        min_energy = float('inf')
        best_cid = -1
        for cid in cluster:
            # Find the energy for this cid
            energy = next(res[1] for res in results if res[0] == cid)
            if energy < min_energy:
                min_energy = energy
                best_cid = cid
        if best_cid != -1:
            rotamer_cids.append(best_cid)

    return rotamer_cids, energies

def get_atom_name_map(mol):
    """
    Creates a map from atom index to a semantic name (e.g., CA, CB, N, C).
    Uses SMARTS matching to identify key atoms.
    """
    name_map = {}

    # Find backbone atoms first
    # Alpha-Carbon (connected to N, C, and another C)
    ca_pattern = Chem.MolFromSmarts('[CX4;H1]([N])([C])')
    # Carboxyl-Carbon
    c_pattern = Chem.MolFromSmarts('[CX3](=[O])[O]')
    # Backbone-Nitrogen
    n_pattern = Chem.MolFromSmarts('[NX3;H2]')
    # Carbonyl-Oxygen
    o_pattern = Chem.MolFromSmarts('[OX1]=C')
    # Hydroxyl-Oxygen (OXT)
    oxt_pattern = Chem.MolFromSmarts('[OX2;H1]-C=O')

    # Find and name them
    for pattern, name in [(ca_pattern, "CA"), (c_pattern, "C"), (n_pattern, "N"), (o_pattern, "O"), (oxt_pattern, "OXT")]:
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            name_map[matches[0][0]] = name

    # Find Beta-Carbon (CB) - the neighbor of CA that isn't N or C
    if "CA" in name_map.values():
        ca_idx = list(name_map.keys())[list(name_map.values()).index("CA")]
        ca_atom = mol.GetAtomWithIdx(ca_idx)
        for neighbor in ca_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C' and neighbor.GetIdx() not in name_map:
                name_map[neighbor.GetIdx()] = "CB"
                break

    # Name remaining atoms generically
    elem_counts = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in name_map:
            symbol = atom.GetSymbol()

            # Use a local dictionary for counts, not globals
            elem_counts[symbol] = elem_counts.get(symbol, 0) + 1

            if symbol == 'H':
                parent = atom.GetNeighbors()[0]
                parent_name = name_map.get(parent.GetIdx(), "X")
                # Find how many H's are already attached to this parent
                h_count_on_parent = 1
                for other_idx, other_name in name_map.items():
                    if other_name.startswith(f"H") and other_name.endswith(parent_name):
                        h_count_on_parent += 1
                name_map[idx] = f"H{h_count_on_parent}{parent_name}"
            else:
                name_map[idx] = f"{symbol}{elem_counts[symbol]}"

    # This is still not perfect. A truly robust system would need a full SMIRKS-based typing.
    # But this is much better.
    return name_map # Return the index -> name map

def extract_bonds(mol, name_map):
    bonds = []
    conf = mol.GetConformer(0) # Assume constant bond lengths across rotamers
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
    # Find all angle triplets
    for i in range(mol.GetNumAtoms()):
        for j in mol.GetAtomWithIdx(i).GetNeighbors():
            for k in mol.GetAtomWithIdx(j.GetIdx()).GetNeighbors():
                if i == k.GetIdx(): continue
                # Found angle i-j-k
                angle_deg = AllChem.GetAngleDeg(mol.GetConformer(0), i, j.GetIdx(), k.GetIdx())
                angles.append({
                    "atoms": [name_map[i], name_map[j.GetIdx()], name_map[k.GetIdx()]],
                    "angle": float(round(angle_deg, 1))
                })
    return angles

def extract_dihedrals(mol, name_map, conf_id):
    dihedrals = []
    # Find all dihedral quartets
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for i_atom in mol.GetAtomWithIdx(j).GetNeighbors():
            i = i_atom.GetIdx()
            if i == k: continue
            for l_atom in mol.GetAtomWithIdx(k).GetNeighbors():
                l = l_atom.GetIdx()
                if l == j: continue
                # Found dihedral i-j-k-l
                dih_deg = AllChem.GetDihedralDeg(mol.GetConformer(conf_id), i, j, k, l)
                dihedrals.append({
                    "atoms": [name_map[i], name_map[j], name_map[k], name_map[l]],
                    "dihedral": round(dih_deg, 1)
                })
    return dihedrals

def generate_rotamer_library(monomer_id, monomer_type, smiles, num_confs=200):
    """
    Main orchestration function.
    """
    mol, results = generate_conformers(smiles, num_confs)
    if not results:
        print("No stable conformers found. Aborting.")
        return

    rotamer_cids, energies = cluster_conformers(mol, results)
    print(f"\nIdentified {len(rotamer_cids)} unique rotamers (lowest energy from each cluster).")

    # --- Assemble the YAML data ---
    idx_to_name_map = get_atom_name_map(mol)
    name_to_idx_map = {v: k for k, v in idx_to_name_map.items()}

    base_geometry = {
        "bonds": extract_bonds(mol, idx_to_name_map),
        "angles": extract_angles(mol, idx_to_name_map)
    }

    rotamers_list = []
    min_energy = energies.min()
    for i, cid in enumerate(rotamer_cids):
        conf_energy = energies[cid]
        rotamers_list.append({
            "id": f"{monomer_id.lower()}_rotamer_{i+1}",
            "relative_energy_kcal_mol": float(round(conf_energy - min_energy, 4)),
            "dihedrals": [
                {"atoms": d["atoms"], "dihedral": float(d["dihedral"])}
                for d in extract_dihedrals(mol, idx_to_name_map, cid)
            ]
        })

    # --- Final YAML structure ---
    n_term_pattern = Chem.MolFromSmarts('[NX3;H2;!$(NC=O)]')
    c_term_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')

    n_term_match = mol.GetSubstructMatches(n_term_pattern)
    c_term_match = mol.GetSubstructMatches(c_term_pattern)

    if not n_term_match or not c_term_match:
        raise ValueError("Could not find standard peptide termini in SMILES.")

    n_term_idx = n_term_match[0][0]
    c_term_idx = c_term_match[0][0]

    c_term_hydroxyl_O_idx = -1
    for atom in mol.GetAtomWithIdx(c_term_idx).GetNeighbors():
        if atom.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(c_term_idx, atom.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
            c_term_hydroxyl_O_idx = atom.GetIdx()
            break

    c_term_remove_names = []
    if c_term_hydroxyl_O_idx != -1:
        c_term_remove_names.append(idx_to_name_map[c_term_hydroxyl_O_idx])
        for neighbor in mol.GetAtomWithIdx(c_term_hydroxyl_O_idx).GetNeighbors():
            if neighbor.GetSymbol() == 'H':
                c_term_remove_names.append(idx_to_name_map[neighbor.GetIdx()])
                break

    n_term_remove_names = [idx_to_name_map[n.GetIdx()] for n in mol.GetAtomWithIdx(n_term_idx).GetNeighbors() if n.GetSymbol() == 'H']
    # We only remove one H for the peptide bond
    n_term_remove_names = n_term_remove_names[:1]

    connections = [
        {'id': 'R1', 'label': 'N_term', 'connect_atom': idx_to_name_map[n_term_idx], 'remove_atoms_on_connect': n_term_remove_names, 'use_standard': 'peptide_N'},
        {'id': 'R2', 'label': 'C_term', 'connect_atom': idx_to_name_map[c_term_idx], 'remove_atoms_on_connect': c_term_remove_names, 'use_standard': 'peptide_C'}
    ]

    final_yaml_data = {
        "format": "UniHelm YAML",
        "version": "2.1",
        "monomer_id": monomer_id,
        "polymer_type": monomer_type,
        "smiles": smiles,
        "atom_names": name_to_idx_map,
        "connections": connections,
        "geometry": base_geometry,
        "rotamers": rotamers_list
    }

    output_filename = f"{monomer_id}.yaml"
    print(f"\nWriting rotamer library to '{output_filename}'...")
    with open(output_filename, 'w') as f:
        yaml.dump(final_yaml_data, f, sort_keys=False, indent=2)
    print("Done.")

if __name__ == '__main__':
    # --- Parameters for Alanine ---
    MONOMER_ID = "ALA"
    MONOMER_TYPE = "PEPTIDE"
    SMILES = "N[C@@H](C)C(=O)O"

    generate_rotamer_library(MONOMER_ID, MONOMER_TYPE, SMILES)
