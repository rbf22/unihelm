import os
import yaml
from rdkit import Chem

PEPTIDE_DIR = "unihelm/monomers/PEPTIDE"

for filename in os.listdir(PEPTIDE_DIR):
    if not filename.endswith(".yaml"):
        continue

    filepath = os.path.join(PEPTIDE_DIR, filename)
    print(f"--- Processing {filename} ---")

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    mol = Chem.MolFromSmiles(data['smiles'])
    mol = Chem.AddHs(mol)

    atom_names = data['atom_names']
    idx_to_name = {v['idx']: k for k, v in atom_names.items()}

    # Find N and C termini from connections
    r1_atom_name, r2_atom_name = None, None
    for conn in data.get('connections', []):
        if conn.get('id') == 'R1':
            r1_atom_name = conn['connect_atom']
        elif conn.get('id') == 'R2':
            r2_atom_name = conn['connect_atom']

    if not r1_atom_name or not r2_atom_name:
        print(f"  Skipping {filename}, could not find R1/R2 connections.")
        continue

    n_idx = atom_names[r1_atom_name]['idx']
    c_idx = atom_names[r2_atom_name]['idx']

    # Find CA: the atom connected to both N and C
    ca_idx = -1
    n_atom = mol.GetAtomWithIdx(n_idx)
    c_atom = mol.GetAtomWithIdx(c_idx)

    # This logic is still a bit fragile, but better than before
    for n_neighbor in n_atom.GetNeighbors():
        for c_neighbor in c_atom.GetNeighbors():
            if n_neighbor.GetIdx() == c_neighbor.GetIdx() and n_neighbor.GetSymbol() == 'C':
                ca_idx = n_neighbor.GetIdx()
                break
        if ca_idx != -1:
            break

    if ca_idx == -1:
         # Fallback for Proline
        if data['monomer_id'] == 'PRO':
            for n_neighbor in n_atom.GetNeighbors():
                if n_neighbor.GetSymbol() == 'C' and n_neighbor.GetIdx() != c_idx:
                     ca_idx = n_neighbor.GetIdx()
                     break
        if ca_idx == -1:
            print(f"  Could not find Alpha Carbon for {filename}")
            continue

    semantic_map = {
        "N": idx_to_name[n_idx],
        "C": idx_to_name[c_idx],
        "CA": idx_to_name[ca_idx],
    }

    data['semantic_map'] = semantic_map

    for conn in data["connections"]:
        if conn.get("id") == "R1":
            conn["label"] = "N_term"
            conn["use_standard"] = "peptide_N"
            conn["connect_atom"] = "N"
        elif conn.get("id") == "R2":
            conn["label"] = "C_term"
            conn["use_standard"] = "peptide_C"
            conn["connect_atom"] = "C"

    with open(filepath, 'w') as f:
        yaml.dump(data, f, sort_keys=False, indent=2)

    print(f"  ... Patched {filename}")

print("\nAll monomers patched.")
