import os
from unihelm.tools.conformation_generator import generate_conformation_library

PEPTIDE_SMILES = {
    "ALA": "[N:1][C@@H](C)[C:2](=O)O",
    "ARG": "[N:1][C@@H](CCCNC(=N)N)[C:2](=O)O",
    "ASN": "[N:1][C@@H](CC(=O)N)[C:2](=O)O",
    "ASP": "[N:1][C@@H](CC(=O)O)[C:2](=O)O",
    "CYS": "[N:1][C@@H](CS)[C:2](=O)O",
    "GLN": "[N:1][C@@H](CCC(=O)N)[C:2](=O)O",
    "GLU": "[N:1][C@@H](CCC(=O)O)[C:2](=O)O",
    "GLY": "[N:1]C[C:2](=O)O",
    "HIS": "[N:1][C@@H](CC1=CN=CN1)[C:2](=O)O",
    "ILE": "[N:1][C@@H](C(C)CC)[C:2](=O)O",
    "LEU": "[N:1][C@@H](CC(C)C)[C:2](=O)O",
    "LYS": "[N:1][C@@H](CCCCN)[C:2](=O)O",
    "MET": "[N:1][C@@H](CCSC)[C:2](=O)O",
    "PHE": "[N:1][C@@H](CC1=CC=CC=C1)[C:2](=O)O",
    "PRO": "[N:1]1CCCC1[C:2](=O)O",
    "SER": "[N:1][C@@H](CO)[C:2](=O)O",
    "THR": "[N:1][C@@H](C(C)O)[C:2](=O)O",
    "TRP": "[N:1][C@@H](CC1=CNC2=CC=CC=C12)[C:2](=O)O",
    "TYR": "[N:1][C@@H](CC1=CC=C(O)C=C1)[C:2](=O)O",
    "VAL": "[N:1][C@@H](C(C)C)[C:2](=O)O",
}

output_dir = "unihelm/monomers/PEPTIDE"

for monomer_id, smiles in PEPTIDE_SMILES.items():
    print(f"--- Generating {monomer_id} ---")
    generate_conformation_library(
        monomer_id,
        "PEPTIDE",
        smiles,
        num_confs=50,
        output_dir=output_dir
    )
    print(f"--- Done {monomer_id} ---")

print("\nAll monomers regenerated.")
