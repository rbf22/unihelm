import pytest
import yaml
from rdkit import Chem
import os
from unihelm.tools.unihelm_builder import build_sequence, load_yaml

def test_pdb_output_for_rna_sequence(tmp_path):
    """
    Tests that the generated PDB file for an RNA sequence has correct
    residue numbering and IUPAC atom names.
    """
    # Load the RNA sequence
    seq_path = "unihelm/sequences/rna_augc.yaml"
    seq_def = load_yaml(seq_path)

    # Build the polymer
    polymer = build_sequence(seq_def)
    assert isinstance(polymer, Chem.Mol)

    # Write to a temporary PDB file
    pdb_path = tmp_path / "rna_output.pdb"
    Chem.MolToPDBFile(polymer, str(pdb_path))

    # Read and verify the PDB content
    with open(pdb_path, 'r') as f:
        pdb_content = f.read()

    atom_lines = [line for line in pdb_content.split('\n') if line.startswith('ATOM')]

    # Verify residue numbers
    residue_numbers = sorted(list(set([int(line[22:26]) for line in atom_lines])))
    assert residue_numbers == [1, 2, 3, 4]

    # Verify residue names
    residue_names = []
    current_res_num = 0
    for line in atom_lines:
        res_num = int(line[22:26])
        if res_num > current_res_num:
            residue_names.append(line[17:20].strip())
            current_res_num = res_num
    assert residue_names == ["A", "U", "G", "C"]

    # Verify atom names
    atom_names = [line[12:16].strip() for line in atom_lines]
    assert "N1" in atom_names
    assert "C1'" in atom_names
    assert "O5'" in atom_names
