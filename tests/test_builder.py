import pytest
import yaml
from rdkit import Chem
from unihelm.tools.unihelm_builder import load_monomer, build_sequence, merge_standard

def test_load_monomer():
    monomer = load_monomer("ALA", "PEPTIDE")
    assert monomer["monomer_id"] == "ALA"
    assert monomer["polymer_type"] == "PEPTIDE"
    assert monomer["smiles"] is not None

def test_merge_standard():
    conn = {"use_standard": "R1-R2"}
    std_conns = {"R1-R2": {"bond": {"length": 1.5}}}
    merged = merge_standard(conn, std_conns)
    assert merged["bond"]["length"] == 1.5

def test_build_sequence(tmp_path):
    seq_def = {
        "sequence": [
            {"monomer": "ALA", "type": "PEPTIDE"},
            {"monomer": "PRO", "type": "PEPTIDE"}
        ]
    }
    p = tmp_path / "sequence.yaml"
    with open(p, "w") as f:
        yaml.dump(seq_def, f)

    mol = build_sequence(seq_def)
    assert isinstance(mol, Chem.Mol)
    ala_mol = Chem.AddHs(Chem.MolFromSmiles("CC(C(=O)O)N"))
    assert mol.GetNumAtoms() > ala_mol.GetNumAtoms()
