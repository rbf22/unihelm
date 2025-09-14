import pytest
import yaml
from rdkit import Chem
import os
from unihelm.tools.conformation_generator import (
    generate_conformation_library,
    get_atom_name_map,
)

@pytest.fixture
def alanine_smiles_mapped():
    return "[N:1][C@@H](C)[C:2](=O)O"

def test_deterministic_atom_naming():
    """
    Tests if canonicalization + naming scheme produces deterministic names.
    """
    mol1 = Chem.MolFromSmiles("C(C)N") # Non-canonical
    mol2 = Chem.MolFromSmiles("NCC")  # Canonical

    # Manually canonicalize and add hydrogens for testing
    canon_mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol1))
    canon_mol1 = Chem.AddHs(canon_mol1)
    canon_mol2 = Chem.MolFromSmiles(Chem.MolToSmiles(mol2))
    canon_mol2 = Chem.AddHs(canon_mol2)

    name_map1 = get_atom_name_map(canon_mol1)
    name_map2 = get_atom_name_map(canon_mol2)

    assert name_map1 == name_map2
    # Expected names after canonicalization (NCC) and H-addition
    # This is complex to predict exactly without running it,
    # so we focus on consistency between two different SMILES for the same molecule.

def test_connection_generation(tmp_path, alanine_smiles_mapped):
    """
    Tests that connection points are correctly generated from a mapped SMILES.
    """
    monomer_id = "ALA_TEST"

    # Run the generator, writing to the temp directory
    generate_conformation_library(
        monomer_id,
        "PEPTIDE",
        alanine_smiles_mapped,
        num_confs=10, # Use fewer confs for speed
        output_dir=str(tmp_path)
    )

    # Check the output file
    generated_file = tmp_path / f"{monomer_id}.yaml"
    assert os.path.exists(generated_file)

    with open(generated_file, "r") as f:
        data = yaml.safe_load(f)

    connections = data["connections"]
    assert len(connections) == 2

    r1 = next(c for c in connections if c['id'] == 'R1')
    r2 = next(c for c in connections if c['id'] == 'R2')

    # Based on "[N:1][C@@H](C)[C:2](=O)O"
    # Canonical SMILES is likely C[C@H](N)C(=O)O
    # C1, C2, H1, N1, C3... the names depend on canonical order.
    # But we can check that the connect_atom is a Nitrogen for R1 and Carbon for R2.
    atom_names = data["atom_names"]

    connect_atom_r1_idx = atom_names[r1['connect_atom']]['idx']
    connect_atom_r2_idx = atom_names[r2['connect_atom']]['idx']

    mol = Chem.MolFromSmiles(data['smiles'])
    mol = Chem.AddHs(mol)

    assert mol.GetAtomWithIdx(connect_atom_r1_idx).GetSymbol() == 'N'
    assert mol.GetAtomWithIdx(connect_atom_r2_idx).GetSymbol() == 'C'

    # No need to clean up, tmp_path handles it

def test_full_generation_output(tmp_path, alanine_smiles_mapped):
    """
    Tests the overall structure of the generated YAML file.
    """
    monomer_id = "ALA_TEST_2"

    generate_conformation_library(
        monomer_id,
        "PEPTIDE",
        alanine_smiles_mapped,
        num_confs=10,
        output_dir=str(tmp_path)
    )

    generated_file = tmp_path / f"{monomer_id}.yaml"
    assert os.path.exists(generated_file)

    with open(generated_file, 'r') as f:
        data = yaml.safe_load(f)

    assert data["format"] == "UniHelm YAML"
    assert data["version"] == "2.3"
    assert data["monomer_id"] == monomer_id
    assert "smiles" in data
    assert "atom_names" in data

    # Check the new atom_names structure
    first_atom_name = list(data["atom_names"].keys())[0]
    first_atom_data = data["atom_names"][first_atom_name]
    assert isinstance(first_atom_data, dict)
    assert "idx" in first_atom_data
    assert "mass" in first_atom_data
    assert "charge" in first_atom_data

    assert "connections" in data
    assert "geometry" in data
    assert "conformations" in data
    assert len(data["conformations"]) > 0
