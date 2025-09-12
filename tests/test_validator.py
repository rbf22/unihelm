import pytest
import os
import yaml
from rdkit import Chem
from unihelm.tools.unihelm_validator import validate_yaml, load_yaml, get_atom_names, check_connections

def test_load_yaml(tmp_path):
    p = tmp_path / "test.yaml"
    p.write_text("key: value")
    data = load_yaml(str(p))
    assert data == {"key": "value"}

def test_get_atom_names_from_map():
    data = {"atom_names": {"1": "N", "2": "CA", "3": "C"}}
    mol = Chem.MolFromSmiles("NCC")
    names = get_atom_names(data, mol)
    assert names == {"N", "CA", "C"}

def test_get_atom_names_generated():
    data = {}
    mol = Chem.MolFromSmiles("CCO")
    names = get_atom_names(data, mol)
    assert names == {"C1", "C2", "O3"}

def test_check_connections_valid():
    data = {
        "connections": [{"connect_atom": "C1"}],
    }
    atom_names = {"C1", "O2"}
    check_connections(data, atom_names, "test_id") # Should not raise

def test_check_connections_invalid():
    data = {
        "connections": [{"connect_atom": "C2"}],
    }
    atom_names = {"C1", "O2"}
    with pytest.raises(ValueError, match="connect_atom 'C2' not found"):
        check_connections(data, atom_names, "test_id")

def test_validate_yaml_valid(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "valid_monomer.yaml"
    content = """
format: "UniHelm YAML"
version: "1.0"
monomer_id: "test"
polymer_type: "PEPTIDE"
smiles: "C"
    """
    p.write_text(content)

    validate_yaml(str(p)) # Should not raise an exception

def test_validate_yaml_invalid(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "invalid_monomer.yaml"
    content = """
format: "UniHelm YAML"
version: "1.0"
monomer_id: "test"
polymer_type: "INVALID_TYPE"
smiles: "C"
    """
    p.write_text(content)

    with pytest.raises(ValueError, match="Invalid polymer_type"):
        validate_yaml(str(p))


def get_all_monomer_files():
    monomer_dir = "unihelm/monomers"
    all_files = []
    for root, _, files in os.walk(monomer_dir):
        for file in files:
            if file.endswith(".yaml"):
                all_files.append(os.path.join(root, file))
    return all_files

@pytest.mark.parametrize("monomer_file", get_all_monomer_files())
def test_all_monomers_are_valid(monomer_file):
    """
    Validates all monomer YAML files found in the project.
    """
    try:
        validate_yaml(monomer_file)
    except Exception as e:
        pytest.fail(f"Validation failed for {monomer_file}: {e}")
