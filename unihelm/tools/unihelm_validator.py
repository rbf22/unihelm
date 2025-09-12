import os, sys, yaml
from rdkit import Chem

VALID_POLYMER_TYPES = ["PEPTIDE", "RNA", "DNA", "CHEM", "POLYKETIDE", "CARBOHYDRATE"]

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Load standard connections
def load_standard_connections():
    base_dir = os.path.dirname(__file__)
    std_path = os.path.abspath(os.path.join(base_dir, "../connections/standard_connections.yaml"))
    if os.path.exists(std_path):
        data = load_yaml(std_path)
        return data.get("standard_connections", {})
    return {}

STANDARD_CONNECTIONS = load_standard_connections()

def merge_standard_connection(conn):
    if "use_standard" in conn:
        std_id = conn["use_standard"]
        if std_id not in STANDARD_CONNECTIONS:
            raise ValueError(f"Unknown standard connection '{std_id}'")
        merged = dict(conn)
        for key, val in STANDARD_CONNECTIONS[std_id].items():
            if key not in merged:
                merged[key] = val
        return merged
    return conn

def allowed_external(name):
    return name.endswith("_prev") or name.endswith("_next") or name in ("P_next", "O3_prev")

def get_atom_names(data, mol):
    if isinstance(data.get("atom_names"), dict):
        names = list(data["atom_names"].values())
        if len(set(names)) != len(names):
            raise ValueError("Duplicate atom names detected in atom_names.")
        return set(names)
    else:
        return {atom.GetSymbol() + str(i+1) for i, atom in enumerate(mol.GetAtoms())}

def check_connections(data, atom_names, file_id):
    for conn in (data.get("connections") or []):
        conn = merge_standard_connection(conn)
        c_atom = conn["connect_atom"]
        if c_atom != "*" and c_atom not in atom_names:
            raise ValueError(f"{file_id}: connect_atom '{c_atom}' not found in atom_names.")

        # Check geometry atoms in standard templates
        for geom_key in ("bond", "angle", "dihedral"):
            if geom_key in conn:
                atom_list = []
                if geom_key == "bond":
                    atom_list = [conn[geom_key]["partner_atom"]]
                else:
                    atom_list = conn[geom_key]["atoms"]
                for name in atom_list:
                    if allowed_external(name):
                        continue
                    if name not in atom_names:
                        raise ValueError(f"{file_id}: Connection '{conn['id']}' {geom_key} refs unknown atom '{name}'.")

def validate_yaml(path):
    data = load_yaml(path)
    file_id = data.get("monomer_id", os.path.basename(path))
    if data.get("format") != "UniHelm YAML":
        raise ValueError(f"{file_id}: format must be 'UniHelm YAML'.")
    if str(data.get("version")) != "1.0":
        raise ValueError(f"{file_id}: version must be 1.0")
    if data["polymer_type"] not in VALID_POLYMER_TYPES:
        raise ValueError(f"{file_id}: Invalid polymer_type '{data['polymer_type']}'")

    mol = Chem.MolFromSmiles(data["smiles"])
    if mol is None:
        raise ValueError(f"{file_id}: SMILES could not be parsed.")

    atom_names = get_atom_names(data, mol)
    check_connections(data, atom_names, file_id)
    print(f"[OK] {file_id}")

if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "./unihelm/monomers"
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".yaml"):
                path = os.path.join(root, file)
                try:
                    validate_yaml(path)
                except Exception as e:
                    print(f"[FAIL] {path}: {e}")