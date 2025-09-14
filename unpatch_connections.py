import os
import yaml

PEPTIDE_DIR = "unihelm/monomers/PEPTIDE"

for filename in os.listdir(PEPTIDE_DIR):
    if filename.endswith(".yaml"):
        filepath = os.path.join(PEPTIDE_DIR, filename)
        print(f"Processing {filepath}...")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        if "semantic_map" in data:
            del data["semantic_map"]

        if "connections" in data:
            for conn in data["connections"]:
                if "use_standard" in conn:
                    del conn["use_standard"]
                if conn.get("id") == "R1":
                    conn["label"] = "R1"
                    # This is tricky, I need to find the original connect_atom
                    # I will just set it to a placeholder, the generator will fix it.
                    conn["connect_atom"] = "R1_placeholder"
                elif conn.get("id") == "R2":
                    conn["label"] = "R2"
                    conn["connect_atom"] = "R2_placeholder"

            with open(filepath, 'w') as f:
                yaml.dump(data, f, sort_keys=False, indent=2)
            print(f"  ...Unpatched connections.")
        else:
            print("  ...No connections to unpatch.")

print("\nDone unpatching all peptide connections.")
