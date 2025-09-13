"""
UniHelm Monte Carlo Sampler (v2)
--------------------------------

This tool performs a Monte Carlo (MC) conformational search on a single monomer
by sampling from a pre-computed rotamer library. It uses the v2.0 data format.
"""

import random
import math
import os
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

# Adjust import paths for modular execution
try:
    from . import rotamer_loader
    from . import ic_builder
    from . import energy
except ImportError:
    # Allow running as a standalone script
    import rotamer_loader
    import ic_builder
    import energy

from . import polymer_builder

def run_mc_sampler(sequence_defs, n_steps, temperature, output_pdb):
    """
    Runs a Monte Carlo simulation on a polymer chain.
    """
    print(f"--- Starting Monte Carlo Sampler for sequence: {[s['monomer_id'] for s in sequence_defs]} ---")
    print(f"Parameters: {n_steps} steps, Temperature: {temperature}K")

    # 1. Load all monomer data
    all_monomer_data = {
        s['monomer_id']: rotamer_loader.load_monomer_data(s['monomer_id'], s['type'])
        for s in sequence_defs
    }

    # 2. Initial State: build with the first rotamer for each monomer
    print("Building initial conformation...")
    current_rotamer_ids = [all_monomer_data[s['monomer_id']]['rotamers'][0]['id'] for s in sequence_defs]

    sequence_with_rotamers = []
    for i, s in enumerate(sequence_defs):
        monomer_data = all_monomer_data[s['monomer_id']]
        rotamer_data = rotamer_loader.get_rotamer_by_id(monomer_data, current_rotamer_ids[i])
        sequence_with_rotamers.append({
            "monomer_data": monomer_data,
            "rotamer_data": rotamer_data
        })

    current_mol = polymer_builder.build_polymer(sequence_with_rotamers)

    # The polymer builder creates a correct topology but a nonsensical geometry.
    # We must optimize it to get a valid starting point.
    if current_mol:
        try:
            Chem.SanitizeMol(current_mol)
            AllChem.UFFOptimizeMolecule(current_mol, maxIters=2000)
        except:
            current_mol = None

    current_energy = energy.calculate_nonbonded_energy(current_mol)
    print(f"Initial non-bonded energy: {current_energy:.4f} kcal/mol")

    accepted_conformations = [current_mol]
    kT = 0.001987 * temperature # Boltzmann constant in kcal/mol*K

    # 3. Monte Carlo Loop
    print("\nStarting MC simulation...")
    n_accepted = 0
    for step in range(n_steps):
        # a. Propose a move: pick a monomer and a new rotamer for it
        mut_idx = random.randrange(len(sequence_defs))
        mut_monomer_id = sequence_defs[mut_idx]['monomer_id']
        mut_monomer_data = all_monomer_data[mut_monomer_id]

        new_rotamer = random.choice(mut_monomer_data['rotamers'])

        # Create the proposed list of rotamer IDs
        proposed_rotamer_ids = list(current_rotamer_ids)
        proposed_rotamer_ids[mut_idx] = new_rotamer['id']

        # b. Build the new conformation
        proposed_sequence_with_rotamers = []
        for i, s in enumerate(sequence_defs):
            monomer_data = all_monomer_data[s['monomer_id']]
            rotamer_data = rotamer_loader.get_rotamer_by_id(monomer_data, proposed_rotamer_ids[i])
            proposed_sequence_with_rotamers.append({
                "monomer_data": monomer_data,
                "rotamer_data": rotamer_data
            })

        proposed_mol = polymer_builder.build_polymer(proposed_sequence_with_rotamers)
        if proposed_mol:
            try:
                Chem.SanitizeMol(proposed_mol)
                AllChem.UFFOptimizeMolecule(proposed_mol, maxIters=2000)
            except:
                proposed_mol = None

        # c. Calculate energy and accept/reject
        proposed_energy = energy.calculate_nonbonded_energy(proposed_mol)

        if proposed_energy < float('inf'):
            delta_e = proposed_energy - current_energy
            if delta_e < 0 or random.random() < math.exp(-delta_e / kT):
                # Accept move
                n_accepted += 1
                current_mol = proposed_mol
                current_energy = proposed_energy
                current_rotamer_ids = proposed_rotamer_ids
                accepted_conformations.append(current_mol)
                if (step+1) % 10 == 0:
                     print(f"Step {step+1}/{n_steps}: Accepted new state (E = {current_energy:.4f})")

    print(f"\n--- Simulation Finished ---")
    print(f"Acceptance rate: {n_accepted / n_steps * 100:.2f}%")

    # 4. Save results
    if accepted_conformations:
        print(f"Saving {len(accepted_conformations)} conformations to {output_pdb}")
        writer = Chem.PDBWriter(output_pdb)
        for i, mol in enumerate(accepted_conformations):
            if mol: writer.write(mol, confId=i)
        writer.close()

if __name__ == '__main__':
    # --- Parameters ---
    SEQUENCE = [
        {'monomer_id': 'ALA', 'type': 'PEPTIDE'},
        {'monomer_id': 'GLY', 'type': 'PEPTIDE'}
    ]
    STEPS = 100
    TEMPERATURE = 300 # Kelvin
    OUTPUT_FILE = "polymer_sampler_output.pdb"

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    run_mc_sampler(SEQUENCE, STEPS, TEMPERATURE, OUTPUT_FILE)
    print(f"\nSampler finished. Check '{OUTPUT_FILE}' for results.")
