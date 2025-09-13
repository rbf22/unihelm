"""
UniHelm Energy Calculator
-------------------------

This module provides functions to calculate the potential energy of a molecule
using standard molecular mechanics force fields available in RDKit.
"""

from rdkit import Chem
from rdkit.Chem import AllChem

def calculate_total_energy(mol):
    """
    Calculates the total potential energy of a molecule using the UFF force field.
    """
    if not mol or mol.GetNumAtoms() == 0:
        return float('inf')
    if mol.GetNumConformers() == 0:
        return float('inf')

    try:
        ff = AllChem.UFFGetMoleculeForceField(mol)
        return ff.CalcEnergy()
    except Exception:
        return float('inf')

def calculate_nonbonded_energy(mol):
    """
    Calculates only the non-bonded (van der Waals and electrostatic) energy
    of a molecule using the UFF force field.
    """
    if not mol or mol.GetNumAtoms() == 0:
        return float('inf')
    if mol.GetNumConformers() == 0:
        return float('inf')

    try:
        ff = AllChem.UFFGetMoleculeForceField(mol)
        ff.Initialize() # Important before getting contributors
        contribs = ff.GetContribs()

        non_bonded_energy = 0.0
        for name, energy in contribs:
            if name.lower() in ["vdw", "ele"]:
                non_bonded_energy += energy

        return non_bonded_energy
    except Exception:
        return float('inf')

if __name__ == '__main__':
    # Example usage
    smiles = "CCO" # Ethanol
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.UFFOptimizeMolecule(mol)

    total_e = calculate_total_energy(mol)
    nonbonded_e = calculate_nonbonded_energy(mol)

    print(f"Ethanol Total UFF Energy: {total_e:.4f} kcal/mol")
    print(f"Ethanol Non-Bonded UFF Energy: {nonbonded_e:.4f} kcal/mol")
