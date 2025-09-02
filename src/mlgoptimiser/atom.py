# from globals import get_global_variables, get_species_data
from .ml import Mott_Littleton
import os
from pathlib import Path
import copy
class Atom:
    def __init__(self) -> None:
        self.label = None
        self.type = None       
        self.x = None
        self.y = None
        self.z = None
        self.q = None

    def copy(self):
        return copy.deepcopy(self)


    def has_shel(self) -> 'bool':
        ml = Mott_Littleton()
        ml.initialise()
        filename = ml.lib_file
        species =[]

        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('species'):
                    natom = int(line.strip().split()[-1])
                    for _ in range(natom):
                        line = f.readline()
                        parts = line.strip().split()
                        if len(parts) > 1 and parts[1] == 'shel':
                            species.append(parts)

        for line in species:
            if self.label == line[0].strip() and line[1].strip() == 'shel':
                return True
        return False

    def print_atom(self):
        print(f" {self.label} {self.type}  {self.x} {self.y} {self.z}")

    def get_atoms_after(self):
        atom_list = []
        filename = os.path.join(os.getcwd(), 'input', 'master.gout')
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('  Final fractional coordinates of atoms :'):
                    for i in range(5):
                        line = f.readline()
                    for line in f:
                        if line.startswith('--'):
                            break
                        parts = line.strip().split()
                        atom = Atom()
                        atom.label = parts[1]
                        atom.type = parts[2]
                        atom.x = float(parts[3])
                        atom.y = float(parts[4])
                        atom.z = float(parts[5])
                        atom_list.append(atom)
        return atom_list

    def get_atoms(self):
        atom_list = []
        ml = Mott_Littleton()
        filename = ml.config_file
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("BEGIN_ATOMS_FRAC"):
                    for line in f:
                        if line.startswith("END_ATOMS_FRAC"):
                            break
                        parts = line.strip().split()
                        atom = Atom()  # Create a new Atom instance
                        atom.label = parts[0]
                        atom.type = parts[1]
                        atom.x = float(parts[2])
                        atom.y = float(parts[3])
                        atom.z = float(parts[4])
                        atom_list.append(atom)  # Append the Atom instance to the list
        return atom_list
    
    def get_r1_after(self):
        r1_list = []
        cwd = os.getcwd()
        cwd = os.getcwd()  # Get current working directory
        input_dir = Path(cwd) / 'input'  # Set input directory path
        res_file = next((file for file in input_dir.iterdir() if file.suffix == '.res'), None)        
        with open(res_file, 'r') as f:
            for line in f:
                if line.startswith('region_1'):
                    for line in f:
                        if line.startswith('spe'):
                            break
                        elif line.startswith('dlist'):
                            break
                        parts = line.strip().split()
                        atom = Atom()
                        atom.label = parts[0]
                        atom.type = parts[1]
                        atom.x = float(parts[2])
                        atom.y = float(parts[3])
                        atom.z = float(parts[4])
                        atom.q = float(parts[5])
                        r1_list.append(atom)
        return r1_list
    def assign_charge(self):
        ml = Mott_Littleton()
        ml.initialise()
        species = ml.species

        for line in species:
            if self.label == line[0] and self.type[0] == line[1][0]:
                self.q = float(line[2])
                
        return self

    def __str__(self) -> str:
        return f" {self.label} {self.type}  {self.x} {self.y} {self.z} {self.q}\n"