# from globals import GlobalVariables
from .ml import Mott_Littleton
from .cell import Cell
from .atom import Atom
from . import input_parser
from collections import Counter
import subprocess
import os
from . import monitor

def write_energy_input() -> 'None':
    ml = Mott_Littleton()
    ml.initialise()
    cell = Cell()
    cell.get_cell()
    a = Atom()
    atoms = a.get_atoms()
    options = ml.options
    lib_file = ml.lib_file
    with open('master.gin', 'w') as f:
        f.write('opti conp nosymm \n')
        f.write('cell\n')
        f.write(str(cell))
        f.write('fractional 1 \n')
        for atom in atoms:
            f.write(str(atom))
        # f.write('lib ' + gbi.libfile + '\n')
        for line in options:
            f.write(line + '\n')
        f.write('')
    with open("master.gin", 'a') as f:
        with open (lib_file, 'r') as libf:
            content = libf.read()
            f.write(content)




def write_defect_dummy() -> 'None':
    ml = Mott_Littleton()
    ml.initialise()
    cell = Cell()
    cell.get_cell_after()
    a = Atom()
    atoms = a.get_atoms_after()
    options = ml.options
    r1, r2 = ml.cutoff

    with open("defect.gin", 'w') as f:
        f.write('defect nodsymm energy regi_before \n')
        f.write('cell\n')
        f.write(str(cell))
        f.write('fractional 1 \n')
        for atom in atoms:
            f.write(str(atom))
        # Write dummy defects
        f.write('size ')
        f.write(str(r1) + ' ')
        f.write(str(r2) + '\n')
        # Analyse all defect species and determine the center
        center, type, atom1, atom2  = input_parser.get_defect_center_from_source(ml.control_file)
        f.write(f'center {str(" ".join(map(str, center)))}\n')
        f.write(f'{type} {atom1} {str(" ".join(map(str, center)))}\n')

        # f.write('impurity Rn 0 0 0 \n')
        # f.write('centre 0 0 0 \n')
        
        for line in options:
            f.write(line)
    with open("defect.gin", 'a') as f:
        with open(ml.lib_file, 'r') as flib:
            content = flib.read()
            f.write(content)

def gulp_submit():
    subprocess.run(['sbatch', 'gulp.sh'])

