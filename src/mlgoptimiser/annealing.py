from .cell import Cell
from .atom import Atom
from . import input_parser
from .defect import Defect
import numpy as np
from . import atom_math
from .globals import get_global_variables
from .optimiser import MC
import os
import subprocess
from . import monte_carlo_util
import random
from . import move_class
from . import monte_carlo_util
def move_defects(r1_list, filename, step_size) -> list:
    dcentre = input_parser.get_dcenter_from_out(filename)
    ri, r1, r2 = input_parser.get_cutoff()
    new_r1 = []
    dlist = input_parser.get_defect_list()

    r1_wo_shell = []
    # Pop shell 
    for atom in r1_list:
        if atom.type != 'shel':
            r1_wo_shell.append(atom)
    

    unique_defect = set()
    for d in dlist:
        new_defect = Defect()
        new_defect = d
        unique_defect.add(new_defect)
    for atom in r1_wo_shell:
        added = False  # Flag to track if the atom is added
        # Move atoms according to move_cyril
        xyz = np.array([atom.x, atom.y, atom.z]) 
        if np.linalg.norm(xyz - dcentre) < 0.1:
            new_atom = atom
        else:
            new_atom = monte_carlo_util.moveclass_cyril(atom, step_size, r1, np.linalg.norm(xyz - dcentre), 1.)
        for d in unique_defect:
            if atom.label == d.label and d.type != 'impurity':  # Condition to avoid defect center
                laccept = False
                while not laccept:
                    atom_n = monte_carlo_util.move_atom(atom, step_size)
                    newxyz = np.array([atom_n.x, atom_n.y, atom_n.z])
                    if np.linalg.norm(newxyz - dcentre) < r1:
                        laccept = True
                        new_r1.append(atom_n)
                        added = True  # Mark as added
                        break  # Break the loop after successful addition
        if not added:  # Append only if the atom was not added in the loop
            new_r1.append(new_atom)
    result = new_r1.copy()
    # put shells on
    for atom in new_r1:
        if atom.has_shel():
            new_atom = atom.copy()
            new_atom.type = 'shel'
            new_atom.q = 0.
            new_atom.assign_charge()
            result.append(new_atom)
    return result, dcentre

def move_cyril(r1_list, filename, step_size):
    dcentre = input_parser.get_dcenter_from_out(filename)
    ri, r1, r2 = input_parser.get_cutoff()
    new_r1 = []
    dlist = input_parser.get_defect_list()

    r1_wo_shell = []
    # Pop shell 
    for atom in r1_list:
        if atom.type != 's':
            r1_wo_shell.append(atom)
    unique_defect = set()
    for d in dlist:
        unique_defect.add(d.label)
    for atom in r1_wo_shell:
        xyz = np.array([atom.x, atom.y, atom.z])
        distance = np.linalg.norm(xyz - dcentre)
        if atom.label in unique_defect:
            laccept = False
            while not laccept:
                atom_n = monte_carlo_util.moveclass_cyril(atom, step_size, r1, distance, 3.0)
                newxyz = np.array([atom_n.x, atom_n.y, atom_n.z])
                if np.linalg.norm(newxyz - dcentre) < r1:
                    laccept = True
                    new_r1.append(atom_n)
        else:
            new_r1.append(atom)
        
    result = new_r1.copy()
    # put shells on
    for atom in new_r1:
        if atom.has_shel():
            new_atom = atom.copy()
            new_atom.type = 'shel'
            new_atom.q = 0.
            new_atom.assign_charge()
            result.append(new_atom)
    return  result, dcentre

def metropolis(delta_en, temp):
    kb = 8.617*10**(-5)
    rn = np.random.uniform(0, 1)
    p = np.exp(-delta_en/(temp *kb))
    if p > rn:
        return True
    return False



def annealing_generatorF():    
    gbi = get_global_variables()
    # First initiate all global constants
    a = Atom()
    cell = Cell()
    r_list = a.get_r1_after()
    ri, r1, r2 = input_parser.get_cutoff()
    dcentre = np.array(input_parser.get_defect_centre())
    r1_wo_shell = []
    for atom in r_list:
        if atom.type != 'she':
            r1_wo_shell.append(atom)
    for atom in r1_wo_shell:
        atom.q = 0.
    
    dlist = input_parser.get_defect_list()
    vac_count, sub_count, inter_count = input_parser.get_defect_count(gbi.infile)

    impurity_list = []
    for d in dlist:
        if d.type =='impurity':
            impurity_list.append((d.label, d.atom2))
    r1_w_impurity = r1_wo_shell.copy()
    # for impurities, simply check if there is enough impurities substituted, if so, terminate
    for impure_atom, old_atom in impurity_list:
        while(input_parser.check_atom_count(r1_w_impurity, impure_atom) != sub_count):
            indices = []
            for index, atom in enumerate(r1_wo_shell):
                if atom.label == old_atom:
                    indices.append(index)
            if indices:
                chosen_index = np.random.choice(indices)
                for _index, _atom in enumerate(r1_w_impurity):
                    if _index == chosen_index:
                        _atom.label = impure_atom

    inter_list = []
    r1_w_inter = r1_w_impurity.copy()

    for d in dlist:
        if d.type == 'interstitial':
            inter = Atom()
            inter.label = d.label
            inter.type = 'cor'
            while True:
                random_point = np.random.uniform(-r1, r1, 3)
                inter.x, inter.y, inter.z = random_point  # Add defect centre
                if np.linalg.norm(random_point - dcentre) <= r1 and atom_math.geo_checker(inter, r1_w_impurity, 0.1):
                # if np.linalg.norm(random_point - dcentre) <= r1:
                    break
            inter.q = 0.
            inter_list.append(inter)
    
    r1_w_inter.extend(inter_list)


    # put shells on and assign charges
    result_wcns = []
    for atom in r1_w_inter:
        atom.assign_charge()
        result_wcns.append(atom)
        if atom.has_shel():
            new_atom = atom.copy()
            new_atom.q = 0.
            new_atom.type = 'shel'
            new_atom.assign_charge()
            result_wcns.append(new_atom)
        

    
    return result_wcns, dcentre
def annealing_generator():    
    gbi = get_global_variables()
    # First initiate all global constants
    a = Atom()
    cell = Cell()
    r_list = a.get_r1_after()
    ri, r1, r2 = input_parser.get_cutoff()
    dcentre = np.array(input_parser.get_defect_centre())
    r1_wo_shell = []
    for atom in r_list:
        if atom.type != 'she':
            r1_wo_shell.append(atom)
    for atom in r1_wo_shell:
        atom.q = 0.
    
    dlist = input_parser.get_defect_list()
    vac_count, sub_count, inter_count = input_parser.get_defect_count(gbi.infile)

    impurity_list = []
    for d in dlist:
        if d.type =='impurity':
            impurity_list.append((d.label, d.atom2))
    r1_w_impurity = r1_wo_shell.copy()
    # for impurities, simply check if there is enough impurities substituted, if so, terminate
    for impure_atom, old_atom in impurity_list:
        while(input_parser.check_atom_count(r1_w_impurity, impure_atom) != sub_count):
            indices = []
            for index, atom in enumerate(r1_wo_shell):
                if atom.label == old_atom:
                    indices.append(index)
            if indices:
                chosen_index = np.random.choice(indices)
                for _index, _atom in enumerate(r1_w_impurity):
                    if _index == chosen_index:
                        _atom.label = impure_atom

    inter_list = []
    r1_w_inter = r1_w_impurity.copy()

    for d in dlist:
        if d.type == 'interstitial':
            inter = Atom()
            inter.label = d.label
            inter.type = 'cor'
            while True:
                random_point = np.random.uniform(-r1, r1, 3)
                inter.x, inter.y, inter.z = random_point  # Add defect centre
                if np.linalg.norm(random_point - dcentre) <= r1 and atom_math.geo_checker(inter, r1_w_impurity, 0.1):
                # if np.linalg.norm(random_point - dcentre) <= r1:
                    break
            inter.q = 0.
            inter_list.append(inter)
    
    r1_w_inter.extend(inter_list)

    # Now have r1 list with interstitial and impurity, randomly place them in within r1_cutoff
    result = [] 
    for atom in r1_w_inter:
        if not atom_math.at_center(atom, dcentre):
            new_atom = move_class.pos_random(atom, dcentre, r1)
            result.append(new_atom)
        else:
            new_atom = move_class.fix_dcenter_atom(atom, dcentre)
            result.append(new_atom)
    # result_wcns = result.copy()
    # put shells on and assign charges
    result_wcns = []
    for atom in result:
        atom.assign_charge()
        result_wcns.append(atom)
        if atom.has_shel():
            new_atom = atom.copy()
            new_atom.q = 0.
            new_atom.type = 'shel'
            new_atom.assign_charge()
            result_wcns.append(new_atom)
        

    
    return result_wcns, dcentre

def write_input(fname, atoms, dcentre, ex_opt='') -> None:
    gbi = get_global_variables()
    a = Atom()
    atom_list = a.get_atoms_after()
    # dcentre = np.array(input_parser.get_defect_centre())
    cell = Cell()
    ri, r1, r2 = input_parser.get_cutoff()
    dcentre_string = ' '.join(map(str, dcentre.flatten()))
    options = input_parser.get_options()
    ocell = cell.get_cell_after()

    # Collect all lines in a list
    lines = []
    lines.append('single defect \n')
    # lines.append('defect nodsymm energy \n')
    lines.append('cell \n')
    lines.append(str(cell))
    lines.append('fractional 1 \n')
    lines.extend([str(atom) for atom in atom_list])
    lines.append('center cart ' + dcentre_string + '\n')
    lines.append(f'size {r1} {r2} \n')
    lines.append('region_1 \n')
    lines.extend([str(atom) for atom in atoms])
    # lines.append('lib ' + gbi.libfile + '\n')  # Uncomment if needed
    lines.extend(options)
    lines.append(ex_opt + '\n')
    with open (gbi.libfile, 'r') as flib:
        content = flib.read()
    lines.extend(content)

    # Write all collected lines to the file at once
    with open(fname, 'w') as f:
        f.writelines(lines)

def write_quench(fname, atoms, dcentre, ex_opt='') -> None:
    gbi = get_global_variables()
    a = Atom()
    atom_list = a.get_atoms_after()
    # dcentre = np.array(input_parser.get_defect_centre())
    cell = Cell()
    ri, r1, r2 = input_parser.get_cutoff()
    dcentre_string = ' '.join(map(str, dcentre.flatten()))
    options = input_parser.get_options()
    ocell = cell.get_cell_after()

    # Collect all lines in a list
    lines = []
    lines.append('opti conp defect \n')
    # lines.append('defect nodsymm energy \n')
    lines.append('cell \n')
    lines.append(str(cell))
    lines.append('fractional 1 \n')
    lines.extend([str(atom) for atom in atom_list])
    lines.append('center cart ' + dcentre_string + '\n')
    lines.append(f'size {r1} {r2} \n')
    lines.append('region_1 \n')
    lines.extend([str(atom) for atom in atoms])
    # lines.append('lib ' + gbi.libfile + '\n')  # Uncomment if needed
    lines.extend(options)
    lines.append(ex_opt + '\n')
    with open (gbi.libfile, 'r') as flib:
        content = flib.read()
    lines.extend(content)

    # Write all collected lines to the file at once
    with open(fname, 'w') as f:
        f.writelines(lines)

