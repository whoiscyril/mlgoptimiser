from .ml import Mott_Littleton
import numpy as np
from .atom import Atom
import random
def insert_defects(old_atoms, defect_list, algorithm, grid = None):
    """Function that takes in an old list of region 1 atoms, put in interstitial and impurities
       and return a new list of atoms, specifict to the algorithm chosen
    """
    ml = Mott_Littleton()
    ml.initialise()
    r1, r2 = ml.cutoff
    center = ml.defect_center
    impurity_list = [] 
    interstitial_list = []
    for d in defect_list:
        if d.type == 'interstitial':
            interstitial_list.append(d)
        elif d.type == 'impurity':
            impurity_list.append(d)

    ninter = len(interstitial_list)
    nimpure = len(impurity_list)
    # First remove all shells and identify how many impurity and interstitials are there in old_atoms
    core_atoms = []
    for atom in old_atoms:
        if atom.type == 'cor':
            atom.q = 0.
            core_atoms.append(atom)
    # As there is already one impurity substituted in region 1, pop the corresponding defect from defect list so that only defects left to do remains. ONLY POP 1 Line from defect list
    for atom in core_atoms:
        for d in defect_list[:]:
            if d.type == 'impurity' and d.label == atom.label:
                defect_list.remove(d)
                break



    # handling impurity 
    impurity_defects = [d for d in defect_list if d.type == 'impurity']
    if nimpure > 1:
        len_imp = sum(1 for d in defect_list if d.type == 'impurity')  # Count impurities

        matched_atoms = []
        for d in impurity_defects:
            for atom in core_atoms:
                distance = np.linalg.norm(center - np.array([atom.x, atom.y, atom.z]))
                if atom.label == d.atom2 and distance < r1:
                    matched_atoms.append(atom)
            if matched_atoms:  # Ensure the list is not empty before choosing
                to_replace = np.random.choice(matched_atoms)
                to_replace.label = d.label  # Replace atom's label

    else:
        pass

    # handling interstitials:
    interstitial_defects = [d for d in defect_list if d.type == 'interstitial']
    # print(len(interstitial_defects))
    for d in interstitial_defects:
        new_atom = Atom()
        xyz = random.choice(grid)
        new_atom.x, new_atom.y, new_atom.z = xyz
        new_atom.label = d.label
        new_atom.type = 'cor'
        core_atoms.append(new_atom)

    # After inserting defects, append shells and assign charges

    atoms_w_shels = core_atoms.copy()

    for atom in core_atoms:
        if atom.has_shel():
            a = Atom()
            a.label = atom.label
            a.type = 'shel'
            a.x, a.y, a.z = atom.x, atom.y, atom.z
            atoms_w_shels.append(a)

    for atom in atoms_w_shels:
        atom.assign_charge()

    return atoms_w_shels


        
                

            




        


    