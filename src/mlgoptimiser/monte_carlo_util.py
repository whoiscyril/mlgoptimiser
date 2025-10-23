# from globals import get_global_variables
# from optimiser import MC
import os
import random
import subprocess

import numpy as np

from . import atom_math, input_parser, monte_carlo_util
from .atom import Atom
from .cell import Cell
from .defect import Defect
from .ml import Mott_Littleton
from .logging_config import get_auto_logger


def moveclass_cyril(atom, stepsize, cutoff, distance, alpha):
    new_atom = atom.copy()
    new_atom.x = (
        new_atom.x
        + np.random.uniform(low=-1, high=1)
        * stepsize
        * (1 - distance / cutoff) ** alpha
    )
    new_atom.y = (
        new_atom.y
        + np.random.uniform(low=-1, high=1)
        * stepsize
        * (1 - distance / cutoff) ** alpha
    )
    new_atom.z = (
        new_atom.z
        + np.random.uniform(low=-1, high=1)
        * stepsize
        * (1 - distance / cutoff) ** alpha
    )
    return new_atom


def move_atom(atom, stepsize):
    new_atom = atom.copy()
    new_atom.x = new_atom.x + np.random.uniform(low=-1, high=1) * stepsize
    new_atom.y = new_atom.y + np.random.uniform(low=-1, high=1) * stepsize
    new_atom.z = new_atom.z + np.random.uniform(low=-1, high=1) * stepsize
    return new_atom


def is_too_close(a, others):
    return any(
        np.linalg.norm(np.array([a.x, a.y, a.z]) - np.array([b.x, b.y, b.z])) < 0.1
        for b in others
    )


def bh_move_cyril(r1_list, filename, step_size):
    dcentre = input_parser.get_dcenter_from_out(filename)
    ml = Mott_Littleton()
    ml.initialise()
    r1, r2 = ml.cutoff
    new_r1 = []
    dlist = input_parser.get_defect_list(ml.control_file)

    for atom in r1_list:
        pass

    r1_wo_shell = []
    # Pop shell
    for atom in r1_list:
        if atom.type != "s":
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
                atom_n = moveclass_cyril(atom, step_size, r1, distance, 1.0)
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
            new_atom.type = "shel"
            new_atom.q = 0.0
            new_atom.assign_charge()
            result.append(new_atom)
    return result, dcentre


def swap_impurity_with_neighbor(atom_list, imp_label, cutoff):
    """
    Find the first atom in atom_list whose label == imp_label;
    pick one random neighbor (label != imp_label) within `cutoff` Å;
    swap their coordinates and charges; return a new list.
    """
    # 1) Copy the list so we don’t mutate the original atoms
    new_list = [atom.copy() for atom in atom_list]

    # 2) Locate the impurity atom
    imp_idx = next((i for i, a in enumerate(new_list) if a.label == imp_label), None)
    if imp_idx is None:
        raise ValueError(f"No atom with label {imp_label}")
    imp = new_list[imp_idx]

    # 3) Build neighbor candidates (exclude shells, exclude itself)
    imp_pos = np.array([imp.x, imp.y, imp.z])
    neighbors = []
    for j, other in enumerate(new_list):
        if j == imp_idx or other.type == "s":
            continue
        other_pos = np.array([other.x, other.y, other.z])
        if np.linalg.norm(other_pos - imp_pos) <= cutoff:
            neighbors.append(j)

    if not neighbors:
        raise ValueError(f"No neighbors within {cutoff} Å of impurity at {imp_pos}")

    # 4) Pick a random neighbor index
    nbr_idx = random.choice(neighbors)
    nbr = new_list[nbr_idx]

    # logger.debug("Swapping impurity %s at %s with neighbor %s at %s",
    #             imp.label, imp_pos, nbr.label, [nbr.x,nbr.y,nbr.z])

    # 5) Swap positions (and charge q, and you could swap label if needed)
    imp.x, nbr.x = nbr.x, imp.x
    imp.y, nbr.y = nbr.y, imp.y
    imp.z, nbr.z = nbr.z, imp.z
    imp.label, nbr.label = nbr.label, imp.label
    imp.q, nbr.q = nbr.q, imp.q

    # 6) Return the modified list
    return new_list


def sa_bimodal(
    r1_list, filename, step_size, p_small=0.5, sigma_small=1.0, sigma_large=4.0
):
    """
    Perform one bimodal displacement on a randomly chosen Li atom:
      - Small jiggle (σ = sigma_small) with probability p_small
      - Large hop (σ = sigma_large) otherwise
    All other atoms remain fixed.
    """
    dcentre = input_parser.get_dcenter_from_out(filename)
    ml = Mott_Littleton()
    ml.initialise()
    r1, r2 = ml.cutoff
    new_r1 = []
    dlist = input_parser.get_defect_list(ml.control_file)

    r1_wo_shell = []
    interstitials = set()

    for atom in r1_list:
        if atom.type != "she":
            r1_wo_shell.append(atom)

    li_idxs = [i for i, at in enumerate(r1_wo_shell) if at.label == "Li"]
    if not li_idxs:
        return
    i = random.choice(li_idxs)
    old_atom = r1_wo_shell[i]

    while True:
        sigma = sigma_small if random.random() < p_small else sigma_large

        delta = np.random.normal(scale=sigma, size=3)

        new_pos = np.array([old_atom.x, old_atom.y, old_atom.z]) + delta
        if np.linalg.norm(new_pos - np.array(dcentre)) <= r1:
            old_atom.x, old_atom.y, old_atom.z = new_pos
            break

    result = r1_wo_shell.copy()
    for atom in r1_wo_shell:
        if atom.has_shel():
            new_atom = atom.copy()
            new_atom.type = "shel"
            new_atom.q = 0.0
            new_atom.assign_charge()
            result.append(new_atom)
    return result, dcentre


def sa_move(r1_list, filename, step_size):
    dcentre = input_parser.get_dcenter_from_out(filename)
    ml = Mott_Littleton()
    ml.initialise()
    r1, r2 = ml.cutoff
    new_r1 = []
    dlist = input_parser.get_defect_list(ml.control_file)

    r1_wo_shell = []
    interstitials = set()

    for atom in r1_list:
        if atom.type != "she":
            r1_wo_shell.append(atom)

    imp_defs = [d for d in dlist if d.type == "impurity"]

    label_to_atom2 = {d.label: d.atom2 for d in imp_defs}

    # # r1 before swap:
    # for atom in r1_wo_shell:
    #     logging.debug(atom)

    # Step 2: Build neighbour map
    neighbour_map = dict()
    for index, atom in enumerate(r1_wo_shell):
        if atom.label in label_to_atom2:
            atom2_label = label_to_atom2[atom.label]
            xyz_d = np.array([atom.x, atom.y, atom.z])
            neighbour_list = []
            for a in r1_wo_shell:
                if a.label == atom2_label:
                    xyz_o = np.array([a.x, a.y, a.z])
                    dist = np.linalg.norm(xyz_d - xyz_o)
                    if dist < 3.0:
                        neighbour_list.append(a)
            if neighbour_list:
                neighbour_map[index] = neighbour_list

    # Step 3: Randomly swap impurity with one of its neighbors
    for index, neighbours in neighbour_map.items():
        atom = r1_wo_shell[index]
        # if impurity site is not at defect center then carry out move class:
        atom_xyz = np.array([atom.x, atom.y, atom.z])
        if np.linalg.norm(atom_xyz - dcentre) < 0.1:
            continue
        if neighbours:
            swap_atom = random.choice(neighbours)

            # Find the index of the swap_atom
            swap_index = r1_wo_shell.index(swap_atom)

            # Save properties
            label1 = atom.label
            label2 = swap_atom.label
            charge1 = atom.q
            charge2 = swap_atom.q
            pos1 = (atom.x, atom.y, atom.z)
            pos2 = (swap_atom.x, swap_atom.y, swap_atom.z)

            # Swap properties
            # atom.x, atom.y, atom.z = pos2
            atom.label = label2
            atom.q = charge2

            # swap_atom.x, swap_atom.y, swap_atom.z = pos1
            swap_atom.label = label1
            swap_atom.q = charge1

            # Insert modified atoms back into list explicitly
            r1_wo_shell[index] = atom
            r1_wo_shell[swap_index] = swap_atom

    # # r1 after swap:
    # for atom in r1_wo_shell:
    #     logging.debug(atom)

    for d in dlist:
        if d.type == "interstitial" and d.label not in interstitials:
            interstitials.add(d.label)

    for atom in r1_wo_shell:
        xyz = np.array([atom.x, atom.y, atom.z])
        distance = np.linalg.norm(xyz - dcentre)
        if atom.label in interstitials:
            laccept = False
            max_attempts = 1000
            attempt = 0
            # Exclude the current atom from geometry check to avoid checking against itself
            r1_wo_shell_excluding_self = [a for a in r1_wo_shell if a is not atom]
            while not laccept:
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Could not find valid position for interstitial {atom.label} "
                        f"after {max_attempts} attempts. Try increasing step_size or relaxing geometry constraints."
                    )
                atom_n = moveclass_cyril(atom, step_size, r1, distance, 3.0)
                newxyz = np.array([atom_n.x, atom_n.y, atom_n.z])
                if np.linalg.norm(newxyz - dcentre) < r1 and atom_math.geo_checker(
                    atom_n, r1_wo_shell_excluding_self, 1.0
                ):
                    laccept = True
                    new_r1.append(atom_n)
                attempt += 1
        else:
            new_r1.append(atom)

    result = new_r1.copy()
    for atom in new_r1:
        if atom.has_shel():
            new_atom = atom.copy()
            new_atom.type = "shel"
            new_atom.q = 0.0
            new_atom.assign_charge()
            result.append(new_atom)
    return result, dcentre


def bh_move_inter_only(r1_list, filename, step_size):
    dcentre = input_parser.get_dcenter_from_out(filename)
    ml = Mott_Littleton()
    ml.initialise()
    r1, r2 = ml.cutoff
    new_r1 = []
    dlist = input_parser.get_defect_list(ml.control_file)

    r1_wo_shell = []
    interstitials = set()

    for atom in r1_list:
        if atom.type != "s":
            r1_wo_shell.append(atom)

    imp_defs = [d for d in dlist if d.type == "impurity"]

    label_to_atom2 = {d.label: d.atom2 for d in imp_defs}

    # # r1 before swap:
    # for atom in r1_wo_shell:
    #     logging.debug(atom)

    # Step 2: Build neighbour map
    neighbour_map = dict()
    for index, atom in enumerate(r1_wo_shell):
        if atom.label in label_to_atom2:
            atom2_label = label_to_atom2[atom.label]
            xyz_d = np.array([atom.x, atom.y, atom.z])
            neighbour_list = []
            for a in r1_wo_shell:
                if a.label == atom2_label:
                    xyz_o = np.array([a.x, a.y, a.z])
                    dist = np.linalg.norm(xyz_d - xyz_o)
                    if dist < 3.0:
                        neighbour_list.append(a)
            if neighbour_list:
                neighbour_map[index] = neighbour_list

    # Step 3: Randomly swap impurity with one of its neighbors
    for index, neighbours in neighbour_map.items():
        atom = r1_wo_shell[index]
        # if impurity site is not at defect center then carry out move class:
        atom_xyz = np.array([atom.x, atom.y, atom.z])
        if np.linalg.norm(atom_xyz - dcentre) < 0.1:
            continue
        if neighbours:
            swap_atom = random.choice(neighbours)

            # Find the index of the swap_atom
            swap_index = r1_wo_shell.index(swap_atom)

            # Save properties
            label1 = atom.label
            label2 = swap_atom.label
            charge1 = atom.q
            charge2 = swap_atom.q
            pos1 = (atom.x, atom.y, atom.z)
            pos2 = (swap_atom.x, swap_atom.y, swap_atom.z)

            # Swap properties
            # atom.x, atom.y, atom.z = pos2
            atom.label = label2
            atom.q = charge2

            # swap_atom.x, swap_atom.y, swap_atom.z = pos1
            swap_atom.label = label1
            swap_atom.q = charge1

            # Insert modified atoms back into list explicitly
            r1_wo_shell[index] = atom
            r1_wo_shell[swap_index] = swap_atom
        elif not neighbours:
            pass

    # # r1 after swap:
    # for atom in r1_wo_shell:
    #     logging.debug(atom)

    for d in dlist:
        if d.type == "interstitial" and d.label not in interstitials:
            interstitials.add(d.label)

    for atom in r1_wo_shell:
        xyz = np.array([atom.x, atom.y, atom.z])
        distance = np.linalg.norm(xyz - dcentre)
        if atom.label in interstitials:
            laccept = False
            max_attempts = 1000
            attempt = 0
            # Exclude the current atom from geometry check to avoid checking against itself
            r1_wo_shell_excluding_self = [a for a in r1_wo_shell if a is not atom]
            while not laccept:
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Could not find valid position for interstitial {atom.label} "
                        f"after {max_attempts} attempts. Try increasing step_size or relaxing geometry constraints."
                    )
                atom_n = moveclass_cyril(atom, step_size, r1, distance, 3.0)
                newxyz = np.array([atom_n.x, atom_n.y, atom_n.z])
                if np.linalg.norm(newxyz - dcentre) < r1 and atom_math.geo_checker(
                    atom_n, r1_wo_shell_excluding_self, 1.0
                ):
                    laccept = True
                    new_r1.append(atom_n)
                attempt += 1
        else:
            new_r1.append(atom)

    result = new_r1.copy()
    for atom in new_r1:
        if atom.has_shel():
            new_atom = atom.copy()
            new_atom.type = "shel"
            new_atom.q = 0.0
            new_atom.assign_charge()
            result.append(new_atom)
    return result, dcentre


def bh_move_dynamic(r1_list, filename, step_size) -> list:
    dcentre = input_parser.get_dcenter_from_out(filename)
    ri, r1, r2 = input_parser.get_cutoff()
    new_r1 = []
    dlist = input_parser.get_defect_list()

    r1_wo_shell = []
    # Pop shell
    for atom in r1_list:
        if atom.type != "s":
            r1_wo_shell.append(atom)

    unique_defect = set()
    for d in dlist:
        unique_defect.add(d.label)
    for atom in r1_wo_shell:
        if atom.label in unique_defect:
            laccept = False
            while not laccept:
                atom_n = move_atom(atom, step_size)
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
            new_atom.type = "shel"
            new_atom.q = 0.0
            new_atom.assign_charge()
            result.append(new_atom)
    return result


def bh_move(r1_list, filename) -> list:
    dcentre = input_parser.get_dcenter_from_out(filename)
    ri, r1, r2 = input_parser.get_cutoff()
    xmax = dcentre[0] + ri
    ymax = dcentre[1] + ri
    zmax = dcentre[2] + ri
    new_r1 = []
    dlist = input_parser.get_defect_list()

    r1_wo_shell = []
    # Pop shell
    for atom in r1_list:
        if atom.type != "s":
            r1_wo_shell.append(atom)

    # Move defect species within ri
    ri = []
    rrest = []
    for atom in r1_wo_shell:
        xyz = np.array([atom.x, atom.y, atom.z])
        dist = np.linalg.norm(xyz - dcentre)
        if dist <= ri:
            ri.append(atom)
        rrest.append(atom)
    step_size = 0.05
    new_ri = []
    l_max = 0.1
    for atom in ri:
        accepted = False
        for d in dlist:
            if atom.label == d.label:
                while not accepted:
                    old = np.array([atom.x, atom.y, atom.z])
                    dist_to_center = np.linalg.norm(old - dcentre)
                    lamda = l_max * (ri - np.sqrt(dist_to_center))
                    atom.x = np.random.uniform(low=-1, high=1) * l_max
                    atom.y = np.random.uniform(low=-1, high=1) * l_max
                    atom.z = np.random.uniform(low=-1, high=1) * l_max
                    xyz = np.array([atom.x, atom.y, atom.z])
                    dist = np.linalg.norm(xyz - dcentre)
                    if dist < ri:
                        new_ri.append(atom)
                        accepted = True
                        break
        if not accepted:
            new_ri.append(atom)

    for atom in new_ri:
        new_r1.append(atom)

    for atom in rrest:
        new_r1.append(atom)

    result = new_r1.copy()
    # put shells on
    for atom in new_r1:
        if atom.has_shel():
            new_atom = atom.copy()
            new_atom.type = "shel"
            new_atom.q = 0.0
            new_atom.assign_charge()
            result.append(new_atom)
    return result


# @profile
def monte_carlo_generator() -> "tuple":
    """Generate Monte Carlo structure with defects and return atom list and defect center."""
    logger = get_auto_logger()
    logger.info("Starting Monte Carlo structure generation")
    
    ml = Mott_Littleton()
    ml.initialise()
    logger.debug(f"Mott-Littleton initialized with cutoffs: r1={ml.cutoff[0]}, r2={ml.cutoff[1]}")
    
    a = Atom()
    cell = Cell()
    r_list = a.get_r1_after()
    r1, r2 = ml.cutoff
    dcentre = input_parser.get_defect_centre_from_res(ml.res_file)
    logger.info(f"Loaded {len(r_list)} atoms from region 1")
    logger.debug(f"Defect center coordinates: {dcentre}")

    # Filter shells and reset charges in one pass
    logger.debug("Filtering shell atoms and resetting charges")
    r1_wo_shell = []
    shell_count = 0
    for atom in r_list:
        if atom.type != "she":
            atom.q = 0.0  # Reset charge inline
            r1_wo_shell.append(atom)
        else:
            shell_count += 1
    
    logger.info(f"Filtered out {shell_count} shell atoms, {len(r1_wo_shell)} core atoms remaining")

    dlist = input_parser.get_defect_list(ml.control_file)
    vac_count, sub_count, inter_count = ml.defect_count
    logger.info(f"Defect counts - Vacancies: {vac_count}, Substitutions: {sub_count}, Interstitials: {inter_count}")

    # Use region_cutoff if specified, otherwise use r1
    region_cutoff = ml.region_cutoff if ml.region_cutoff is not None else r1
    logger.debug(f"Using region cutoff: {region_cutoff}")
    dcentre_arr = np.array(dcentre)

    # Pre-filter impurity defects
    impurity_list = [(d.label, d.atom2) for d in dlist if d.type == "impurity"]
    logger.debug(f"Found {len(impurity_list)} impurity defects: {impurity_list}")

    r1_w_impurity = r1_wo_shell.copy()
    # Optimize impurity substitution - only substitute atoms within region_cutoff
    for impure_atom, old_atom in impurity_list:
        # Filter atoms within region_cutoff
        old_atom_indices = [
            i
            for i, atom in enumerate(r1_wo_shell)
            if atom.label == old_atom
            and np.linalg.norm(np.array([atom.x, atom.y, atom.z]) - dcentre_arr)
            <= region_cutoff
        ]

        while input_parser.check_atom_count(r1_w_impurity, impure_atom) != sub_count:
            if old_atom_indices:
                chosen_index = np.random.choice(old_atom_indices)
                r1_w_impurity[chosen_index].label = impure_atom
                # Remove used index to avoid reselection
                old_atom_indices.remove(chosen_index)

    #####################
    #######Vaccancies####
    #####################
    vac_list = [d for d in dlist if d.type == "vacancy"]
    # Iterate through list of atoms containing impurity, select all atoms that vacancy applies within region_cutoff, and randomly choose one to delete.

    for vac in vac_list:
        to_del = vac.label
        matching_indices = [
            i
            for i, atom in enumerate(r1_w_impurity)
            if atom.label == to_del
            and np.linalg.norm(np.array([atom.x, atom.y, atom.z]) - dcentre_arr)
            <= region_cutoff
        ]

        if matching_indices:
            chosen_index = np.random.choice(matching_indices)
            r1_w_impurity.pop(chosen_index)

    # Process interstitials
    inter_list = []
    interstitial_defects = [d for d in dlist if d.type == "interstitial"]

    for d in interstitial_defects:
        inter = Atom()
        inter.label = d.label
        inter.type = "cor"
        inter.q = 0.0

        # Optimized placement with early exit - use region_cutoff instead of r1
        max_attempts = 1000  # Prevent infinite loops
        attempts = 0
        while attempts < max_attempts:
            random_point = np.random.uniform(-region_cutoff, region_cutoff, 3)
            if np.linalg.norm(random_point - dcentre_arr) <= region_cutoff:
                inter.x, inter.y, inter.z = random_point
                if atom_math.geo_checker(inter, r1_w_impurity, 1.0):
                    break
            attempts += 1

        inter_list.append(inter)

    r1_w_inter = (
        r1_w_impurity + inter_list
    )  # Use + instead of extend for better performance
    
    logger.info(f"Created combined atom list with {len(inter_list)} interstitials added")
    logger.debug(f"Total atoms after defect insertion: {len(r1_w_inter)}")

    # Combine charge assignment and shell creation
    logger.debug("Assigning charges and creating shell atoms")
    r1_new = []
    shell_atoms_created = 0
    for atom in r1_w_inter:
        atom.assign_charge()
        r1_new.append(atom)
        if atom.has_shel():
            new_atom = atom.copy()
            new_atom.type = "shel"
            new_atom.q = 0.0
            new_atom.assign_charge()
            r1_new.append(new_atom)
            shell_atoms_created += 1

    logger.info(f"Monte Carlo structure generation completed")
    logger.info(f"Final structure: {len(r1_new)} total atoms ({shell_atoms_created} shell atoms created)")
    logger.debug(f"Returning defect center: {dcentre}")
    
    return r1_new, dcentre


def write_input(fname, atoms, dcentre, ex_opt="") -> None:
    """Write GULP input file for optimization with given atoms and defect center."""
    logger = get_auto_logger()
    logger.info(f"Writing GULP input file: {fname}")
    logger.debug(f"Writing {len(atoms)} atoms with defect center {dcentre}")
    
    # gbi = get_global_variables()
    dcentre = np.array(dcentre)
    ml = Mott_Littleton()
    ml.initialise()
    a = Atom()
    atom_list = a.get_atoms_after()
    # dcentre = np.array(input_parser.get_defect_centre())
    cell = Cell()
    r1, r2 = ml.cutoff
    dcentre_string = " ".join(map(str, dcentre.flatten()))
    options = ml.options
    
    logger.debug(f"Using cutoffs: r1={r1}, r2={r2}")
    logger.debug(f"Defect center string: {dcentre_string}")

    # Collect all lines in a list
    lines = []
    lines.append("opti conp defect nodsymm \n")
    # lines.append('defect nodsymm energy \n')
    lines.append("cell \n")
    lines.append(str(cell.get_cell()))
    lines.append("fractional 1 \n")
    lines.extend([str(atom) for atom in atom_list])
    lines.append("center cart " + dcentre_string + "\n")
    lines.append(f"size {r1} {r2} \n")
    lines.append("region_1 \n")
    lines.extend([str(atom) for atom in atoms])
    # lines.append('lib ' + gbi.libfile + '\n')  # Uncomment if needed
    lines.extend(options)
    lines.extend([line + "\n" for line in ex_opt])
    with open(ml.lib_file, "r") as flib:
        content = flib.read()
    lines.extend(content)

    # Write all collected lines to the file at once
    logger.debug(f"Writing {len(lines)} lines to file")
    with open(fname, "w") as f:
        f.writelines(lines)
    
    logger.info(f"Successfully wrote GULP input file: {fname}")
    logger.debug(f"File contains {len(atom_list)} bulk atoms + {len(atoms)} region 1 atoms")


def write_input_sa(fname, atoms, dcentre, ex_opt="") -> None:
    # gbi = get_global_variables()
    dcentre = np.array(dcentre)
    ml = Mott_Littleton()
    ml.initialise()
    a = Atom()
    atom_list = a.get_atoms_after()
    # dcentre = np.array(input_parser.get_defect_centre())
    cell = Cell()
    r1, r2 = ml.cutoff
    dcentre_string = " ".join(map(str, dcentre.flatten()))
    options = ml.options

    # Collect all lines in a list
    lines = []
    # lines.append('opti conp defect nodsymm \n')
    lines.append("defect nodsymm energy \n")
    lines.append("cell \n")
    lines.append(str(cell.get_cell()))
    lines.append("fractional 1 \n")
    lines.extend([str(atom) for atom in atom_list])
    lines.append("center cart " + dcentre_string + "\n")
    lines.append(f"size {r1} {r2} \n")
    lines.append("region_1 \n")
    lines.extend([str(atom) for atom in atoms])
    # lines.append('lib ' + gbi.libfile + '\n')  # Uncomment if needed
    lines.extend(options)
    lines.extend([line + "\n" for line in ex_opt])
    with open(ml.lib_file, "r") as flib:
        content = flib.read()
    lines.extend(content)

    # Write all collected lines to the file at once
    logger.debug(f"Writing {len(lines)} lines to file")
    with open(fname, "w") as f:
        f.writelines(lines)
    
    logger.info(f"Successfully wrote GULP input file: {fname}")
    logger.debug(f"File contains {len(atom_list)} bulk atoms + {len(atoms)} region 1 atoms")


def write_input_once(n) -> None:
    ml = Mott_Littleton()
    # print("The direcctory before writing inputs is : \n")
    # print(f"{ml.control_file}")
    ml.initialise()
    a = Atom()
    atom_list = a.get_atoms_after()
    # dcentre = np.array(input_parser.get_defect_centre())
    cell = Cell()
    cell.get_cell()
    r1, r2 = ml.cutoff
    options = ml.options
    atoms, dcentre = monte_carlo_util.monte_carlo_generator()
    # print(dcentre)
    # atoms = monte_carlo_util.monte_carlo_generator()
    dcentre = np.array([dcentre])
    dcentre_string = " ".join(map(str, dcentre.flatten()))
    fname = "A" + str(n) + ".gin"
    lines = []
    lines.append("opti conp defect nodsymm \n")
    lines.append("cell \n")
    lines.append(str(cell))
    lines.append("fractional 1 \n")
    lines.extend([str(atom) for atom in atom_list])
    lines.append("center cart " + dcentre_string + "\n")
    lines.append(f"size {r1} {r2} \n")
    lines.append("region_1 \n")
    lines.extend([str(atom) for atom in atoms])
    lines.extend(options)
    # lines.extend([line + '\n' for line in ex_opt])
    with open(fname, "w") as f:
        f.writelines(lines)


def write_input_once_sphere(n, atoms) -> None:
    ml = Mott_Littleton()
    # print("The direcctory before writing inputs is : \n")
    # print(f"{ml.control_file}")
    ml.initialise()
    a = Atom()
    atom_list = a.get_atoms_after()
    dcentre = np.array(input_parser.get_defect_centre(ml.control_file))
    cell = Cell()
    cell.get_cell()
    r1, r2 = ml.cutoff
    options = ml.options
    # print(dcentre)
    # atoms = monte_carlo_util.monte_carlo_generator()
    dcentre = np.array([dcentre])
    dcentre_string = " ".join(map(str, dcentre.flatten()))
    fname = "A" + str(n) + ".gin"
    lines = []
    lines.append("opti conp defect nodsymm \n")
    lines.append("cell \n")
    lines.append(str(cell))
    lines.append("fractional 1 \n")
    lines.extend([str(atom) for atom in atom_list])
    lines.append("center frac " + dcentre_string + "\n")
    lines.append(f"size {r1} {r2} \n")
    lines.append("region_1 \n")
    lines.extend([str(atom) for atom in atoms])
    lines.extend(options)
    # lines.extend([line + '\n' for line in ex_opt])
    with open(fname, "w") as f:
        f.writelines(lines)
