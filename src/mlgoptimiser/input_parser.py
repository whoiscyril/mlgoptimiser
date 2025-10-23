# from globals import GlobalVariables
import math

import numpy as np

from .defect import Defect


def get_seed_coords(file):
    atoms = []
    with open(file, "r") as f:
        for line in f:
            atoms.append(line)
    return atoms


def get_sphere_param(file):
    with open(file, "r") as f:
        for line in f:
            if line.startswith("DENSITY"):
                density = int(line.strip().split()[-1])
            elif line.startswith("RADIUS"):
                radius = float(line.strip().split()[-1])
    return tuple((density, radius))


def get_gulp_path(file):
    path = ""
    with open(file, "r") as f:
        for line in f:
            if line.startswith("GULP_PATH"):
                parts = line.split()
                path = parts[-1].strip("'\"")
    return path


def get_species_data(filename):
    species = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("species"):
                natom = int(line.strip().split()[-1])
                for _ in range(natom):
                    line = f.readline()
                    species.append(line.strip().split())
    return species


def check_atom_count(atom_list, atom_to_be_checked):
    n = 0
    for atom in atom_list:
        if atom.label == atom_to_be_checked:
            n += 1
    return n


def get_defect_count(filename):
    vac_count = 0
    sub_count = 0
    inter_count = 0

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("BEGIN_DLIST"):
                # Start counting defects after 'dlist' is found
                for line in f:
                    if line.startswith("END_DLIST"):
                        break
                    line = line.strip()  # Remove any leading/trailing spaces
                    if not line:  # Stop if the line is empty
                        break
                    if line.startswith("vacancy"):
                        vac_count += 1
                    elif line.startswith("interstitial"):
                        inter_count += 1
                    elif line.startswith("impurity"):
                        sub_count += 1

    return vac_count, sub_count, inter_count


def get_defect_center_from_source(filename):
    dcenter = None  # Initialize dcenter to handle cases where it's not found
    type = None
    atom1 = None
    atom2 = None

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("DEFECT_CENTER"):
                parts = line.strip().split()
                # Ensure we have enough parts in the line to prevent index errors
                if len(parts) >= 6:
                    type = parts[2]
                    atom1 = parts[3]
                    atom2 = parts[4]
                    dcenter = np.array(
                        [float(parts[5]), float(parts[6]), float(parts[7])]
                    )
                break  # Exit loop after finding the defect center

    # Handle the case where defect_center is not found
    if dcenter is None:
        raise ValueError("defect_center not found in the file")

    return dcenter, type, atom1, atom2


def check_energy(filename) -> "bool":
    energy = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("  Start of defect optimisation :"):
                for line in f:
                    if line.startswith("  Cycle:"):
                        parts = line.strip().split()
                        energy.append(parts[4])
                        energy.append(parts[6])
    if any("***" in s for s in energy):
        return True
    else:
        return False


def append_content(file, content):
    with open(file, "a") as dest:
        dest.write(content)


def check_caution(filename) -> "bool":
    with open(filename, "r") as f:
        content = f.read()
    if "with caution" in content:
        return True
    return False


def get_gnorm(filename) -> "float":
    result = 0.0
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("  Final defect Gnorm   ="):
                result = line.strip().split()[-1]
    return float(result)


def get_r1_before(filename) -> "list":
    from .atom import Atom
    result = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("  Explicit region 1 specified :"):
                for _ in range(5):
                    line = f.readline()
                for line in f:
                    if line.startswith("--"):
                        break
                    parts = line.strip().split()
                    a = Atom()
                    a.label = parts[1]
                    a.type = parts[2]
                    a.x = float(parts[3])
                    a.y = float(parts[5])
                    a.z = float(parts[7])
                    a.q = float(parts[-2])
                    result.append(a)
    return result


def get_r1_after(filename) -> "list":
    from .atom import Atom
    result = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("  Final coordinates of region 1 :"):
                for _ in range(5):
                    line = f.readline()
                for line in f:
                    if line.startswith("--"):
                        break
                    parts = line.strip().split()
                    a = Atom()
                    a.label = parts[1]
                    a.type = parts[2]
                    a.x = float(parts[3])
                    a.y = float(parts[4])
                    a.z = float(parts[5])
                    a.q = float(parts[-1])
                    result.append(a)
    return result


def get_r1_after_from_res(filename) -> "list":
    from .atom import Atom
    result = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("region_1"):
                for line in f:
                    if line.startswith("species"):
                        break
                    parts = line.strip().split()
                    a = Atom()
                    a.label = parts[0]
                    a.type = parts[1]
                    a.x = float(parts[2])
                    a.y = float(parts[3])
                    a.z = float(parts[4])
                    a.q = float(parts[5])
                    result.append(a)
    return result


def get_dcenter_from_out(filename):
    dcenter = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("  Defect centre is at"):
                parts = line.strip().split()
                dcenter.append(parts[4])
                dcenter.append(parts[5])
                dcenter.append(parts[6])
    dcenter = np.array(dcenter)
    dcenter = dcenter.astype(np.float64)
    return dcenter


# def get_r1(filename) -> 'list':
#     r0, r1, r2 = get_cutoff()
#     result = []
#     dcenter = []
#     with open(filename, 'r') as f:
#         for line in f:
#             if line.startswith('  Defect centre is at'):
#                 parts = line.strip().split()
#                 dcenter.append(parts[4])
#                 dcenter.append(parts[5])
#                 dcenter.append(parts[6])
#         f.seek(0)

#     dcenter = np.array(dcenter)
#     dcenter = dcenter.astype(np.float64)
#     r1_atoms = get_r1_after(filename)
#     for line in r1_atoms:
#         parts = line.strip().split()
#         a = Atom()
#         a.label = parts[1]
#         a.type = parts[2]
#         a.x = float(parts[3])
#         a.y = float(parts[4])
#         a.z = float(parts[5])
#         a.q = float(parts[-1])
#         xyz = np.array([a.x, a.y, a.z])
#         dist_to_center = np.linalg.norm(dcenter - xyz)

#         if dist_to_center < r0 and a.type == 'c':
#             result.append(a)

#     return result


def get_algo(file) -> "str":
    algo = ""
    with open(file, "r") as f:
        for line in f:
            if line.startswith("ALGORITHM"):
                algo = line.strip().split()[-1]
    return algo


def get_defect_list(file) -> "list":
    list = []
    with open(file, "r") as f:
        for line in f:
            if line.startswith("BEGIN_DLIST"):
                for line in f:
                    if line.startswith("END_DLIST"):
                        break
                    parts = line.strip().split()
                    d = Defect()
                    d.type = parts[0] if len(parts) > 0 else None
                    d.label = parts[1] if len(parts) > 1 else None
                    d.atom2 = parts[2] if d.type == "impurity" else None
                    list.append(d)
    return list


def get_options(file) -> "list":
    list = []
    with open(file, "r") as f:
        for line in f:
            if line.startswith("BEGIN_OPTIONS"):
                for line in f:
                    if line.startswith("END_OPTIONS"):
                        break
                    list.append(line)
    return list


def get_cutoff(file) -> "list":
    list = []
    with open(file, "r") as f:
        for line in f:
            if line.startswith("ML_SIZE"):
                vals = line.split(":")[-1].strip().split()
                list = [float(x) for x in vals[-2:]]
    return list


def get_defect_centre(file) -> "list":
    list = []
    with open(file, "r") as f:
        for line in f:
            if line.startswith("DEFECT_CENTER"):
                parts = line.strip().split()
                vals = line.split(":")[-1].strip().split()
                list = [float(x) for x in vals[-3:]]
    return list


def get_defect_centre_from_res(filename) -> "list":
    list = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("centre"):
                parts = line.strip().split()
                list.append([float(x) for x in parts[-3:]])
    return list


def get_mc_steps(file) -> "int":
    with open(file, "r") as f:
        for line in f:
            if line.startswith("MC_STEPS"):
                steps = line.strip().split()[-1]
    return int(steps)


def get_sa_max_temp() -> "float":
    max_temp = 0.0
    gbi = GlobalVariables()
    gbi.initialise()
    filename = gbi.infile
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("sa_param"):
                for line in f:
                    if line and line.startswith("max_temp"):
                        max_temp = line.strip().split()[-1]
    return max_temp


def get_sa_mc_steps() -> "int":
    steps = 0
    gbi = GlobalVariables()
    gbi.initialise()
    filename = gbi.infile
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("sa_param"):
                for line in f:
                    if line and line.startswith("sa_mc_steps"):
                        steps = line.strip().split()[-1]
    return steps


def get_defect_energy(fname) -> "str":
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("  Final defect energy  ="):
                denergy = line.strip().split()[-1]
    return denergy


def get_defect_energy_single(fname) -> "str":
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("  Total defect energy"):
                denergy = line.strip().split()[-2]
    return denergy


def get_region_cutoff(file) -> "float":
    """Get the region cutoff for defect placement from CONTROL file.
    If not specified, returns None to use default r1 behavior."""
    with open(file, "r") as f:
        for line in f:
            if line.startswith("REGION_CUTOFF"):
                cutoff = line.strip().split()[-1]
                return float(cutoff)
    return None
