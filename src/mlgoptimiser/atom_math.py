from .atom import Atom
import math
import numpy as np

def get_dist(atom1, atom2):
    dx = atom1.x - atom2.x
    dy = atom1.y - atom2.y
    dz = atom1.z - atom2.z
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    return dist

def csmap(atom_list) -> 'dict':
    map = {}
    for index1, atom1 in enumerate(atom_list):
        for index2, atom2 in enumerate(atom_list):
            dist2 = (atom1.x - atom2.x)**2 + (atom1.y - atom2.y)**2 + (atom1.z - atom2.z)**2
            dist = math.sqrt(dist2)
            if index1 != index2 and dist < 0.3 and atom1.label == atom2.label and atom1.type != atom2.type:
                map[index1] = index2
    return map

def pop_shell(atom_list) -> 'list':
    list = []
    for atom in atom_list:
        if atom.type == 'cor':
            list.append(atom)
    return list

def geo_checker(atom, atom_list, cutoff):
    for _atom in atom_list:
        if get_dist(atom, _atom) <= cutoff:
            return False
        return True

def at_center(atom, center):
    xyz = np.array([atom.x, atom.y, atom.z])
    center = np.array(center)
    if np.linalg.norm(xyz - center) < 0.01:
        return True
    return False

def filter_grid_points(grid, atoms):
    """Filter out grid points that are occupied by atoms.

    Parameters:
        grid (np.ndarray): N x 3 array containing 3D Cartesian coordinates of grid points.
        atoms (list): List of atom objects with x, y, z attributes.

    Returns:
        np.ndarray: Filtered grid points.
    """
    new_grid = []
    for point in grid:
        g_xyz = np.array(point)
        if all(np.linalg.norm(g_xyz - np.array([atom.x, atom.y, atom.z])) > 0.3 for atom in atoms):
            new_grid.append(point)
    
    return np.array(new_grid)
