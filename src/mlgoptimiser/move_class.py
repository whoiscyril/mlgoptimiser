import random
import numpy as np
def pos_random(atom, dcenter, r1):
    dx = dcenter[0][0]
    dy = dcenter[0][1]
    dz = dcenter[0][2]
    x = np.random.uniform(dx-r1, dx+r1)
    y = np.random.uniform(dy-r1, dy+r1)
    z = np.random.uniform(dz-r1, dz+r1)
    atom.x = x
    atom.y = y
    atom.z = z
    return atom
def fix_dcenter_atom(atom, dcenter):
    dx = dcenter[0][0]
    dy = dcenter[0][1]
    dz = dcenter[0][2]
    atom.x = dx
    atom.y = dy
    atom.z = dz
    return atom


    
