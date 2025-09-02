import time
import os
import math
import re
def has_finished(filename):
    with open(filename, 'r') as f:
        content = f.read()
        if "Job Finished" in content:
            return True
        return False

def exist(filename):
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path) == True:
        return True
    return False

def same_energy(num1, num2, thresh) -> "bool":
    if math.isclose(num1, num2, rel_tol=thresh):
        return True
    return False

def get_slurm_id() -> 'str':
    for _, _, filenames in os.walk(os.getcwd()):
        for filename in filenames:
            if filename.endswith(".out"):
                match = re.search(r'\d+', filename)
                if match:
                    return match.group()
    return None

def check_terminate(deltaE, nstep, nstepchange, nstable) :
    # If total attempt exceeds cretain number, terminates:
    if nstep > 100:
        return True, "Terminates due to too many attemps \n"
    # if energy is converegd for a certain number of step, terminates:
    elif abs(deltaE) < 1e-05 and nstable >= 10 :
        return True, "Terminates due to sucessful location of minimum \n"
    # if there are 10 step changes already, terminates:
    # elif nstepchange >= 10:
    #     return True, "Terminates due to too many step size changes \n "
    # If rejected attemps greater than 10 then terminates
    else:
        return False, "Continue running \n"
    
    




