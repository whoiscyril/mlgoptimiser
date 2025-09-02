import subprocess
import os
import numpy as np

def get_dirs():
    cwd = os.getcwd()
    dir_path = []
    dir_name = []
    for dirpath, dirnames, filenames in os.walk(cwd):
        for dirname in dirnames:
            dir_path.append(os.path.join(dirpath, dirname))
            dir_name.append(dirname)
    return dir_path, dir_name

def get_defect_energy(path):
    old_dir = os.getcwd()
    energies = []
    result = []
    for dir in path:
        os.chdir(dir)
        basename = os.path.basename(dir)
        outfile_name = "gulp_klmc.gout"
        with open(outfile_name, "r") as f:
            for line in f:
                if line.startswith("  Final defect energy  ="):
                    parts = line.strip().split()
                    energies.append((basename, float(parts[4])))
                elif line.startswith("  Final coordinates of region 1 :"):
                    for _ in range(5):
                        next(f)
                    for line in f:
                            parts = line.strip().split()
                            if line.startswith("--"):
                                break
                            elif parts[1] == "Tc" and parts[2] == "c":
                                tc_array = np.array([float(x) for x in parts[3:6]])
                            elif parts[1] == "Li" and parts[2] == "c":
                                li_array = np.array([float(x) for x in parts[3:6]])
        
        result_array = np.subtract(tc_array,li_array)
        distance = np.linalg.norm(tc_array-li_array)
        result.append((float(result_array[0]), float(result_array[1]), float(result_array[2]), float(distance)))
    os.chdir(old_dir)

    with open("energies.txt", "w") as f:
        for energy, res in zip(energies, result):
            f.write(f"{energy[0]}\t{energy[1]:.4f}\t" + 
                "\t".join(f"{x:.4f}" for x in res) + "\n")


def check_vecs(vec1, vec2, epsilon=1e-2):
    #assuming same shape
    if len(vec1) != len(vec2):
        return False    
    for i in range(len(vec1)):
        if abs(vec1[i] - vec2[i]) > epsilon:
            return False
    return True

    

def remove_duplicates():
    threshold = 0.01
    ids = []
    energies = []
    vecs = []
    dist = []
    temp = {}
    result_dict = {}  # Renamed dict to result_dict to avoid conflict
    with open("energies.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            ids.append(parts[0])
            energy = float(parts[1])
            energies.append(energy)
            vecs.append([float(x) for x in parts[2:5]])
            dist.append(float(parts[5]))

    for i in range(len(energies)):
        n = 0
        for j in range(len(energies)):
            e_diff = abs(energies[i] - energies[j])
            same = check_vecs(vecs[i], vecs[j])  # Assuming check_vecs is defined elsewhere
            if e_diff < threshold and same:
                n += 1
        temp[ids[i]] = {'energy': energies[i], 'vector': vecs[i], 'distance': dist[i], 'frequency': n}

    result = {}
    unique_value1 = set()

    for key, values in temp.items():
        value1 = values['energy']
        value2 = values['vector']
        is_unique = True
        for existing_key, existing_values in result.items():
            existing_value1 = existing_values['energy']
            existing_value2 = existing_values['vector']
            if abs(value1 - existing_value1) <= threshold and check_vecs(value2, existing_value2):
                is_unique = False
                break
        if is_unique:
            result[key] = values
    sorted_result = dict(sorted(result.items(), key=lambda item: item[1]['energy'], reverse=True))

    with open("result.txt", "w") as f_out:
        for key, values in sorted_result.items():
            energy = values['energy']
            vector = ', '.join(map(str, values['vector']))
            distance = values['distance']
            frequency = values['frequency']
            f_out.write(f"{key} : {energy}, [{vector}], {distance}, {frequency}\n")
        
def generate_xyz() -> 'None':
    # get .gout file name
    suffix = '.gout'
    filename = ' '
    pos = []
    center = []
    result = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(suffix):
                filename = file
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('  Defect centre is at'):
                parts = line.strip().split()
                center = [parts[4], parts[5], parts[6]]
            if line.startswith('  Final coordinates of region 1 :'):
                for _ in range(5):
                    line = f.readline()
                for line in f:
                    if line.startswith("--"):
                        break
                    parts = line.strip().split()
                    if parts[2] != 's':
                        pos.append(line)

    # Start calculating distance between defect center and atoms
    center = np.array(center).astype(np.float64)
    for line in pos:
        parts = line.strip().split()
        xyz = np.array([parts[3], parts[4], parts[5]]).astype(np.float64)
        dist = np.linalg.norm(center-xyz)
        if dist < 7. :
            line_to_write = parts[1] + ' ' + ' '.join(map(str, xyz))
            result.append(line_to_write)

    with open('r0.xyz', 'w') as f:
        f.write(str(len(result)))
        f.write('\n')
        f.write('\n')
        for line in result:
            f.write(line)
            f.write('\n')

def create_xyz(n):
    dirs = []
    with open("result.txt", "r") as f:
        for line in f:
            dir = line.strip().split()[0]   
            dirs.append(dir)
    dirs.reverse()
    for index, dir in enumerate(dirs):
        parent_dir = os.getcwd()
        path = os.path.join(os.getcwd(), dir)    
        if index > n:
            break
        os.chdir(path)
        generate_xyz()
        os.chdir(parent_dir)







def main():
    subprocess.run(['rm', '-rf', '__pycache__'], check=True)
    path, name = get_dirs()
    get_defect_energy(path)
    remove_duplicates()
    create_xyz(10)
if __name__ == "__main__":
    main()
