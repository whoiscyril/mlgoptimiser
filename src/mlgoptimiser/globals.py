import os
from . import input_parser
# import argparse

class GlobalOptimisation:
    def __init__(self):
        dir_path = os.getcwd()
        input_path = os.path.join(dir_path, 'input')
        config_file = os.path.join(input_path, 'CONFIG')
        control_file = os.path.join(input_path, 'CONTROL')
        self.gulp_path = input_parser.get_gulp_path(control_file)
        self.algorithm = input_parser.get_algo(control_file)
        if self.algorithm.lower() == 'mc':
            self.mc_steps = input_parser.get_mc_steps(control_file)
        else:
            self.mc_steps = None
        if self.algorithm.lower() == 'sphere':
            self.sphere_params = input_parser.get_sphere_param(control_file)
        else:
            self.sphere_params = None
        
# class GlobalVariables:
#     def __init__(self):
#         self.goutfile = None
#         self.ginfile = None
#         self.gresfile = None
#         self.gmode = None
#         self.infile = None
#         self.gulp_path = None
#         self.libfile = None
#         self.submission = None
    
#     def initialise(self):
#         # setting path
#         dir_path = os.getcwd()
        

#     # def initialise(self):
#     #     #dir_path = os.path.dirname(os.getcwd())
#     #     dir_path = os.getcwd()
#     #     if not dir_path.endswith(os.sep):
#     #         dir_path += os.sep
#     #     suffixes = ['.mc']
#     #     infile = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if any(file.endswith(suffix) for suffix in suffixes)]
#     #     if len(infile) == 0:
#     #         print("No mc_input file found")
#     #     elif len(infile) >= 2:
#     #         print("Too many mc_input files detected")
#     #     else:
#     #         self.infile = os.path.join(os.getcwd(),infile[0])
#     #         logging.info("The program input file is identified as: " + self.infile)
        
#     #     with open(self.infile, 'r') as f:
#     #         for line in f:
#     #             if line.startswith("input_file"):
#     #                 line = f.readline()
#     #                 self.ginfile = os.path.join(os.getcwd(),line.strip())
#     #             if line.startswith("library"):
#     #                 line = f.readline()
#     #                 self.libfile = os.path.join(os.getcwd(),line.strip())
#     #     suffixes = [".sh"]
#     #     submissionfile = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if any(file.endswith(suffix) for suffix in suffixes)]
#     #     if len(submissionfile) > 1:
#     #         print("Too many submission scripts")
#     #     # elif len(submissionfile) == 0:
#     #         # print("No submission script")
#     #     # else:
#     #         # self.submission = submissionfile[0]

#     #     suffixes = [".res"]
#     #     resfile = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if any(file.endswith(suffix) for suffix in suffixes)]
#     #     if len(resfile) > 1:
#     #         print("Too many restart files")
#     #     elif len(resfile) == 0:
#     #         print("No restart file found")
#     #     else:
#     #         self.gresfile = resfile[0]
#     #         logging.info("The restart file is identified as: " + self.gresfile)
#     #     parser = argparse.ArgumentParser()
#     #     parser.add_argument('-m', '--mode', type=int, help="Mode of program")
#     #     args = parser.parse_args()
#     #     self.gmode = args.mode

#     #     with open(self.infile, 'r') as f:
#     #         for line in f:
#     #             if line.startswith("gulp_path"):
#     #                 line = f.readline()
#     #                 self.gulp_path = line.strip()
# _gbi_instance = None

# def get_global_variables():
#     global _gbi_instance
#     if _gbi_instance is None:
#         _gbi_instance = GlobalVariables()
#         _gbi_instance.initialise()
#     return _gbi_instance

# _species_cache = None

# def get_species_data(filename):
#     global _species_cache
#     if _species_cache is None:
#         species = []
#         with open(filename, 'r') as f:
#             for line in f:
#                 if line.startswith('species'):
#                     natom = int(line.strip().split()[-1])
#                     for _ in range(natom):
#                         line = f.readline()
#                         species.append(line.strip().split())
#         _species_cache = species
#     return _species_cache
