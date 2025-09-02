import os
from . import input_parser
import glob

class Mott_Littleton:
    def __init__(self, directory=None):
        # Set working directory to the given one or use current directory
        self.directory = os.path.abspath(directory) if directory else os.getcwd()
        
        # Define file paths
        self.config_file = os.path.join(self.directory, 'input', 'CONFIG')
        self.control_file = os.path.join(self.directory, 'input', 'CONTROL')
        self.lib_file = os.path.join(self.directory, 'input', 'std.lib')

        res_files = glob.glob(os.path.join(self.directory, 'input', '*.res'))
        self.res_file = res_files[0] if res_files else None 

        # Initialize properties
        self.cutoff = None
        self.options = None
        self.defect_center = None
        self.defect_list = None
        self.defect_count = None
        self.species = None
        self.region_cutoff = None

    def initialise(self):
        # Check for the existence of input files before attempting to read them
        if not os.path.exists(self.control_file):
            print(f"Warning: CONTROL file not found in {self.control_file}")
        else:
            self.cutoff = input_parser.get_cutoff(self.control_file)
            self.defect_center = input_parser.get_defect_centre(self.control_file)
            self.defect_list = input_parser.get_defect_list(self.control_file)
            self.defect_count = input_parser.get_defect_count(self.control_file)
            self.region_cutoff = input_parser.get_region_cutoff(self.control_file)

        if not os.path.exists(self.config_file):
            print(f"Warning: CONFIG file not found in {self.config_file}")
        else:
            self.options = input_parser.get_options(self.config_file)

        if not os.path.exists(self.lib_file):
            print(f"Warning: std.lib file not found in {self.lib_file}")
        else:
            self.species = input_parser.get_species_data(self.lib_file)
