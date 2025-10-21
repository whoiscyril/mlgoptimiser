import math
import os
import pickle
import shutil
import subprocess
import time
from collections import Counter
from typing import Dict, Tuple

import numpy as np

from . import input_parser, monitor, monte_carlo_util, run_gulp

DEFAULT_STEP_SIZE_RANGE = (0.2, 1.0)
ENERGY_TOLERANCE = 1e-4
GNORM_THRESHOLD = 0.01
FILE_POLL_INTERVAL = 5
TIMEOUT_LIMIT = 240

class BasinHoppingSimulator:
    # Class-level default configuration
    DEFAULT_CONFIG= {
        # "temperature": 298,       # Temperature for the Metropolis criterion.
        "max_cycles": 1000,  
        "default_step_size" : 1.     # Maximum number of cycles to run.
    }

    # def __init__(self, input_dir: str = "input", base_dir: str = os.getcwd(), config: Dict = None):
    #     self.input_dir = input_dir
    #     self.base_dir = base_dir
    #     self.bh_dir = os.path.join(self.base_dir, 'bh_test')
    #     os.makedirs(self.bh_dir, exist_ok=True)
    #     self.config = config if config is not None else BasinHoppingSimulator.DEFAULT_CONFIG.copy()
    #     self.setup_logging()
    #     self.reset_state()
    #     self.seed = input_parser.get_seed_coords(os.path.join(input_dir, 'seed.coord'))
    #     self.checkpoint_file = os.path.join(self.bh_dir, 'checkpoint.pkl')
    #     # Load or initialize state
    #     if os.path.exists(self.checkpoint_file):
    #         self.load_checkpoint()
    #     else:
    #         self.reset_state()
    #         self.seed = input_parser.get_seed_coords(os.path.join(self.input_dir, 'seed.coord'))
    #         self.start_cycle = 0

    def __init__(
        self,
        input_dir: str = "input",
        base_dir: str = None,
        config: Dict = None,
        step_size: float = None,
        fixed_step_size: bool = False
    ):
        # Merge Configs
        cfg = { **BasinHoppingSimulator.DEFAULT_CONFIG, **(config or {}) }
        self.config = cfg

        # Paths and directories
        self.input_dir = input_dir
        self.base_dir = base_dir or os.getcwd()
        self.bh_dir = os.path.join(self.base_dir, 'bh_test')
        os.makedirs(self.bh_dir, exist_ok=True)

        # Simulation config
        self.config = config.copy() if config is not None else BasinHoppingSimulator.DEFAULT_CONFIG.copy()

        # Step-size control flags
        self.fixed_step_size = fixed_step_size
        self.step_size_arg = step_size

        # Initialize state
        self.reset_state()

        # Override initial step size if provided
        default_sz = cfg["default_step_size"]
        if self.step_size_arg is not None:
            self.current_step_size = self.step_size_arg
        else:
            self.current_step_size = default_sz

        # Seed structure
        seed_path = os.path.join(self.input_dir, 'seed.coord')
        if os.path.exists(seed_path):
            self.seed = input_parser.get_seed_coords(os.path.join(self.input_dir, 'seed.coord'))
        else:
            self.seed = None

        # Checkpointing
        self.checkpoint_file = os.path.join(self.bh_dir, 'checkpoint.pkl')
        if os.path.exists(self.checkpoint_file):
            self.load_checkpoint()
        else:
            self.start_cycle = 0


    def reset_state(self):
        # Initialize counters and simulation parameters
        self.attempt_counter = 1
        self.total_attempt = 0
        self.success_attempt = 0
        self.rejected_attempt = 0
        self.stable_energy_count = 0
        self.best_energy = float('inf')
        self.best_dir = ""
        #self.current_step_size = np.random.uniform(*DEFAULT_STEP_SIZE_RANGE)
        # self.current_step_size = self.step_size
        self.is_first_cycle = True
        self.energy_log = {}  # Maps cycle: energy
        self.terminate = False

    def save_checkpoint(self, cycle: int):
        state = {
            'cycle': cycle,
            'attempt_counter': self.attempt_counter,
            'total_attempt': self.total_attempt,
            'success_attempt': self.success_attempt,
            'rejected_attempt': self.rejected_attempt,
            'stable_energy_count': self.stable_energy_count,
            'best_energy': self.best_energy,
            'best_dir': self.best_dir,
            'current_step_size': self.current_step_size,
            'is_first_cycle': self.is_first_cycle,
            'energy_log': self.energy_log,
            'last_energy': getattr(self, 'last_energy', None),
            'last_position': getattr(self, 'last_position', None),
            'prev_output': getattr(self, 'prev_output', None)
        }
        with open(self.checkpoint_file, 'wb') as cp:
            pickle.dump(state, cp)
        pass

    def load_checkpoint(self):
        with open(self.checkpoint_file, 'rb') as cp:
            state = pickle.load(cp)
        self.attempt_counter = state['attempt_counter']
        self.total_attempt = state['total_attempt']
        self.success_attempt = state['success_attempt']
        self.rejected_attempt = state['rejected_attempt']
        self.stable_energy_count = state['stable_energy_count']
        self.best_energy = state['best_energy']
        self.best_dir = state['best_dir']
        self.current_step_size = state['current_step_size']
        self.is_first_cycle = state['is_first_cycle']
        self.energy_log = state['energy_log']
        self.last_energy = state.get('last_energy')
        self.last_position = state.get('last_position')
        self.prev_output = state.get('prev_output')
        self.seed = input_parser.get_seed_coords(os.path.join(self.input_dir, 'seed.coord'))
        self.start_cycle = state['cycle'] + 1
        pass


    def setup_cycle_directory(self, cycle_label: str) -> str:
        cycle_dir = os.path.join(self.bh_dir, cycle_label)
        os.makedirs(cycle_dir, exist_ok=True)
        # Use our dedicated function to copy input files and scripts
        self.copy_input_files(cycle_dir)
        return cycle_dir

    def generate_structure(self) -> Tuple[object, object]:
        # Depending on the cycle, choose the appropriate structure generation:
        if self.is_first_cycle:
            pass
            if self.seed is not None:
                structure = self.seed
                _, center = monte_carlo_util.monte_carlo_generator()
            else:
                structure, center = monte_carlo_util.monte_carlo_generator()
            self.total_attempt += 1
        else:
            pass
            try:
                structure, center = monte_carlo_util.bh_move_inter_only(self.last_position, self.prev_output, self.current_step_size)
                self.total_attempt += 1
            except RuntimeError as e:
                pass
                return
        pass
        return structure, center

    def copy_input_files(self, dest_dir: str):
        # Assume input_dir exists in self.base_dir; copy it to dest_dir
        shutil.copytree(self.input_dir, os.path.join(dest_dir, 'input'), dirs_exist_ok=True)
        shutil.copy("gulp.sh", dest_dir)


    def wait_for_completion(self, outfile: str) -> bool:
        """
        Wait for the completion of defect calculation.
        Returns False if the job is cancelled due to timeout or insensible energy,
        True if it finishes normally.
        """
        start_time = time.time()
        time.sleep(10)
        slurm_id = monitor.get_slurm_id()
        pass

        # Wait for the outfile to be created
        while not monitor.exist(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                pass
                subprocess.run(["scancel", slurm_id])
                pass
                return False
            time.sleep(FILE_POLL_INTERVAL)

        # Wait for the simulation to finish
        while not monitor.has_finished(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                pass
                subprocess.run(["scancel", slurm_id])
                pass
                return False

            if input_parser.check_energy(outfile):
                pass
                subprocess.run(["scancel", slurm_id])
                pass
                return False

            time.sleep(FILE_POLL_INTERVAL)

        pass
        return True

    def gm_found(self, energies: dict[int, float]):
        if not energies:
            return False
        # Needs to modify such that the lowest energy is found greater than 10 times
        # Needs to caraefully think about the logic
        rounded_vals = [round(val,4) for val in energies.values()]
        min_energy = min(rounded_vals)
        count = rounded_vals.count(min_energy)
        return count >= 10

    def process_simulation_output(self, cycle_label: str, outfile: str):
        # Wait for simulation to finish and parse results
        start_time = time.time()
        time.sleep(10)
        slurm_id = monitor.get_slurm_id()
        pass

        # Wait for the outfile to be created
        while not monitor.exist(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                pass
                subprocess.run(["scancel", slurm_id])
                pass
                return
            time.sleep(FILE_POLL_INTERVAL)
        # Wait for the simulation to finish
        while not monitor.has_finished(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                pass
                subprocess.run(["scancel", slurm_id])
                pass
                return

            if input_parser.check_energy(outfile):
                pass
                subprocess.run(["scancel", slurm_id])
                pass
                return
            time.sleep(FILE_POLL_INTERVAL)


        pass
        energy_str = input_parser.get_defect_energy(outfile)
        if not energy_str:
            pass
            return
        if energy_str.startswith('**'):
            self.rejected_attempt += 1
            pass
            return
        new_energy = float(energy_str)
        if new_energy <= 0:
            self.rejected_attempt += 1
            pass
            return
        self.evaluate_and_update(cycle_label, new_energy, outfile)

    def evaluate_and_update(self, cycle_label: str, new_energy: float, outfile: str):
        reference_energy = 0. if self.is_first_cycle else self.best_energy
        delta_energy = new_energy - reference_energy
        gnorm = input_parser.get_gnorm(outfile)
        pass
        # 1) First cycle: accept outright
        if self.is_first_cycle:
            self.last_energy   = new_energy
            self.best_energy   = new_energy
            self.last_position = input_parser.get_r1_after(outfile)
            self.prev_output   = os.path.join(os.getcwd(), outfile)
            self.success_attempt += 1
            self.energy_log[cycle_label] = new_energy
            self.is_first_cycle = False
            pass
            self.write_xyz(cycle_label)
            return

        # Check for cases that lead to an immediate rejection:
        if (not self.is_first_cycle and monitor.same_energy(new_energy, self.best_energy, ENERGY_TOLERANCE)):
            self.reject_move(cycle_label, "Global minimum unchanged.")
            # Here energy must be added to log for counting GM found, same below
            self.energy_log[cycle_label] = new_energy
            return
        elif (not self.is_first_cycle and monitor.same_energy(new_energy, self.last_energy, ENERGY_TOLERANCE)):
            self.reject_move(cycle_label, "Local minimum unchanged.")
            self.energy_log[cycle_label] = new_energy
            return
        elif gnorm >= GNORM_THRESHOLD:
            self.reject_move(cycle_label, "High gradient norm; unreliable result.")
            return
        elif (not self.is_first_cycle and any(math.isclose(energy, new_energy, abs_tol = 0.01) for energy in self.energy_log.values())):
            self.reject_move(cycle_label, "Energy has been seen before.")
            self.energy_log[cycle_label] = new_energy
            return

        # For moves that increase the energy (delta_energy > 0), reject straight away
        elif not self.is_first_cycle and delta_energy > 0.:
            # Retrieve the simulation temperature from configuration, with a default value (e.g., 1.0)
            # T = self.config.get("temperature", 1.0)
            # metropolis_prob = np.exp(-delta_energy / T)
            # random_value = np.random.random()
            # pass

            # if random_value >= metropolis_prob:
            #     self.reject_move(cycle_label,
            #                     f"Delta energy positive and move rejected by Metropolis criterion (probability {metropolis_prob:.3f}).")
            #     return
            # else:
            #     pass
            #     # Here only accepted steps are added into the energy_log
            #     self.energy_log[cycle_label] = new_energy
            self.reject_move(cycle_label, f"Found a higher local minimum, rejected")
            self.energy_log[cycle_label] = new_energy
        # If no rejection condition is met, accept the move.
        else:
            self._accept_move(
                cycle_label, new_energy, outfile,
                f"Move accepted with energy {new_energy}."
            )
            # pass
            # self.last_energy = new_energy
            # self.best_energy = new_energy if new_energy < self.best_energy else self.best_energy
            # self.last_position = input_parser.get_r1_after(outfile)
            # self.prev_output = os.path.join(os.getcwd(), outfile)
            # self.success_attempt += 1
            # self.energy_log[cycle_label] = new_energy
            # self.is_first_cycle = False



    def reject_move(self, cycle_label: str, reason: str):
        self.rejected_attempt += 1
        pass

    def adjust_step_size(self):
        current_acceptance_ratio = self.success_attempt / (self.total_attempt + 1)
        if current_acceptance_ratio > 0.5:
            self.current_step_size *= 0.9
            pass
        else:
            self.current_step_size *= 1.1
            pass

    def generate_ranking_report(self):
        sorted_energy = dict(sorted(self.energy_log.items(), key=lambda item: item[1]))
        ranking_file = os.path.join(self.bh_dir, "rankings.txt")
        with open(ranking_file, 'w') as report_file:
            report_file.write("Energy trajectories for Basin Hopping moves\n")
            report_file.write("=" * 80 + "\n")
            for i, (label, energy_value) in enumerate(sorted_energy.items(), start=1):
                report_file.write(f"{i:<3} {label:<16} {energy_value:>8}\n")

    def run(self):
        cycle = self.start_cycle
        while not self.terminate:
            suceess = False
            cycle_label = str(cycle)
            try:
                self.run_simulation_cycle(cycle_label)
                pass
                if not self.fixed_step_size:
                    if int(cycle_label) >= 3 and int(cycle_label) % 4 == 0:
                        self.adjust_step_size()
                self.generate_ranking_report()
                self.write_unique_rounded_energies()
                self.save_checkpoint(cycle)
                suceess = True
            except TimeoutError as e:
                pass
                # self.save_checkpoint()
                self.terminate = True
                break
            # Add any termination condition check here
            if self.total_attempt == self.config.get("max_cycles"):
                pass
                self.terminate = True
                break
            # if self.gm_found(self.energy_log):
            #     pass
            #     self.terminate = True
            #     break
            if suceess:
                cycle += 1

    def run_simulation_cycle(self, cycle_label: str):
        parent_dir = os.getcwd()
        cycle_dir = self.setup_cycle_directory(cycle_label)
        os.chdir(cycle_dir)

        # Generate structure and prepare simulation input
        structure, center = self.generate_structure()
        input_filename = f"{cycle_label}.gin"
        output_filename = os.path.join(cycle_dir, f"{cycle_label}.gout")
        monte_carlo_util.write_input(input_filename, structure, center)

        # Submit gulp job
        run_gulp.gulp_submit()
        self.process_simulation_output(cycle_label, output_filename)
        os.chdir(parent_dir)

    def write_unique_rounded_energies(self):
        from collections import defaultdict
        energies = self.energy_log
        # Round values to 4 decimal places
        rounded_energy_dict = {k: round(v, 4) for k, v in energies.items()}

        # Build a reverse mapping: rounded_value -> list of keys
        value_to_keys = defaultdict(list)
        for key, rounded_val in rounded_energy_dict.items():
            value_to_keys[rounded_val].append(key)

        # Sort the unique rounded energies
        sorted_unique_values = sorted(value_to_keys.items())

        # Write output to file
        filename = os.path.join(self.bh_dir, "unique_energies.txt")
        with open(filename, 'w') as f:
            f.write("key,value,duplicate_keys,count\n")
            for value, keys in sorted_unique_values:
                primary_key = keys[0]
                duplicate_keys = keys[1:]  # Others that map to the same value
                f.write(f"{primary_key},{value},{duplicate_keys},{len(duplicate_keys)}\n")

    def write_xyz(self, cycle_label: str):
        """Append the current self.last_position as one XYZ frame."""
        filename = os.path.join(self.bh_dir, 'trajectory.xyz')
        atoms = self.last_position
        # If the file already exists and isn't empty, insert a blank line to separate frames
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            sep = "\n"
        else:
            sep = ""
        with open(filename, 'a') as f:
            # 1) optional blank separator
            # f.write(sep)
            # 2) number of atoms
            ncore = sum(1 for atom in atoms if atom.type=='cor')
            f.write(f"{ncore}\n")
            # 3) comment line (timestamp is useful)
            f.write(f"Frame {cycle_label}\n")
            # 4) atom lines
            for atom in atoms:
                if atom.type == 'cor':
                    f.write(f"{atom.label} {atom.x:.6f} {atom.y:.6f} {atom.z:.6f}\n")

    def _accept_move(self, cycle_label, new_energy, outfile, msg):
        self.last_energy   = new_energy
        self.best_energy   = min(new_energy, self.best_energy)
        self.last_position = input_parser.get_r1_after(outfile)
        self.prev_output   = os.path.join(os.getcwd(), outfile)
        self.success_attempt += 1
        self.energy_log[cycle_label] = new_energy
        pass
        self.write_xyz(cycle_label)

