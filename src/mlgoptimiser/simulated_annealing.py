import os
import shutil
import subprocess
import time
import matplotlib.pyplot as plt
import math
import logging
import numpy as np
from typing import Tuple, Dict
from collections import Counter
import pickle
import random

from . import monte_carlo_util
from . import run_gulp
from . import monitor
from . import input_parser


DEFAULT_STEP_SIZE_RANGE = (0.2, 1.0)
ENERGY_TOLERANCE = 1e-4
GNORM_THRESHOLD = 0.01
FILE_POLL_INTERVAL = 5
TIMEOUT_LIMIT = 240

class SimulatedAnnealingSimulator:
    # Class-level default configuration
    DEFAULT_CONFIG = {
        "temperature": 3000,
        "max_cycles": 5000,
        "default_step_size": 1.
    }

    def __init__(
        self,
        input_dir: str = "input",
        base_dir: str = None,
        config: Dict = None,
        step_size: float = None,
    ):
        # Configure output path
        # root = Path(base_dir or Path.cwd())
        # self.bh_dir = root / run_id
        # self.bh_dir.mkdir(parents=True, exist_ok=True)
        # self.out = OutputManager(self.bh_dir)
        # Merge Configs
        cfg = {**SimulatedAnnealingSimulator.DEFAULT_CONFIG, **(config or {})}
        self.config = cfg
        self.temperature = cfg["temperature"]

        # Paths and directories
        self.base_dir  = base_dir or os.getcwd()
        if os.path.isabs(input_dir):
            self.input_dir = input_dir
        else:
            self.input_dir = os.path.join(self.base_dir, input_dir)

        self.bh_dir = os.path.join(self.base_dir, 'sa_test')
        os.makedirs(self.bh_dir, exist_ok=True)

        # Simulation config
        self.config = config.copy() if config is not None else SimulatedAnnealingSimulator.DEFAULT_CONFIG.copy()

        # Step-size control flags
        self.step_size_arg = step_size

        # Initialize logging and state
        self.setup_logging()
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

        # Quenched energies
        self.quenched_energies = {}

    def setup_logging(self):
        logging.basicConfig(
            filename=os.path.join(self.bh_dir, "simulation.log"),
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def reset_state(self):
        # Initialize counters and simulation parameters
        self.attempt_counter = 1
        self.total_attempt = 0
        self.success_attempt = 0
        self.rejected_attempt = 0
        self.stable_energy_count = 0
        self.best_energy = float('inf')
        self.best_dir = ""
        self.is_first_cycle = True
        self.energy_log = {}
        self.terminate = False

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
        logging.info(f"Loaded checkpoint. Resuming from cycle {self.start_cycle}")

    def run(self):
        cycle = self.start_cycle
        while not self.terminate:
            success = False
            cycle_label = str(cycle)
            try:
                self.run_simulation_cycle(cycle_label)
                logging.info(f"Total attempt: {self.total_attempt}, Accepted attempt: {self.success_attempt}, temperature: {self.temperature}")
                if cycle != 0 and int(cycle) % 50 == 0:
                    # do quench
                    logging.info(f"Carrying out a quench run at {cycle_label}")
                    self.quench(cycle_label)
                    self.write_quenched_energies()
                self.generate_ranking_report()
                self.write_unique_rounded_energies()
                self.save_checkpoint(cycle)
                success = True
                if cycle != 0 and int(cycle) % 100 == 0:
                    self.temperature *= 0.9
                    logging.info(f"Changing the temperature to {self.temperature}")
            except TimeoutError:
                logging.info(f"The current acceptance ratio is {self.success_attempt / (self.total_attempt + 1)}")
                logging.error("Terminating simulation due to timeout.")
                self.terminate = True
                break
            if self.total_attempt == self.config.get("max_cycles"):
                logging.info("Terminated as total attempt threshold met")
                self.terminate = True
                break
            if success:
                cycle += 1

    def run_simulation_cycle(self, cycle_label: str):
        parent_dir = os.getcwd()
        cycle_dir = self.setup_cycle_directory(cycle_label)
        os.chdir(cycle_dir)

        structure, center = self.generate_structure()
        input_filename = f"{cycle_label}.gin"
        output_filename = os.path.join(cycle_dir, f"{cycle_label}.gout")
        res_filename = os.path.join(cycle_dir, "1.res")
        monte_carlo_util.write_input_sa(input_filename, structure, center)

        run_gulp.gulp_submit()
        self.process_simulation_output(cycle_label, output_filename, res_filename)
        os.chdir(parent_dir)

    def setup_cycle_directory(self, cycle_label: str) -> str:
        cycle_dir = os.path.join(self.bh_dir, cycle_label)
        os.makedirs(cycle_dir, exist_ok=True)
        self.copy_input_files(cycle_dir)
        return cycle_dir

    def copy_input_files(self, dest_dir: str):
        shutil.copytree(self.input_dir, os.path.join(dest_dir, 'input'), dirs_exist_ok=True)
        src_gulp = os.path.join(self.base_dir, 'gulp.sh')
        if not os.path.isfile(src_gulp):
            raise FileNotFoundError(f"gulp.sh not found at: {src_gulp!r}")
        shutil.copy(src_gulp, dest_dir)

    def generate_structure(self) -> Tuple[object, object]:
        if self.is_first_cycle:
            logging.info("Using Monte Carlo generator for initial structure.")
            if self.seed is not None:
                structure = self.seed
                _, center = monte_carlo_util.monte_carlo_generator()
            else:
                structure, center = monte_carlo_util.monte_carlo_generator()
            self.total_attempt += 1
        else:
            try:
                rn = random.random()
                if rn < 0.5:
                    logging.info("Applying standard monte carlo move.")
                    structure, center = monte_carlo_util.bh_move_inter_only(
                        self.last_position, self.prev_output, self.current_step_size
                    )
                else:
                    logging.info("Applying bimodal monte carlo move")
                    structure, center = monte_carlo_util.sa_bimodal(
                        self.last_position, self.prev_output, self.current_step_size
                        )
                self.total_attempt += 1
            except RuntimeError:
                logging.warning(f"Move failed ({e}); reusing last structure.")
                return
        logging.debug(f"Current step size: {self.current_step_size}")
        return structure, center

    def process_quench_output(self, outfile: str, cycle_label: str):
        start_time = time.time()
        time.sleep(10)
        slurm_id = monitor.get_slurm_id()
        logging.info(f"The current SLURM job ID is {slurm_id}")

        while not monitor.exist(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                logging.error("Timeout reached waiting for the output file.")
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return
            time.sleep(FILE_POLL_INTERVAL)

        while not monitor.has_finished(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                logging.error("Time limit reached, cancelling job.")
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return
            if input_parser.check_energy(outfile):
                logging.warning("Insensible energy detected; cancelling job and skipping this cycle.")
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return
            time.sleep(FILE_POLL_INTERVAL)

        logging.info("Finished calculating outfile.")
        energy_str = input_parser.get_defect_energy(outfile)
        if not energy_str:
            logging.info("Will return due to no final energy")
            return
        if energy_str.startswith('**'):
            self.rejected_attempt += 1
            logging.warning(f"Cycle {cycle_label}: Unsensible energy. Skipping to next cycle.")
            return
        new_energy = float(energy_str)
        if new_energy <= 0:
            self.rejected_attempt += 1
            logging.info(f"Cycle {cycle_label}: Negative or zero energy encountered. Skipping cycle.")
            return
        return energy_str


    def process_simulation_output(self, cycle_label: str, outfile: str, resfile: str):
        start_time = time.time()
        time.sleep(10)
        slurm_id = monitor.get_slurm_id()
        logging.info(f"The current SLURM job ID is {slurm_id}")

        while not monitor.exist(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                logging.error("Timeout reached waiting for the output file.")
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return
            time.sleep(FILE_POLL_INTERVAL)

        while not monitor.has_finished(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                logging.error("Time limit reached, cancelling job.")
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return
            if input_parser.check_energy(outfile):
                logging.warning("Insensible energy detected; cancelling job and skipping this cycle.")
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return
            time.sleep(FILE_POLL_INTERVAL)

        logging.info("Finished calculating outfile.")
        energy_str = input_parser.get_defect_energy_single(outfile)
        if not energy_str:
            logging.info("Will return due to no final energy")
            return
        if energy_str.startswith('**'):
            self.rejected_attempt += 1
            logging.warning(f"Cycle {cycle_label}: Unsensible energy. Skipping to next cycle.")
            return
        new_energy = float(energy_str)
        if new_energy <= 0:
            self.rejected_attempt += 1
            logging.info(f"Cycle {cycle_label}: Negative or zero energy encountered. Skipping cycle.")
            return
        self.evaluate_and_update(cycle_label, new_energy, outfile, resfile)

    def evaluate_and_update(self, cycle_label: str, new_energy: float, outfile: str, resfile: str):
        gnorm = input_parser.get_gnorm(outfile)
        logging.debug(f"Cycle {cycle_label} - Energy: {new_energy}, Gnorm: {gnorm}")

        # 1) First cycle: accept outright
        if self.is_first_cycle:
            self.last_energy   = new_energy
            self.best_energy   = new_energy
            self.last_position = input_parser.get_r1_after_from_res(resfile)
            self.prev_output   = os.path.join(os.getcwd(), outfile)
            self.success_attempt += 1
            self.energy_log[cycle_label] = new_energy
            self.is_first_cycle = False
            logging.info(f"Cycle {cycle_label}: First cycle accepted with energy {new_energy}.")
            self.write_xyz(cycle_label)
            return

        # 2) Now for all later cycles:
        reference_energy = self.last_energy
        delta_energy     = new_energy - reference_energy
        logging.debug(f"ΔE = {new_energy:.6f} - {reference_energy:.6f} = {delta_energy:.6f}")

        # a) identical-energy step
        if monitor.same_energy(new_energy, self.last_energy, ENERGY_TOLERANCE):
            self._accept_move(cycle_label, new_energy, outfile, resfile,
                              f"Identical energies; accepted at {new_energy}.")
            return

        # b) downhill step (ΔE < 0)
        if delta_energy < 0:
            self._accept_move(cycle_label, new_energy, outfile, resfile,
                              f"Downhill move accepted with energy {new_energy}.")
            return

        # c) too-noisy result
        if gnorm >= GNORM_THRESHOLD:
            self.reject_move(cycle_label, "High gradient norm; unreliable result.")
            return

        # d) uphill step → Metropolis criterion
        T = self.temperature
        to_eV = 8.617333e-5  
        p_met = np.exp(-delta_energy / (to_eV * T))
        r     = np.random.random()
        logging.debug(f"Metropolis p={p_met:.3f}, r={r:.3f}")

        if r >= p_met:
            self.reject_move(
                cycle_label,
                f"ΔE positive ({delta_energy:.3f}) and rejected by Metropolis (p={p_met:.3f})."
            )
            self.energy_log[cycle_label] = new_energy
        else:
            self._accept_move(
                cycle_label, new_energy, outfile, resfile,
                f"Accepted uphill step by Metropolis (p={p_met:.3f})."
            )


    def reject_move(self, cycle_label: str, reason: str):
        self.rejected_attempt += 1
        logging.warning(f"Cycle {cycle_label}: Move rejected. Reason: {reason}")

    def _accept_move(self, cycle_label, new_energy, outfile, resfile, msg):
        self.last_energy   = new_energy
        self.best_energy   = min(new_energy, self.best_energy)
        self.last_position = input_parser.get_r1_after_from_res(resfile)
        self.prev_output   = os.path.join(os.getcwd(), outfile)
        self.success_attempt += 1
        self.energy_log[cycle_label] = new_energy
        logging.info(f"Cycle {cycle_label}: {msg}")
        self.write_xyz(cycle_label)


    def adjust_step_size(self):
        current_acceptance_ratio = self.success_attempt / (self.total_attempt + 1)
        if current_acceptance_ratio > 0.5:
            self.current_step_size *= 0.9
            logging.info(f"Current acceptance ratio: {current_acceptance_ratio}, Adjusted step size: {self.current_step_size}")
        else:
            self.current_step_size *= 1.1
            logging.info(f"Current acceptance ratio: {current_acceptance_ratio}, Adjusted step size: {self.current_step_size}")

    def generate_ranking_report(self):
        sorted_energy = dict(sorted(self.energy_log.items(), key=lambda item: item[1]))
        ranking_file = os.path.join(self.bh_dir, "rankings.txt")
        with open(ranking_file, 'w') as report_file:
            report_file.write("Energy trajectories for Basin Hopping moves\n")
            report_file.write("=" * 80 + "\n")
            for i, (label, energy_value) in enumerate(sorted_energy.items(), start=1):
                report_file.write(f"{i:<3} {label:<16} {energy_value:>8}\n")

    def write_unique_rounded_energies(self):
        energies = self.energy_log
        filename = os.path.join(self.bh_dir, "energies.txt")
        with open(filename, 'w') as f:
            f.write("directory,energy\n")
            for directory, energy in sorted(energies.items()):
                f.write(f"{directory},{energy}\n")

    def write_quenched_energies(self):
        energies = self.quenched_energies
        filename = os.path.join(self.bh_dir, "quenched_energies.txt")
        with open(filename, 'w') as f:
            f.write("directory,energy\n")
            for directory, energy in sorted(energies.items(), key=lambda item: int(item[0])):
                f.write(f"{directory},{energy}\n")


        # directories = list(energies.keys())
        # energy_values = list(energies.values())

        # plt.figure(figsize=(10, 6))
        # plt.scatter(directories, energy_values)
        # plt.xlabel('Directory')
        # plt.ylabel('Energy')
        # plt.title('Scatter Plot of Directory Energies')
        # plt.xticks(rotation=90)
        # plt.tight_layout()

        # plot_filename = os.path.join(self.bh_dir, "energies_plot.png")
        # plt.savefig(plot_filename)
        # plt.close()


    # def write_unique_rounded_energies(self):
    #     from collections import defaultdict
    #     energies = self.energy_log
    #     rounded_energy_dict = {k: round(v, 4) for k, v in energies.items()}
    #     value_to_keys = defaultdict(list)
    #     for key, rounded_val in rounded_energy_dict.items():
    #         value_to_keys[rounded_val].append(key)
    #     sorted_unique_values = sorted(value_to_keys.items())
    #     filename = os.path.join(self.bh_dir, "unique_energies.txt")
    #     with open(filename, 'w') as f:
    #         f.write("key,value,duplicate_keys,count\n")
    #         for value, keys in sorted_unique_values:
    #             primary_key = keys[0]
    #             duplicate_keys = keys[1:]
    #             f.write(f"{primary_key},{value},{duplicate_keys},{len(duplicate_keys)}\n")

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
        logging.info(f"Checkpoint saved at cycle {cycle}")

    def gm_found(self, energies: dict[int, float]):
        if not energies:
            return False
        rounded_vals = [round(val,4) for val in energies.values()]
        min_energy = min(rounded_vals)
        count = rounded_vals.count(min_energy)
        return count >= 10 

    def quench(self, cycle_label: str):
        quench_root = os.path.join(self.bh_dir, f"{cycle_label}_quench")
        os.makedirs(quench_root, exist_ok=True)

        # 1) Copy input directory
        src_input = os.path.abspath(self.input_dir)
        shutil.copytree(src_input, os.path.join(quench_root, "input"), dirs_exist_ok=True)

        # 2) Copy gulp.sh
        shutil.copy(
            os.path.join(self.base_dir, "gulp.sh"),
            os.path.join(quench_root, "gulp.sh")
        )

        # 3) Switch into quench directory
        os.chdir(quench_root)

        # 4) Write the .gin, submit, process output, etc.
        structure = self.last_position
        centre = input_parser.get_dcenter_from_out(self.prev_output)
        input_filename  = f"{cycle_label}_quench.gin"
        output_filename = f"{cycle_label}_quench.gout"
        monte_carlo_util.write_input(input_filename, structure, centre)

        run_gulp.gulp_submit()
        energy = self.process_quench_output(output_filename, os.path.basename(quench_root))
        self.quenched_energies[cycle_label] = energy

        logging.info(f"Cycle {cycle_label}: Quenched energy recorded: {energy}")

        # 5) Return to bh_dir
        os.chdir(self.bh_dir)

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

