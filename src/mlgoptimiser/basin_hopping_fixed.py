import logging
import math
import os
import shutil
import subprocess
import time
from collections import Counter
from typing import Dict, Tuple

from . import input_parser
from . import monitor
from . import monte_carlo_util
import numpy as np
from . import run_gulp

DEFAULT_STEP_SIZE_RANGE = (0.2, 1.0)
ENERGY_TOLERANCE = 1e-4
GNORM_THRESHOLD = 0.01
FILE_POLL_INTERVAL = 5
TIMEOUT_LIMIT = 240


class BasinHoppingSimulator:
    # Class-level default configuration
    DEFAULT_CONFIG: Dict[str, float] = {
        "max_cycles": 1000,  # Maximum number of cycles to run.
    }

    def __init__(
        self, input_dir: str = "input", base_dir: str = os.getcwd(), config: Dict = None
    ):
        self.input_dir = input_dir
        self.base_dir = base_dir
        self.bh_dir = os.path.join(self.base_dir, "bh_test")
        os.makedirs(self.bh_dir, exist_ok=True)
        self.config = (
            config
            if config is not None
            else BasinHoppingSimulator.DEFAULT_CONFIG.copy()
        )
        self.setup_logging()
        self.reset_state()

    def setup_logging(self):
        logging.basicConfig(
            filename=os.path.join(self.bh_dir, "simulation.log"),
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def reset_state(self):
        # Initialize counters and simulation parameters
        self.attempt_counter = 1
        self.total_attempt = 0
        self.success_attempt = 0
        self.rejected_attempt = 0
        self.stable_energy_count = 0
        self.best_energy = float("inf")
        self.best_dir = ""
        # self.current_step_size = np.random.uniform(*DEFAULT_STEP_SIZE_RANGE)
        self.current_step_size = 1.0
        self.is_first_cycle = True
        self.energy_log = {}  # Maps cycle: energy
        self.terminate = False

    def setup_cycle_directory(self, cycle_label: str) -> str:
        cycle_dir = os.path.join(self.bh_dir, cycle_label)
        os.makedirs(cycle_dir, exist_ok=True)
        # Use our dedicated function to copy input files and scripts
        self.copy_input_files(cycle_dir)
        return cycle_dir

    def generate_structure(self) -> Tuple[object, object]:
        # Depending on the cycle, choose the appropriate structure generation:
        if self.is_first_cycle:
            logging.info("Using Monte Carlo generator for initial structure.")
            structure, center = monte_carlo_util.monte_carlo_generator()
        else:
            logging.info("Using basin hopping move.")
            try:
                structure, center = monte_carlo_util.bh_move_inter_only(
                    self.last_position, self.prev_output, self.current_step_size
                )
                self.total_attempt += 1
            except RuntimeError as e:
                logging.info("Unable to generate structure for this step")
                return
        logging.debug(f"Current step size: {self.current_step_size}")
        return structure, center

    def copy_input_files(self, dest_dir: str):
        # Assume input_dir exists in self.base_dir; copy it to dest_dir
        shutil.copytree(
            self.input_dir, os.path.join(dest_dir, "input"), dirs_exist_ok=True
        )
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
        logging.info(f"The current SLURM job ID is {slurm_id}")

        # Wait for the outfile to be created
        while not monitor.exist(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                logging.error("Timeout reached waiting for the output file.")
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return False
            time.sleep(FILE_POLL_INTERVAL)

        # Wait for the simulation to finish
        while not monitor.has_finished(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                logging.error("Time limit reached, cancelling job.")
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return False

            if input_parser.check_energy(outfile):
                logging.warning(
                    "Insensible energy detected; cancelling job and skipping this cycle."
                )
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return False

            time.sleep(FILE_POLL_INTERVAL)

        logging.info("Finished calculating outfile.")
        return True

    def gm_found(self, energies: dict[int, float]):
        if not energies:
            return False
        # Needs to modify such that the lowest energy is found greater than 10 times
        # Needs to caraefully think about the logic
        rounded_vals = [round(val, 4) for val in energies.values()]
        min_energy = min(rounded_vals)
        count = rounded_vals.count(min_energy)
        return count >= 10

    def process_simulation_output(self, cycle_label: str, outfile: str):
        # Wait for simulation to finish and parse results
        start_time = time.time()
        time.sleep(10)
        slurm_id = monitor.get_slurm_id()
        logging.info(f"The current SLURM job ID is {slurm_id}")

        # Wait for the outfile to be created
        while not monitor.exist(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                logging.error("Timeout reached waiting for the output file.")
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return
            time.sleep(FILE_POLL_INTERVAL)
        # Wait for the simulation to finish
        while not monitor.has_finished(outfile):
            if time.time() - start_time > TIMEOUT_LIMIT:
                logging.error("Time limit reached, cancelling job.")
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return

            if input_parser.check_energy(outfile):
                logging.warning(
                    "Insensible energy detected; cancelling job and skipping this cycle."
                )
                subprocess.run(["scancel", slurm_id])
                logging.info(f"Cancelling job: {slurm_id}")
                return
            time.sleep(FILE_POLL_INTERVAL)

        logging.info("Finished calculating outfile.")
        energy_str = input_parser.get_defect_energy(outfile)
        if not energy_str:
            logging.info("Will return due to no final energy")
            return
        if energy_str.startswith("**"):
            self.rejected_attempt += 1
            logging.warning(
                f"Cycle {cycle_label}: Unsensible energy. Skipping to next cycle."
            )
            return
        new_energy = float(energy_str)
        if new_energy <= 0:
            self.rejected_attempt += 1
            logging.info(
                f"Cycle {cycle_label}: Negative or zero energy encountered. Skipping cycle."
            )
            return
        self.evaluate_and_update(cycle_label, new_energy, outfile)

    def evaluate_and_update(self, cycle_label: str, new_energy: float, outfile: str):
        # Reject new energies that are close to the current energy and accept energies that are different to the current energy
        # Last position is changed to the position of the new structure after quench
        reference_energy = 0.0 if self.is_first_cycle else self.best_energy
        delta_energy = new_energy - reference_energy
        gnorm = input_parser.get_gnorm(outfile)
        logging.debug(
            f"Cycle {cycle_label} - Energy: {new_energy}, DeltaE: {delta_energy}, Gnorm: {gnorm}"
        )

        # Check for cases that lead to an immediate rejection:
        if not self.is_first_cycle and monitor.same_energy(
            new_energy, self.best_energy, ENERGY_TOLERANCE
        ):
            self.reject_move(cycle_label, "Global minimum unchanged.")
            # Here energy must be added to log for counting GM found, same below
            self.energy_log[cycle_label] = new_energy
            return
        elif not self.is_first_cycle and monitor.same_energy(
            new_energy, self.last_energy, ENERGY_TOLERANCE
        ):
            self.reject_move(cycle_label, "Local minimum unchanged.")
            self.energy_log[cycle_label] = new_energy
            return
        elif gnorm >= GNORM_THRESHOLD:
            self.reject_move(cycle_label, "High gradient norm; unreliable result.")
            return
        elif not self.is_first_cycle and any(
            math.isclose(energy, new_energy, abs_tol=0.01)
            for energy in self.energy_log.values()
        ):
            self.reject_move(cycle_label, "Energy has been seen before.")
            self.energy_log[cycle_label] = new_energy
            return

        # If no rejection condition is met, accept the move.
        logging.info(
            f"Cycle {cycle_label}: Found new LM, Move accepted with energy {new_energy}."
        )
        self.last_energy = new_energy
        self.best_energy = (
            new_energy if new_energy < self.best_energy else self.best_energy
        )
        self.last_position = input_parser.get_r1_after(outfile)
        self.prev_output = os.path.join(os.getcwd(), outfile)
        self.success_attempt += 1
        self.energy_log[cycle_label] = new_energy
        # if int(cycle_label) >= 10:
        #     self.adjust_step_size()
        self.is_first_cycle = False

    def reject_move(self, cycle_label: str, reason: str):
        self.rejected_attempt += 1
        logging.warning(f"Cycle {cycle_label}: Move rejected. Reason: {reason}")

    def adjust_step_size(self):
        current_acceptance_ratio = self.success_attempt / (self.total_attempt + 1)
        if self.total_attempt >= 10:
            if current_acceptance_ratio > 0.5:
                self.current_step_size *= 0.9
                logging.info(f"Adjusted step size: {self.current_step_size}")
            else:
                self.current_step_size *= 1.1
                logging.info(f"Adjusted step size: {self.current_step_size}")

    def generate_ranking_report(self):
        sorted_energy = dict(sorted(self.energy_log.items(), key=lambda item: item[1]))
        ranking_file = os.path.join(self.bh_dir, "rankings.txt")
        with open(ranking_file, "w") as report_file:
            report_file.write("Energy trajectories for Basin Hopping moves\n")
            report_file.write("=" * 80 + "\n")
            for i, (label, energy_value) in enumerate(sorted_energy.items(), start=1):
                report_file.write(f"{i:<3} {label:<16} {energy_value:>8}\n")

    def run(self):
        cycle = 0
        while not self.terminate:
            cycle_label = str(cycle)
            try:
                self.run_simulation_cycle(cycle_label)
                self.generate_ranking_report()
                self.write_unique_rounded_energies()
                # logging.info(f"The current acceptance ratio is {self.success_attempt / (self.total_attempt + 1)}")
            except TimeoutError as e:
                logging.error("Terminating simulation due to timeout.")
                self.terminate = True
                break
            # Add any termination condition check here
            if self.total_attempt == 500:
                logging.info("Terminated as total attempt threshold met")
                self.terminate = True
                break
            # if self.gm_found(self.energy_log):
            #     logging.info("Terminated as GM found at least 10 times")
            #     self.terminate = True
            #     break
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
        with open(filename, "w") as f:
            f.write("key,value,duplicate_keys,count\n")
            for value, keys in sorted_unique_values:
                primary_key = keys[0]
                duplicate_keys = keys[1:]  # Others that map to the same value
                f.write(
                    f"{primary_key},{value},{duplicate_keys},{len(duplicate_keys)}\n"
                )
