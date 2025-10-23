"""
Basin Hopping variant that explores unoptimized (perturbed) positions.

This variant:
- Uses GULP minimization for ENERGY EVALUATION only
- Compares minimized energies for acceptance (Metropolis on transformed landscape)
- BUT stores and explores PERTURBED (unoptimized) positions
- Fixed step size (no adaptation)

Differences from standard Basin Hopping:
- Walker moves on full configuration space, not just basin minima
- Explores different entry paths into basins
- May be less stable but could escape difficult basins
"""

import os
from . import input_parser
from .basin_hopping import BasinHoppingSimulator


class BasinHoppingUnoptimized(BasinHoppingSimulator):
    """
    Basin Hopping variant exploring unoptimized positions.

    Inherits from BasinHoppingSimulator but overrides position update
    to store perturbed structures instead of minimized ones.
    """

    def __init__(
        self,
        input_dir: str = "input",
        base_dir: str = None,
        config: dict = None,
        step_size: float = None
    ):
        """
        Initialize unoptimized Basin Hopping simulator.

        Note: fixed_step_size is ALWAYS True for this variant.
        """
        # Force fixed step size for this variant
        super().__init__(
            input_dir=input_dir,
            base_dir=base_dir,
            config=config,
            step_size=step_size,
            fixed_step_size=True  # Always fixed for this variant
        )

        # Change output directory to distinguish from standard BH
        self.bh_dir = os.path.join(self.base_dir, 'bh_unoptimized')
        os.makedirs(self.bh_dir, exist_ok=True)

        # Update checkpoint path
        self.checkpoint_file = os.path.join(self.bh_dir, 'checkpoint.pkl')
        if os.path.exists(self.checkpoint_file):
            self.load_checkpoint()

    def run_simulation_cycle(self, cycle_label: str):
        """
        Run one Basin Hopping cycle, passing perturbed structure through.

        Overrides parent to pass perturbed structure to evaluation.
        """
        parent_dir = os.getcwd()
        cycle_dir = self.setup_cycle_directory(cycle_label)
        os.chdir(cycle_dir)

        # Generate perturbed structure
        from . import monte_carlo_util
        structure, center = self.generate_structure()

        # Write GULP input with perturbed structure
        input_filename = f"{cycle_label}.gin"
        output_filename = os.path.join(cycle_dir, f"{cycle_label}.gout")
        monte_carlo_util.write_input(input_filename, structure, center)

        # Submit GULP job (will minimize for energy evaluation)
        from . import run_gulp
        run_gulp.gulp_submit()

        # Process output, passing perturbed structure
        self.process_simulation_output(cycle_label, output_filename, structure, center)

        os.chdir(parent_dir)

    def process_simulation_output(self, cycle_label: str, outfile: str,
                                  perturbed_structure, perturbed_center):
        """
        Process GULP output, passing perturbed structure for storage.

        Overrides parent to pass perturbed structure to acceptance evaluation.

        Args:
            cycle_label: Cycle identifier
            outfile: GULP output file path
            perturbed_structure: The perturbed (unoptimized) atom positions
            perturbed_center: The perturbed defect center
        """
        import time
        import subprocess
        from . import monitor, input_parser

        # Wait for simulation to finish and parse results
        # (Same monitoring logic as parent)
        start_time = time.time()
        time.sleep(10)
        slurm_id = monitor.get_slurm_id()

        # Wait for the outfile to be created
        while not monitor.exist(outfile):
            if time.time() - start_time > 240:  # TIMEOUT_LIMIT
                subprocess.run(["scancel", slurm_id])
                return
            time.sleep(5)  # FILE_POLL_INTERVAL

        # Wait for the simulation to finish
        while not monitor.has_finished(outfile):
            if time.time() - start_time > 240:
                subprocess.run(["scancel", slurm_id])
                return

            if input_parser.check_energy(outfile):
                subprocess.run(["scancel", slurm_id])
                return
            time.sleep(5)

        # Extract minimized energy (for acceptance decision)
        energy_str = input_parser.get_defect_energy(outfile)
        if not energy_str:
            return
        if energy_str.startswith('**'):
            self.rejected_attempt += 1
            return

        new_energy = float(energy_str)
        if new_energy <= 0:
            self.rejected_attempt += 1
            return

        # Pass both minimized energy AND perturbed structure
        self.evaluate_and_update(cycle_label, new_energy, outfile,
                                perturbed_structure, perturbed_center)

    def evaluate_and_update(self, cycle_label: str, new_energy: float, outfile: str,
                           perturbed_structure, perturbed_center):
        """
        Evaluate acceptance using minimized energy, but store perturbed positions.

        Overrides parent to pass perturbed structure to _accept_move.

        Args:
            cycle_label: Cycle identifier
            new_energy: Minimized energy from GULP (for acceptance)
            outfile: GULP output file
            perturbed_structure: Perturbed (unoptimized) positions to store if accepted
            perturbed_center: Perturbed defect center
        """
        import numpy as np
        from . import input_parser, monitor

        # Get gradient norm for quality check
        gnorm = input_parser.get_gnorm(outfile)

        # ===== QUALITY CHECK =====
        if gnorm >= 0.01:  # GNORM_THRESHOLD
            self.reject_move(cycle_label, f"High gradient norm ({gnorm:.4f}).")
            return

        # ===== FIRST CYCLE =====
        if self.is_first_cycle:
            # Store PERTURBED positions, not minimized!
            self.last_energy = new_energy  # Energy from minimization
            self.best_energy = new_energy
            self.last_position = perturbed_structure  # Perturbed positions!
            self.last_center = perturbed_center  # Perturbed center!
            self.prev_output = os.path.join(os.getcwd(), outfile)
            self.success_attempt += 1
            self.energy_log[cycle_label] = new_energy
            self.is_first_cycle = False
            self.write_xyz(cycle_label)
            return

        # ===== CALCULATE ENERGY DIFFERENCE =====
        # Compare minimized energies (transformed landscape)
        delta_energy = new_energy - self.last_energy

        # ===== BASIN HOPPING METROPOLIS =====
        if delta_energy < 0:
            # Downhill - always accept
            self._accept_move(
                cycle_label, new_energy, outfile,
                perturbed_structure, perturbed_center,
                f"Downhill move accepted (ΔE={delta_energy:.4f} eV)."
            )
        else:
            # Uphill - Metropolis criterion
            T = self.config.get("temperature", 1.0)
            metropolis_prob = np.exp(-delta_energy / T)
            random_value = np.random.random()

            if random_value < metropolis_prob:
                # Accept uphill move
                self.uphill_accepted += 1
                self._accept_move(
                    cycle_label, new_energy, outfile,
                    perturbed_structure, perturbed_center,
                    f"Uphill accepted via Metropolis (ΔE={delta_energy:.4f} eV, P={metropolis_prob:.3f})."
                )
            else:
                # Reject uphill move
                self.reject_move(
                    cycle_label,
                    f"Uphill rejected by Metropolis (ΔE={delta_energy:.4f} eV, P={metropolis_prob:.3f})."
                )
                self.energy_log[cycle_label] = new_energy

    def _accept_move(self, cycle_label, new_energy, outfile,
                    perturbed_structure, perturbed_center, msg):
        """
        Accept move and update walker to PERTURBED (unoptimized) position.

        Overrides parent to store perturbed positions instead of minimized.

        Key difference: self.last_position = perturbed, not minimized!
        """
        # Update energies (from minimized structure)
        self.last_energy = new_energy
        self.best_energy = min(new_energy, self.best_energy)

        # Store PERTURBED positions (NOT minimized!)
        self.last_position = perturbed_structure
        self.last_center = perturbed_center

        # Still track minimized output for reference
        self.prev_output = os.path.join(os.getcwd(), outfile)

        # Update counters
        self.success_attempt += 1
        self.energy_log[cycle_label] = new_energy

        # Write trajectory (will write perturbed positions)
        self.write_xyz(cycle_label)

    def generate_structure(self):
        """
        Generate perturbed structure from current walker position.

        Note: self.last_position now contains PERTURBED coords, not minimized!
        """
        from . import monte_carlo_util, input_parser

        if self.is_first_cycle:
            if self.seed is not None:
                structure = self.seed
                _, center = monte_carlo_util.monte_carlo_generator()
            else:
                structure, center = monte_carlo_util.monte_carlo_generator()
            self.total_attempt += 1
        else:
            # Perturb from UNOPTIMIZED position
            # Need to use last_center instead of extracting from prev_output
            if hasattr(self, 'last_center'):
                center = self.last_center
            else:
                # Fallback: extract from output (shouldn't happen)
                center_array = input_parser.get_dcenter_from_out(self.prev_output)
                from .defect import Defect
                center = Defect()
                center.x, center.y, center.z = center_array

            structure, _ = monte_carlo_util.bh_move_inter_only(
                self.last_position,  # Now perturbed positions!
                self.prev_output,
                self.current_step_size
            )
            self.total_attempt += 1

        return structure, center

    def save_checkpoint(self, cycle: int):
        """
        Save checkpoint including perturbed positions.

        Extends parent to also save perturbed center.
        """
        import pickle

        state = {
            'cycle': cycle,
            'attempt_counter': self.attempt_counter,
            'total_attempt': self.total_attempt,
            'success_attempt': self.success_attempt,
            'rejected_attempt': self.rejected_attempt,
            'uphill_accepted': self.uphill_accepted,
            'stable_energy_count': self.stable_energy_count,
            'best_energy': self.best_energy,
            'best_dir': self.best_dir,
            'current_step_size': self.current_step_size,
            'is_first_cycle': self.is_first_cycle,
            'energy_log': self.energy_log,
            'last_energy': getattr(self, 'last_energy', None),
            'last_position': getattr(self, 'last_position', None),
            'last_center': getattr(self, 'last_center', None),  # NEW
            'prev_output': getattr(self, 'prev_output', None)
        }
        with open(self.checkpoint_file, 'wb') as cp:
            pickle.dump(state, cp)

    def load_checkpoint(self):
        """
        Load checkpoint including perturbed center.

        Extends parent to also load perturbed center.
        """
        import pickle
        from . import input_parser

        with open(self.checkpoint_file, 'rb') as cp:
            state = pickle.load(cp)

        self.attempt_counter = state['attempt_counter']
        self.total_attempt = state['total_attempt']
        self.success_attempt = state['success_attempt']
        self.rejected_attempt = state['rejected_attempt']
        self.uphill_accepted = state.get('uphill_accepted', 0)
        self.stable_energy_count = state['stable_energy_count']
        self.best_energy = state['best_energy']
        self.best_dir = state['best_dir']
        self.current_step_size = state['current_step_size']
        self.is_first_cycle = state['is_first_cycle']
        self.energy_log = state['energy_log']
        self.last_energy = state.get('last_energy')
        self.last_position = state.get('last_position')
        self.last_center = state.get('last_center')  # NEW
        self.prev_output = state.get('prev_output')
        self.seed = input_parser.get_seed_coords(os.path.join(self.input_dir, 'seed.coord'))
        self.start_cycle = state['cycle'] + 1
