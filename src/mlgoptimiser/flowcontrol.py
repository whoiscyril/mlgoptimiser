import glob
import os
import pathlib
import shutil
import subprocess

from line_profiler import LineProfiler

from . import algorithms, input_parser, run_gulp
from .atom import Atom
# from basin_hopping_fixed import BasinHoppingSimulator
from .basin_hopping import BasinHoppingSimulator
from .basin_hopping_unoptimized import BasinHoppingUnoptimized
from .cell import Cell
from .globals import GlobalOptimisation
from .simulated_annealing import SimulatedAnnealingSimulator
from .logging_config import get_auto_logger


def initialise():
    """Initialize the MLGOptimiser system by running GULP calculations and organizing files."""
    logger = get_auto_logger()
    logger.info("Starting MLGOptimiser initialization")
    
    # Run GULP geo_opt
    go = GlobalOptimisation()
    logger.info(f"Loaded global configuration, algorithm: {go.algorithm}")

    gulp_path = go.gulp_path
    logger.info(f"Using GULP executable at: {gulp_path}")
    
    # Write and run energy input
    logger.info("Writing energy input file")
    run_gulp.write_energy_input()
    
    logger.info("Running GULP master calculation")
    result = subprocess.run([gulp_path, "master"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"GULP master calculation failed with return code {result.returncode}")
        logger.error(f"STDERR: {result.stderr}")
    else:
        logger.info("GULP master calculation completed successfully")
    
    logger.info("Copying master.gout to input directory")
    subprocess.run(["cp", "master.gout", "input"])
    
    # Write and run defect dummy calculation
    logger.info("Writing defect dummy input file")
    run_gulp.write_defect_dummy()
    
    logger.info("Running GULP defect calculation")
    result = subprocess.run([gulp_path, "defect"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"GULP defect calculation failed with return code {result.returncode}")
        logger.error(f"STDERR: {result.stderr}")
    else:
        logger.info("GULP defect calculation completed successfully")

    # Find all .res files in the current directory
    logger.debug("Searching for .res files to organize")
    res_files = glob.glob("*.res")
    lib_files = glob.glob("*.res")  # This line seems redundant, as it's the same as res_files
    logger.info(f"Found {len(res_files)} .res files to move")

    # List of files to be moved into 'input' directory
    files_to_move = res_files + lib_files + ["master.gout"]
    logger.debug(f"Files to move to input directory: {files_to_move}")

    # Move each relevant file to the 'input' directory
    moved_count = 0
    for file in files_to_move:
        if pathlib.Path(file).exists():
            logger.debug(f"Moving {file} to input directory")
            subprocess.run(["mv", file, "input"])
            moved_count += 1
        else:
            logger.warning(f"File {file} not found, skipping move")
    
    logger.info(f"Moved {moved_count} files to input directory")

    # Files to unlink (delete)
    files_to_unlink = ["master.gin", "defect.gin", "defect.gout"]
    logger.debug(f"Files to delete: {files_to_unlink}")

    # Unlink (delete) the specified files if they exist
    deleted_count = 0
    for file in files_to_unlink:
        file_path = pathlib.Path(file)
        if file_path.exists():
            logger.debug(f"Deleting temporary file: {file}")
            file_path.unlink()
            deleted_count += 1
        else:
            logger.debug(f"Temporary file {file} not found, skipping deletion")
    
    logger.info(f"Cleaned up {deleted_count} temporary files")
    logger.info("MLGOptimiser initialization completed successfully")


def execute():
    """Execute the selected optimization algorithm."""
    logger = get_auto_logger()
    logger.info("Starting algorithm execution phase")
    
    go = GlobalOptimisation()
    algo = go.algorithm
    logger.info(f"Selected algorithm: {algo}")
    
    try:
        if algo.lower() == "mc":
            logger.info("Starting Monte Carlo algorithm")
            logger.info("Enabling line profiler for performance analysis")
            profiler = LineProfiler()
            profiler.add_function(algorithms.monte_carlo)
            profiler.enable()
            
            algorithms.monte_carlo()
            
            profiler.disable()
            logger.info("Monte Carlo algorithm completed, printing profiler stats")
            profiler.print_stats()
            
        elif algo.lower() == "sa":
            logger.info("Starting Simulated Annealing algorithm")
            logger.info("Initializing SimulatedAnnealingSimulator with step_size=2.0")
            simulator = SimulatedAnnealingSimulator(step_size=2.0)
            simulator.run()
            logger.info("Simulated Annealing algorithm completed")
            
        elif algo.lower() == "bh":
            logger.info("Starting Basin Hopping algorithm")
            # Configure Basin Hopping parameters
            bh_config = {
                "temperature": 0.2,           # kT in eV - controls Metropolis acceptance of uphill moves
                "max_cycles": 1000,
                "default_step_size": 2.0,      # Initial step size in Angstroms
                "target_acceptance": 0.4,      # Target 40% acceptance for optimal exploration
                "duplicate_energy_tol": 0.01,  # Energy tolerance for duplicate detection (eV)
                "duplicate_rmsd_tol": 0.1      # RMSD tolerance for duplicate detection (Angstroms)
            }
            logger.info(f"Basin Hopping config: temperature={bh_config['temperature']} eV, "
                       f"max_cycles={bh_config['max_cycles']}, step_size={bh_config['default_step_size']} Å, "
                       f"target_acceptance={bh_config['target_acceptance']}, adaptive step size enabled, "
                       f"duplicate detection enabled (E_tol={bh_config['duplicate_energy_tol']} eV, "
                       f"RMSD_tol={bh_config['duplicate_rmsd_tol']} Å)")
            simulator = BasinHoppingSimulator(
                fixed_step_size=False,   # Enable adaptive step size
                step_size=1.5,           # Initial step size
                config=bh_config
            )
            simulator.run()
            logger.info("Basin Hopping algorithm completed")
            
        elif algo.lower() == "bh_unopt":
            logger.info("Starting Basin Hopping (Unoptimized Variant) algorithm")
            # Configure Basin Hopping Unoptimized parameters
            bh_config = {
                "temperature": 1.0,      # kT in eV - controls Metropolis acceptance
                "max_cycles": 1000,
                "default_step_size": 1.0  # Fixed step size for this variant
            }
            logger.info(f"Basin Hopping Unoptimized config: temperature={bh_config['temperature']} eV, "
                       f"max_cycles={bh_config['max_cycles']}, fixed step size (explores perturbed positions)")
            logger.warning("This variant explores UNOPTIMIZED (perturbed) positions - experimental!")
            simulator = BasinHoppingUnoptimized(
                step_size=1.0,           # Fixed step size
                config=bh_config
            )
            simulator.run()
            logger.info("Basin Hopping (Unoptimized) algorithm completed")

        elif algo.lower() == "sphere":
            logger.info("Starting Sphere algorithm")
            algorithms.sphere()
            logger.info("Sphere algorithm completed")

        else:
            logger.error(f"Unknown algorithm specified: {algo}")
            raise ValueError(f"Unsupported algorithm: {algo}")
            
    except Exception as e:
        logger.error(f"Algorithm execution failed: {e}")
        logger.error(f"Algorithm: {algo}")
        raise
    
    logger.info("Algorithm execution phase completed successfully")
