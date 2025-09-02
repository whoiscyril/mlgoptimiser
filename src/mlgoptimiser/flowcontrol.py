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
from .cell import Cell
from .globals import GlobalOptimisation
from .simulated_annealing import SimulatedAnnealingSimulator


def initialise():
    # Run GULP geo_opt
    go = GlobalOptimisation()

    gulp_path = go.gulp_path
    # input_file = gbi.ginfile.removesuffix('.gin')
    run_gulp.write_energy_input()
    #    print('finished writing energy input')
    subprocess.run([gulp_path, "master"])
    subprocess.run(["cp", "master.gout", "input"])
    #     print('finished running master file')
    run_gulp.write_defect_dummy()
    # #    print('finished writing dummy defect input')
    subprocess.run([gulp_path, "defect"])
    #     print('finished running defect calc')

    # Find all .res files in the current directory
    res_files = glob.glob("*.res")
    lib_files = glob.glob(
        "*.res"
    )  # This line seems redundant, as it's the same as res_files

    # List of files to be moved into 'input' directory
    files_to_move = res_files + lib_files + ["master.gout"]

    # Move each relevant file to the 'input' directory
    for file in files_to_move:
        if pathlib.Path(file).exists():
            subprocess.run(["mv", file, "input"])

    # Files to unlink (delete)
    files_to_unlink = ["master.gin", "defect.gin", "defect.gout"]

    # Unlink (delete) the specified files if they exist
    for file in files_to_unlink:
        file_path = pathlib.Path(file)
        if file_path.exists():
            file_path.unlink()


def execute():
    go = GlobalOptimisation()
    # print("Finished initialising GO to find algorithms")
    # cell = Cell()
    # cell.get_cell_after()
    # atoms = Atom()
    # atom_list = atoms.get_atoms()
    # r1_list = atoms.get_r1_after()
    algo = go.algorithm
    # defects = input_parser.get_defect_list()
    # options = input_parser.get_options()
    if algo.lower() == "mc":
        profiler = LineProfiler()
        profiler.add_function(algorithms.monte_carlo)
        profiler.enable()
        algorithms.monte_carlo()
        profiler.disable()
        profiler.print_stats()
    elif algo.lower() == "sa":
        simulator = SimulatedAnnealingSimulator(step_size=2.0)
        simulator.run()
        # algorithms.simulated_annealing()
    elif algo.lower() == "bh":
        # algorithms.basin_hopping_schemeA()
        simulator = BasinHoppingSimulator(fixed_step_size=True, step_size=1.0)
        simulator.run()
    elif algo.lower() == "sphere":
        algorithms.sphere()
