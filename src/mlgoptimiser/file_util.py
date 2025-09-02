# import subprocess
# from globals import get_global_variables
# import shutil
# def move_files(dest_dir):
#     gbi = get_global_variables()
#     subprocess.run(['cp', gbi.infile, dest_dir])
#     subprocess.run(['cp', gbi.gresfile, dest_dir])
#     subprocess.run(['cp', gbi.libfile, dest_dir])
#     shutil.copy('../master.gout', dest_dir)
#     subprocess.run(['cp', '../gulp.sh', dest_dir])