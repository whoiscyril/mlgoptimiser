# from globals import GlobalVariables
# class MC:
#     def __init__(self) -> None:
#         gbi = GlobalVariables()
#         gbi.initialise()
#         infile = gbi.infile
#         with open(infile, 'r') as f:
#             for line in f:
#                 if line.startswith('mc_param'):
#                     for line in f:
#                         if line.startswith('step'):
#                             line = f.readline()
#                             self.step = int(line.strip().split()[-1])    