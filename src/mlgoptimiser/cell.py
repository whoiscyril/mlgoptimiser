# from globals import GlobalVariables
import numpy as np

from .ml import Mott_Littleton


class Cell:
    def __init__(self) -> None:
        self.a = None
        self.b = None
        self.c = None
        self.alpha = None
        self.beta = None
        self.gamma = None

    def __str__(self) -> str:
        return f"{self.a}  {self.b}  {self.c} {self.alpha}  {self.beta} {self.gamma} \n"

    def user_input(self, a, b, c, alpha, beta, gamma) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_cell_after(self) -> "Cell":
        # filename = gbi.goutfile
        filename = "input/master.gout"
        temp = []
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("  Final cell parameters and derivatives"):
                    for _ in range(2):
                        line = f.readline()
                    for _ in range(6):
                        line = f.readline()
                        parts = line.strip().split()
                        temp.append(float(parts[1]))
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = temp
        return self

    def get_cell_before(self) -> "Cell":
        gbi = GlobalVariables()
        gbi.initialise()
        filename = gbi.goutfile
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("  Cell parameters (Angstroms/Degrees)"):
                    for _ in range(2):
                        line = f.readline()
                    self.a = float(line.strip().split()[2])
                    self.alpha = line.strip().split()[5]
                    line = f.readline()
                    self.b = float(line.strip().split()[2])
                    self.beta = line.strip().split()[5]
                    line = f.readline()
                    self.c = float(line.strip().split()[2])
                    self.gamma = line.strip().split()[5]
        return self

    def const_to_vecs(self) -> np.ndarray:
        try:
            alpha_rad = np.radians(float(self.alpha))
            beta_rad = np.radians(float(self.beta))
            gamma_rad = np.radians(float(self.gamma))
        except (TypeError, ValueError):
            # Handle the case where alpha, beta, or gamma are not numeric
            raise ValueError("Alpha, beta, and gamma must be numeric")

        vec_a = np.array([self.a, 0, 0])
        vec_b = np.array([self.b * np.cos(gamma_rad), self.b * np.sin(gamma_rad), 0])
        vec_cx = self.c * np.cos(beta_rad)
        vec_cy = (
            self.c
            * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad))
            / np.sin(gamma_rad)
        )
        vec_cz = np.sqrt(self.c**2 - vec_cx**2 - vec_cy**2)
        vec_c = np.array([vec_cx, vec_cy, vec_cz])
        return np.array([vec_a, vec_b, vec_c])

    def get_cell(self) -> "Cell":
        ml = Mott_Littleton()
        filename = ml.config_file
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("BEGIN_CELL"):
                    line = f.readline()
                    parts = line.strip().split()
        self.a = parts[0]
        self.b = parts[1]
        self.c = parts[2]
        self.alpha = parts[3]
        self.beta = parts[4]
        self.gamma = parts[5]
        return self
