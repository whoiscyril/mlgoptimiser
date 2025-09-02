class Defect:
    def __init__(self) -> None:
        self.label = None
        self.type = None
        self.atom2 = None
        self.x = None
        self.y = None
        self.z = None
        self.fix = None

    def __str__(self) -> str:
        atom2_str = f", Atom2: {self.atom2}" if self.type == "impurity" else ""
        return f"Label: {self.label}, Type: {self.type}{atom2_str}, X: {self.x}, Y: {self.y}, Z: {self.z}, Fix: {self.fix}"
