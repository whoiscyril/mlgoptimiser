import sys

from . import flowcontrol, input_parser
from .globals import GlobalOptimisation


def main():
    flowcontrol.initialise()
    flowcontrol.execute()


if __name__ == "__main__":
    main()
