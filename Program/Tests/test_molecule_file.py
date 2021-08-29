import numpy as np

from Program.Symmetry.MolParser import Molecule


def test_init_water():
    water = Molecule('937')
    assert water._natoms == 3