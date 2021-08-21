import numpy as np
import pandas as pd
import pytest

from Program.Symmetry.molecule import Molecule


@pytest.fixture()
def water():
    water_mol = ['962\n', '  Marvin  12300703363D          \n', '\n', '  3  2  0  0  0  0            999 V2000\n',
                 '   -0.2309   -0.3265    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '    0.7484   -0.2843    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '   -0.5175    0.6108    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '  1  2  1  0  0  0  0\n', '  1  3  1  0  0  0  0\n', 'M  END\n', '\n', '> <StdInChI>\n',
                 'InChI=1S/H2O/h1H2\n', '\n', '> <StdInChIKey>\n', 'XLYOFNOQVPJJNP-UHFFFAOYSA-N\n', '\n',
                 '> <AuxInfo>\n', '1/0/N:1/rA:3nOHH/rB:s1;s1;/rC:-.2309,-.3265,0;.7484,-.2843,0;-.5175,.6108,0;\n',
                 '\n', '> <Formula>\n', 'H2 O\n', '\n', '> <Mw>\n', '18.01528\n', '\n', '> <SMILES>\n', 'O([H])[H]\n',
                 '\n', '> <CSID>\n', '937\n', '\n', '$$$$\n']
    return Molecule(water_mol)


def test_init_natoms(water):
    assert water.natoms == 3


def test_init_nbonds(water):
    assert water.nbonds == 2


def test_init_chiral(water):
    assert water.chiral is False


def test_init_structure(water):
    structure = pd.DataFrame([[-0.2309, -0.3265, 0.0],
                              [0.7484, -0.2843, 0.0],
                              [-0.5175, 0.6108, 0.0]])
    structure.index = ['O', 'H', 'H']
    structure = structure.rename(columns={0: 'x', 1: 'y', 2: 'z'})
    assert water.structure.equals(structure)


def test_init_bonds(water):
    bonds = pd.DataFrame([[1, 2, 1, 0],
                          [1, 3, 1, 0]],
                         ['O-H', 'O-H'],
                         ['Atom1', 'Atom2', 'Bond type', 'Stereochemistry'])
    assert (water.bonds.iloc[0] == bonds.iloc[0]).all()
    assert (water.bonds.iloc[1] == bonds.iloc[1]).all()
    assert (water.bonds.index == bonds.index).all()


def test_init_faulty_input_non_numeric_counts_block_handled(capsys):
    water_mol = ['962\n', '  Marvin  12300703363D          \n', '\n', '  X  2  0  0  0  0            999 V2000\n',
                 '   -0.2309   -0.3265    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '    0.7484   -0.2843    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '   -0.5175    0.6108    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '  1  2  1  0  0  0  0\n', '  1  3  1  0  0  0  0\n', 'M  END\n', '\n', '> <StdInChI>\n',
                 'InChI=1S/H2O/h1H2\n', '\n', '> <StdInChIKey>\n', 'XLYOFNOQVPJJNP-UHFFFAOYSA-N\n', '\n',
                 '> <AuxInfo>\n', '1/0/N:1/rA:3nOHH/rB:s1;s1;/rC:-.2309,-.3265,0;.7484,-.2843,0;-.5175,.6108,0;\n',
                 '\n', '> <Formula>\n', 'H2 O\n', '\n', '> <Mw>\n', '18.01528\n', '\n', '> <SMILES>\n', 'O([H])[H]\n',
                 '\n', '> <CSID>\n', '937\n', '\n', '$$$$\n']
    Molecule(water_mol)
    captured = capsys.readouterr()
    assert captured.out == ''
