import numpy as np
import pandas as pd
import pytest

from Program.Symmetry.MolParser import MolParser, CorruptedFileWarning


############################################################################################
# TESTS FOR CREATING MOLECULE OBJECT
############################################################################################
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
    return MolParser(water_mol)


@pytest.fixture()
def corrupted_natoms():
    water_mol = ['962\n', '  Marvin  12300703363D          \n', '\n', '  X  2  0  0  0  0            999 V2000\n',
                 '   -0.2309   -0.3265    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '    0.7484   -0.2843    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '   -0.5175    0.6108    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '  1  2  1  0  0  0  0\n', '  1  3  1  0  0  0  0\n', 'M  END\n', '\n', '> <StdInChI>\n',
                 'InChI=1S/H2O/h1H2\n', '\n', '> <StdInChIKey>\n', 'XLYOFNOQVPJJNP-UHFFFAOYSA-N\n', '\n',
                 '> <AuxInfo>\n', '1/0/N:1/rA:3nOHH/rB:s1;s1;/rC:-.2309,-.3265,0;.7484,-.2843,0;-.5175,.6108,0;\n',
                 '\n', '> <Formula>\n', 'H2 O\n', '\n', '> <Mw>\n', '18.01528\n', '\n', '> <SMILES>\n', 'O([H])[H]\n',
                 '\n', '> <CSID>\n', '937\n', '\n', '$$$$\n']
    return water_mol


@pytest.fixture()
def corrupted_nbonds():
    water_mol = ['962\n', '  Marvin  12300703363D          \n', '\n', '  3  X  0  0  0  0            999 V2000\n',
                 '   -0.2309   -0.3265    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '    0.7484   -0.2843    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '   -0.5175    0.6108    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '  1  2  1  0  0  0  0\n', '  1  3  1  0  0  0  0\n', 'M  END\n', '\n', '> <StdInChI>\n',
                 'InChI=1S/H2O/h1H2\n', '\n', '> <StdInChIKey>\n', 'XLYOFNOQVPJJNP-UHFFFAOYSA-N\n', '\n',
                 '> <AuxInfo>\n', '1/0/N:1/rA:3nOHH/rB:s1;s1;/rC:-.2309,-.3265,0;.7484,-.2843,0;-.5175,.6108,0;\n',
                 '\n', '> <Formula>\n', 'H2 O\n', '\n', '> <Mw>\n', '18.01528\n', '\n', '> <SMILES>\n', 'O([H])[H]\n',
                 '\n', '> <CSID>\n', '937\n', '\n', '$$$$\n']
    return water_mol


@pytest.fixture()
def corrupted_chirality():
    water_mol = ['962\n', '  Marvin  12300703363D          \n', '\n', '  3  2  0  2  0  0            999 V2000\n',
                 '   -0.2309   -0.3265    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '    0.7484   -0.2843    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '   -0.5175    0.6108    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '  1  2  1  0  0  0  0\n', '  1  3  1  0  0  0  0\n', 'M  END\n', '\n', '> <StdInChI>\n',
                 'InChI=1S/H2O/h1H2\n', '\n', '> <StdInChIKey>\n', 'XLYOFNOQVPJJNP-UHFFFAOYSA-N\n', '\n',
                 '> <AuxInfo>\n', '1/0/N:1/rA:3nOHH/rB:s1;s1;/rC:-.2309,-.3265,0;.7484,-.2843,0;-.5175,.6108,0;\n',
                 '\n', '> <Formula>\n', 'H2 O\n', '\n', '> <Mw>\n', '18.01528\n', '\n', '> <SMILES>\n', 'O([H])[H]\n',
                 '\n', '> <CSID>\n', '937\n', '\n', '$$$$\n']
    return water_mol


@pytest.fixture()
def correct_structure():
    structure = pd.DataFrame([[-0.2309, -0.3265, 0.0],
                              [0.7484, -0.2843, 0.0],
                              [-0.5175, 0.6108, 0.0]])
    structure.index = ['O', 'H', 'H']
    structure = structure.rename(columns={0: 'x', 1: 'y', 2: 'z'})
    return structure


@pytest.fixture()
def correct_bonds():
    bonds = pd.DataFrame([[1, 2, 1, 0],
                          [1, 3, 1, 0]],
                         ['O-H', 'O-H'],
                         ['Atom1', 'Atom2', 'Bond type', 'Stereochemistry'])
    return bonds


# PERFECT INPUT
def test_init_natoms(water):
    assert water.natoms == 3


def test_init_nbonds(water):
    assert water.nbonds == 2


def test_init_chiral(water):
    assert water.chiral is False


def test_init_structure(water, correct_structure):
    assert water.structure.equals(correct_structure)


def test_init_bonds(water, correct_bonds):
    assert (water.bonds.iloc[0] == correct_bonds.iloc[0]).all()
    assert (water.bonds.iloc[1] == correct_bonds.iloc[1]).all()
    assert (water.bonds.index == correct_bonds.index).all()


# NUMBER OF ATOMS BROKEN
def test_init_faulty_input_non_numeric_natoms_correct_warnings(corrupted_natoms):
    with pytest.warns(CorruptedFileWarning) as record:
        MolParser(corrupted_natoms)
    assert record[0].message.args[0][:32] == 'The counts block of the provided'
    assert record[1].message.args[0][:44] == "The molecule's number of atoms was estimated"


def test_init_faulty_input_non_numeric_natoms_guess_correct(corrupted_natoms):
    water = MolParser(corrupted_natoms)
    assert water.natoms == 3


def test_init_faulty_input_non_numeric_natoms_impact_other_parsers_values(corrupted_natoms, correct_structure,
                                                                          correct_bonds):
    water = MolParser(corrupted_natoms)
    assert water.nbonds == 2
    assert water.chiral is False
    assert water.structure.equals(correct_structure)
    assert (water.bonds.iloc[0] == correct_bonds.iloc[0]).all()
    assert (water.bonds.iloc[1] == correct_bonds.iloc[1]).all()
    assert (water.bonds.index == correct_bonds.index).all()


# NUMBER OF BONDS BROKEN
def test_init_faulty_input_non_numeric_nbonds_gives_warning(corrupted_nbonds):
    with pytest.warns(CorruptedFileWarning) as record:
        MolParser(corrupted_nbonds)
    assert record[0].message.args[0][:32] == 'The counts block of the provided'
    assert record[1].message.args[0][:44] == "The molecule's number of bonds was estimated"


def test_init_faulty_input_non_numeric_nbonds_guess_correct(corrupted_nbonds):
    water = MolParser(corrupted_nbonds)
    assert water.nbonds == 2


def test_init_faulty_input_non_numeric_nbonds_values_set_correctly(corrupted_nbonds, correct_structure, correct_bonds):
    water = MolParser(corrupted_nbonds)
    assert water.natoms == 3
    assert water.chiral is False
    assert water.structure.equals(correct_structure)
    assert (water.bonds.iloc[0] == correct_bonds.iloc[0]).all()
    assert (water.bonds.iloc[1] == correct_bonds.iloc[1]).all()
    assert (water.bonds.index == correct_bonds.index).all()


# CHIRALITY BROKEN
def test_init_faulty_input_chirality_not_0_1_warnings(corrupted_chirality):
    with pytest.warns(CorruptedFileWarning, match='The chirality in the provided'):
        MolParser(corrupted_chirality)


def test_init_faulty_input_chirality_not_0_1_values_set_correctly(corrupted_chirality):
    water = MolParser(corrupted_chirality)
    assert water.chiral is None


# ATOMS BLOCK BROKEN
def test_init_faulty_input_non_numeric_coordinates_atoms_block_handled():
    water_mol = ['962\n', '  Marvin  12300703363D          \n', '\n', '  3  2  0  0  0  0            999 V2000\n',
                 '   -0.2309   -0.3265    X O   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '    0.7484   -0.2843    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '   -0.5175    0.6108    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '  1  2  1  0  0  0  0\n', '  1  3  1  0  0  0  0\n', 'M  END\n', '\n', '> <StdInChI>\n',
                 'InChI=1S/H2O/h1H2\n', '\n', '> <StdInChIKey>\n', 'XLYOFNOQVPJJNP-UHFFFAOYSA-N\n', '\n',
                 '> <AuxInfo>\n', '1/0/N:1/rA:3nOHH/rB:s1;s1;/rC:-.2309,-.3265,0;.7484,-.2843,0;-.5175,.6108,0;\n',
                 '\n', '> <Formula>\n', 'H2 O\n', '\n', '> <Mw>\n', '18.01528\n', '\n', '> <SMILES>\n', 'O([H])[H]\n',
                 '\n', '> <CSID>\n', '937\n', '\n', '$$$$\n']
    with pytest.warns(CorruptedFileWarning, match='The structure of the molecule'):
        MolParser(water_mol)


# BONDS BLOCK BROKEN
def test_init_faulty_input_non_numeric_values_bonds_block_handled():
    water_mol = ['962\n', '  Marvin  12300703363D          \n', '\n', '  3  2  0  0  0  0            999 V2000\n',
                 '   -0.2309   -0.3265    0.0 O   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '    0.7484   -0.2843    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '   -0.5175    0.6108    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n',
                 '  1  2  X  0  0  0  0\n', '  1  3  1  0  0  0  0\n', 'M  END\n', '\n', '> <StdInChI>\n',
                 'InChI=1S/H2O/h1H2\n', '\n', '> <StdInChIKey>\n', 'XLYOFNOQVPJJNP-UHFFFAOYSA-N\n', '\n',
                 '> <AuxInfo>\n', '1/0/N:1/rA:3nOHH/rB:s1;s1;/rC:-.2309,-.3265,0;.7484,-.2843,0;-.5175,.6108,0;\n',
                 '\n', '> <Formula>\n', 'H2 O\n', '\n', '> <Mw>\n', '18.01528\n', '\n', '> <SMILES>\n', 'O([H])[H]\n',
                 '\n', '> <CSID>\n', '937\n', '\n', '$$$$\n']
    with pytest.warns(CorruptedFileWarning, match='The bonds of the molecule could not'):
        MolParser(water_mol)
