"""Tests for symmetry that involve files"""

import pandas as pd
import pytest

from Program.Symmetry.symmetry import PointGroup


@pytest.fixture()
def Cs():
    cs = pd.DataFrame([[1, 1, None, None],
                       [1, 1, 'x, y, $R_z$', '$x^2$, $y^2$, $z^2$, xy'],
                       [1, -1, 'z, $R_x$, $R_y$', 'xz, yz']])
    cs.index = ['Cs', "A'", "A''"]
    cs = cs.rename(columns={0: 'E', 1: '$σ_h$', 2: 'h', 3: '= 2'})
    return cs


def test_init_with_filename_full_character_table(Cs):
    cs = PointGroup('Cs')
    assert cs.full_character_table.equals(Cs)


def test_init_with_filename_character_table():
    cs = PointGroup('Cs')
    expected = pd.DataFrame([[1, 1],
                             [1, 1],
                             [1, -1]])
    expected.index = ['Cs', "A'", "A''"]
    expected = expected.rename(columns={0: 'E', 1: '$σ_h$'})
    assert cs.character_table.equals(expected)


def test_init_with_path_full_character_table(Cs):
    cs = PointGroup(r'C:\Users\TACHYON\Documents\GitHub\symmetry\Program\Data\CSV\Cs.csv')
    assert cs.full_character_table.equals(Cs)


def test_init_with_path_character_table():
    cs = PointGroup(r'C:\Users\TACHYON\Documents\GitHub\symmetry\Program\Data\CSV\Cs.csv')
    expected = pd.DataFrame([[1, 1],
                             [1, 1],
                             [1, -1]])
    expected.index = ['Cs', "A'", "A''"]
    expected = expected.rename(columns={0: 'E', 1: '$σ_h$'})
    assert cs.character_table.equals(expected)


def test_init_with_wrong_path_handled(capsys):
    cs = PointGroup('gg')
    captured = capsys.readouterr()
    assert captured.out[:55] == 'FileNotFoundError: Provided csv file could not be found'
