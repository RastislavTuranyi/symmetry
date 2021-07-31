"""
Unit tests for symmetry.py.
"""

import pytest
import pandas as pd

from symmetry import PointGroup


@pytest.fixture()
def create_cs():
    Cs = pd.DataFrame([[1, 1],
                       [1, 1],
                       [1, -1]])
    Cs.index = ['Cs', "A'", "A''"]
    return Cs


@pytest.fixture()
def c2v():
    C2v = pd.DataFrame([[1, 1, 1, 1],
                        [+1, +1, +1, +1],
                        [+1, +1, -1, -1],
                        [+1, -1, +1, -1],
                        [+1, -1, -1, +1]
                        ])
    C2v.index = ['C2v', 'A1', 'A2', 'B1', 'B2']
    return C2v


@pytest.fixture()
def point_group(c2v):
    C2v = PointGroup(c2v)
    return C2v


def test_create_using_str(create_cs):
    Cs = PointGroup('Cs')
    assert Cs.character_table.equals(create_cs) is True


def test_create_using_df(create_cs):
    Cs = PointGroup(create_cs)
    assert Cs.character_table.equals(create_cs) is True


def test_create_with_wrong_point_group(capsys):
    PointGroup('gg')
    captured = capsys.readouterr()
    assert captured.out[:61] == 'The requested point group is not supported, please try again.'


def test_create_with_wrong_type(capsys):
    PointGroup([1, 5])
    captured = capsys.readouterr()
    assert captured.out[:59] == 'Incorrect type has been used to create a PointGroup object.'


def test_repr_as_expected(point_group):
    assert repr(point_group) == 'PointGroup(C2v)'


def test_no_overwrite_character_table(c2v):
    Cs = PointGroup('Cs')
    with pytest.raises(AttributeError):
        Cs.character_table = c2v


def test_reduction_correct(point_group):
    result = point_group.reduction([4, 0, 4, 0])
    expected = pd.DataFrame([[4, 0, 4, 0],
                             [4, 0, -4, 0],
                             [4, 0, 4, 0],
                             [4, 0, -4, 0], ])
    expected['number of appearances'] = [2, 0, 2, 0]
    assert (result.iloc[0] == expected.iloc[0]).all
    assert (result.iloc[1] == expected.iloc[1]).all
    assert (result.iloc[2] == expected.iloc[2]).all
    assert (result.iloc[3] == expected.iloc[3]).all


def test_reduction_with_wrong_number_of_elements_in_reducible_representation(point_group, capsys):
    point_group.reduction([4, 0, 4, 0, 5])
    captured = capsys.readouterr()
    assert captured.out[:98] == 'The reducible representation must have the same number of ' \
                                'elements as a row of the character table'


def test_reduction_with_str_input(point_group, capsys):
    point_group.reduction([4, '0', 'four', 0])
    captured = capsys.readouterr()
    assert captured.out[:69] == 'The reducible representation contains elements of unsupported dtypes.'


def test_constituents_returns_correctly(point_group):
    result = point_group.constituents([4, 0, 4, 0])
    assert result.tolist() == [2, 0, 2, 0]


def test_constituents_prints_correctly(point_group, capsys):
    point_group.constituents([4, 0, 4, 0])
    captured = capsys.readouterr()
    assert captured.out[:17] == '2 × A1  +  2 × B1'
