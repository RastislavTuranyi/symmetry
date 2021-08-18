"""
Unit tests for symmetry.py.
"""

import pytest
import pandas as pd

from Program.Symmetry.symmetry import PointGroup


@pytest.fixture()
def create_cs():
    Cs = pd.DataFrame([[1, 1, None, None],
                       [1, 1, 'x, y, $R_z$', '$x^2$, $y^2$, $z^2$, xy'],
                       [1, -1, 'z, $R_x$, $R_y$', 'xz, yz']])
    Cs.index = ['Cs', "A'", "A''"]
    Cs = Cs.rename(columns={0: 'E', 1: '$σ_h$', 2: 'h', 3: '= 2'})
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


def test_create_with_wrong_type_handled(capsys):
    PointGroup([1, 5])
    captured = capsys.readouterr()
    assert 'TypeError: The provided point group' == captured.out[:35]


def test_create_from_pandas_correct(create_cs):
    cs = PointGroup.create_from_pandas('Cs')
    assert cs.full_character_table.equals(create_cs)


def test_create_from_pandas_with_unsupported_point_group_handled(capsys):
    PointGroup.create_from_pandas('gg')
    captured = capsys.readouterr()
    assert 'The requested point group is not supported, please try again.' == captured.out[:61]


def test_create_from_pandas_with_wrong_type_handled(capsys):
    PointGroup.create_from_pandas([1, 5])
    captured = capsys.readouterr()
    assert 'Incorrect type has been used to create a PointGroup object.' == captured.out[:59]


def test_init_with_chtable_with_no_function_columns():
    expected = pd.DataFrame([[1, 1],
                             [1, 1],
                             [1, -1]])
    expected.index = ['Cs', "A'", "A''"]
    expected = expected.rename(columns={0: 'E', 1: '$σ_h$'})
    cs = PointGroup(expected)
    assert cs.full_character_table.equals(expected)
    assert cs.character_table.equals(expected)


def test_init_with_chtable_with_one_function_column():
    expected = pd.DataFrame([[1, 1, None],
                             [1, 1, 'x, y'],
                             [1, -1, 'z']])
    expected.index = ['Cs', "A'", "A''"]
    expected = expected.rename(columns={0: 'E', 1: '$σ_h$', 2: 'h'})
    cs = PointGroup(expected)
    assert cs.full_character_table.equals(expected)
    assert cs.character_table.equals(expected.drop('h', 1))


def test_init_with_chtable_with_text_column_in_the_middle():
    expected = pd.DataFrame([[1, None, 1],
                             [1, 'x, y', 1],
                             [1, 'z', -1]])
    expected.index = ['Cs', "A'", "A''"]
    expected = expected.rename(columns={0: 'E', 1: 'h', 2: '$σ_h$'})
    cs = PointGroup(expected)
    assert cs.full_character_table.equals(expected)
    assert cs.character_table.equals(expected.drop('h', 1))


def test_repr_as_expected(point_group):
    assert 'PointGroup(C2v)' == repr(point_group)


def test_no_overwrite_character_table(point_group, c2v):
    with pytest.raises(AttributeError):
        point_group.character_table = c2v


def test_no_overwrite_full_character_table(point_group, c2v):
    with pytest.raises(AttributeError):
        point_group.full_character_table = c2v


def test_reduction_correct(point_group):
    result = point_group.reduction([4, 0, 4, 0])
    expected = pd.DataFrame([[4, 0, 4, 0],
                             [4, 0, -4, 0],
                             [4, 0, 4, 0],
                             [4, 0, -4, 0], ])
    expected['number of appearances'] = [2, 0, 2, 0]
    assert (expected.iloc[0] == result.iloc[0]).all
    assert (expected.iloc[1] == result.iloc[1]).all
    assert (expected.iloc[2] == result.iloc[2]).all
    assert (expected.iloc[3] == result.iloc[3]).all


def test_reduction_with_wrong_number_of_elements_in_reducible_representation_handled(point_group, capsys):
    point_group.reduction([4, 0, 4, 0, 5])
    captured = capsys.readouterr()
    assert 'ReductionError: The reducible representation must have the same number of elements as a row ' \
           'of the character table' == captured.out[:114]


def test_reduction_with_str_input_handled(point_group, capsys):
    point_group.reduction([4, '0', 'four', 0])
    captured = capsys.readouterr()
    assert 'The reducible representation contains elements of unsupported dtypes.' == captured.out[:69]


def test_constituents_returns_correctly(point_group):
    result = point_group.constituents([4, 0, 4, 0])
    assert result.tolist() == [2, 0, 2, 0]


def test_constituents_prints_correctly(point_group, capsys):
    point_group.constituents([4, 0, 4, 0])
    captured = capsys.readouterr()
    assert captured.out[:20] == 'result = 2A1  +  2B1'


def test_convolution_correct(point_group):
    result = point_group.convolution('A1', 'A2')
    assert result.to_list() == [1, 1, -1, -1]


def test_convolution_wrong_key_handled(point_group, capsys):
    point_group.convolution('A1', 'gg')
    captured = capsys.readouterr()
    assert 'The inputted irreducible representations do not exist in this character table.' == captured.out[:78]


def test_convolution_one_argument_raises_typeerror(point_group):
    with pytest.raises(TypeError):
        point_group.convolution('A1')


def test_match_representation_works_when_irreducible_is_provided(point_group):
    result = point_group.match_representation([1, 1, 1, 1])
    expected = pd.Series([1, 1, 1, 1], name='A1')
    assert result.equals(expected) is True


def test_match_representation_incorrect_representation_length_handled(point_group, capsys):
    point_group.match_representation([1, 1, 1, 1, 1])
    captured = capsys.readouterr()
    assert 'MatchRepresentationError: The reducible representation must have the same number of elements as a ' \
           'row of the character table' == captured.out[:124]


def test_match_representation_with_dict_doesnt_work_but_handled(point_group, capsys):
    result = point_group.match_representation({'A1': [1, 1, 1, 1]})
    captured = capsys.readouterr()
    assert 'MatchRepresentationError: The reducible representation must have the same number of elements as a row of ' \
           'the character table' == captured.out[:124]


def test_match_representation_works_when_reducible_is_provided(point_group):
    result = point_group.match_representation([2, 2, 2, 2])
    expected = pd.DataFrame([[2, 2, 2, 2],
                             [2, 2, -2, -2],
                             [2, -2, 2, -2],
                             [2, -2, -2, 2]])
    expected['number of appearances'] = [2, 0, 0, 0]
    expected.index = ['A1', 'A2', 'B1', 'B2']
    assert (expected.iloc[0] == result.iloc[0]).all
    assert (expected.iloc[1] == result.iloc[1]).all
    assert (expected.iloc[2] == result.iloc[2]).all
    assert (expected.iloc[3] == result.iloc[3]).all


def test_show_matched_representation_returns_correct(point_group):
    result = point_group.show_matched_representation([1, 1, 1, 1])
    expected = pd.Series([1, 1, 1, 1], name='A1')
    assert result.equals(expected) is True


def test_show_matched_representation_prints_correct_with_series(point_group, capsys):
    result = point_group.show_matched_representation(pd.Series([1, 1, 1, 1], name='A1'))
    captured = capsys.readouterr()
    assert captured.out[:7] == 'A1 = A1'


def test_show_matched_representation_prints_correct_with_list(point_group, capsys):
    point_group.show_matched_representation([1, 1, 1, 1])
    captured = capsys.readouterr()
    assert captured.out[:11] == 'result = A1'


def test_print_result_correct_with_dataframe_default_lhs(point_group, c2v, capsys):
    c2v['number of appearances'] = [0, 1, 1, 0, 0]
    point_group.print_result(c2v)
    captured = capsys.readouterr()
    assert captured.out[:18] == 'result = A1  +  A2'


def test_print_result_correct_with_series_changed_lhs(point_group, capsys):
    c2v = pd.Series([1, 1, 1, 1], name='A1')
    point_group.print_result(c2v, 'A1 × A1')
    captured = capsys.readouterr()
    assert captured.out[:12] == 'A1 × A1 = A1'


def test_convolution_results_correct_with_two_arguments(point_group, capsys):
    result = point_group.convolution_results('A1', 'A2')
    captured = capsys.readouterr()
    assert result.equals(pd.Series([1, 1, -1, -1], name='A2'))
    assert captured.out[:12] == 'A1 × A2 = A2'

def test_convolution_results_correct_with_four_arguments(point_group, capsys):
    result = point_group.convolution_results('A1', 'A2', 'B1', 'B2')
    captured = capsys.readouterr()
    assert result.equals(pd.Series([1, 1, 1, 1], name='A1'))
    assert captured.out[:22] == 'A1 × A2 × B1 × B2 = A1'
