"""
Program to manipulate chemically important point groups.
"""

import numpy as np
import pandas as pd

import Program.Data.character_tables as character_tables


class PointGroup:
    """
    Point group object.
    """

    def __init__(self, point_group):
        """
        Creates a PointGroup object either using data describing a point in DataFrame format,
        or by passing the name of a supported point group.

        :param point_group: pandas.DataFrame or str
        """
        # If a DataFrame is provided, use that data as the working point group:
        if isinstance(point_group, pd.DataFrame):
            print(point_group)
            self._CHARACTER_TABLE = point_group
        else:
            # If a name of a point group is provided, use that to search character_tables for the data:
            try:
                self._CHARACTER_TABLE = character_tables.character_tables[point_group]

            # If provided name does not exist in character_tables, inform user:
            except KeyError:
                print('The requested point group is not supported, please try again. \n'
                      'The point groups that can be used are the following: ',
                      character_tables.character_tables.keys())

            # Handle other incorrect inputs:
            except TypeError:
                print('Incorrect type has been used to create a PointGroup object.')

    def __repr__(self):
        """Show the name of class and the point group used to construct it. For development purposes."""
        return f'PointGroup({self._CHARACTER_TABLE.index.values[0]})'

    def __str__(self):
        """Show the entire character table. For user purposes."""
        return self._CHARACTER_TABLE

    @property
    def character_table(self):
        """Show the character table, which is immutable."""
        return self._CHARACTER_TABLE

    def reduction(self, representation) -> pd.DataFrame:
        """
        Reduces the provided reducible representation into its constituent irreducible representation, showing all
        working. Uses the reduction formula.

        :param representation: iterable of ints or floats
            The reducible representation to which the reduction formula will be applied.

        :return: pandas.DataFrame
            A table showing the reduction formula being applied to each irreducible representation; ie. each element
            shows the result of the multiplication of an element of the provided reducible representation by the
            corresponding element in the character table and the number of symmetry elements of the corresponding type.
            The last column shows the number of each irreducible representation in the provided reducible
            representation.
        """
        representation = np.array(representation)  # Needed for later calculations.

        # If the provided list is not the same length as the character table, show a warning and stop the attempted
        # reduction. Proceeding would result, at minimum, in incorrect results.
        if len(representation) != len(self._CHARACTER_TABLE.iloc[0]):
            print('ReductionError: The reducible representation must have the same '
                  'number of elements as a row of the character table')
        else:
            try:
                # Apply the reduction formula:
                result = self._CHARACTER_TABLE.iloc[1:] * self._CHARACTER_TABLE.iloc[0] * representation
                # Determine and save the number of times each irreducible representation appears in the reducible repr.
                result['number of appearances'] = result.sum(axis=1) / self._CHARACTER_TABLE.iloc[0].sum()
                return result.astype(int)

            # Handle weird input (eg. words in the array):
            except TypeError:
                print('The reducible representation contains elements of unsupported dtypes. Please make sure that'
                      'the representation parameter is an iterable object of ints or floats.')

    def constituents(self, representation) -> pd.Series:
        """
        Applies the reduction formula to the provided reducible representation, and prints the constituent irreducible
        representations in a nice format.

        :param representation: iterable of ints or floats
            The reducible representation to which the reduction formula will be applied.

        :return: pandas.Series
            A pandas Series of the numbers each irreducible representation appears in the provided reducible
            representation.
        """
        result = self.reduction(representation)  # Reduce the reducible representation.
        self.print_result(result)
        return result['number of appearances']

    def convolution(self, arg1: str, arg2: str, *args) -> pd.Series:
        """
        Performs a convolution of irreducible representation of the point group, ie. multiplies the chosen rows of the
        character table by each other. At least two irreducible representations have to be selected, but the total
        amount is unlimited.

        :param arg1: str, a name of a irreducible representation of the point group
        :param arg2: str, a name of a irreducible representation of the point group
        :param args: zero or more strs or an iterable of strs
            any number of names of irreducible representations of the point group

        :return: pandas.Series, the result of the multiplication, displayed as raw data
        """
        try:
            # Multiply all the chosen row by each other, and rename the Series after the irreducible representations.
            return self._CHARACTER_TABLE.loc[[arg1, arg2, *args]].product().rename(' × '.join(args))
        except KeyError:
            print('The inputted irreducible representations do not exist in this character table. convolution in'
                  f' this point group can be performed using any two or more of {self._CHARACTER_TABLE.index.values}.')

    def match_representation(self, representation):
        """
        Matches the provided representation to the character and determines if it is reducible or not. If it is 
        reducible, it reduces the representation. If it is irreducible, it determines which irreducible representation
        it represents.
        
        :param representation: The representaion to be matched with the character table. Must be the same length as
            there are symmetry elements in the character table.
        :type representation: array-like of ints or floats
            
        :return: Either the row of the character table corresponding to the representation or the result of
            reduction formula.
        :return type: pandas.Series or pandas.DataFrame
        """
        try:
            # Check if the provided representation matches any of the irreducible representations:
            if (self._CHARACTER_TABLE[1:] == representation).all(1).any():
                # Find which rows it matches:
                temp = (self._CHARACTER_TABLE[1:] == representation).all(1)
                # Retreive the correct row from the character table:
                return self._CHARACTER_TABLE.loc[temp[temp == True].index.values[0]]
            # The provided representation is reducible:
            else:
                # Reduce the provided reducible representation:
                result = self.reduction(representation)
                return result.astype(int)
        # If the length of the provided representation is not the same as the number of symmetry elements:
        except ValueError:
            print('MatchRepresentationError: The reducible representation must have '
                  'the same number of elements as a row of the character table')

    def show_matched_representation(self, representation):
        """
        Prints the results of matching the provided representation to the character table in a clear, user-friendly
        format.

        :param representation: array-like of ints or floats
            The representaion to be matched with the character table. Must be the same length as there are symmetry
            elements in the character table.

        :return: pandas.Series or pandas.DataFrame
            Either the row of the character table corresponding to the representation or the result of reduction formula
        """
        # Match the representation to the character table:
        result = self.match_representation(representation)
        try:
            self.print_result(result, representation.name)  # Print neatly
        except:
            self.print_result(result)
        return result

    def print_result(self, df, left_hand_side='result'):
        """
        Prints the provided result in a user-friendly format.

        :param df: The result to be printed nicely. A DataFrame must have 'number of appearances' column, and Series
            must have a name.
        :type df: pandas.DataFrame or pandas.Series

        :param left_hand_side: A description of the result's meaning or a representation of the starting values.
            eg. A1; A1 × B2

        :return: 'number of appearances' column if df is a DataFrame, df if df is a Series
        """
        # If df is a DataFrame, print the result as sum of names of irreducible representations which have nonzero
        # value in the 'number of appearances' column.
        if isinstance(df, pd.DataFrame):
            # Filter out the irreducible representations which don't appear in the reducible reprs, and display nicely:
            to_print = [f'{i if i != 1 else ""}{j}' for i, j in zip(df['number of appearances'],
                                                                    df.index.values) if i != 0]
            print(left_hand_side, '=', '  +  '.join(to_print))
            return df['number of appearances']
        # If df is a Series, print its name as the result.
        elif isinstance(df, pd.Series):
            print(left_hand_side, '=', df.name)
            return df

    def convolution_results(self, arg1: str, arg2: str, *args):
        """
        Performs a convolution of 2 or more irreducible representations, matches the result to the character table,
        and prints the result in a user-friendly format.

        :param arg1: Irreducible representation to be convoluted.
        :type arg1: str
        :param arg2: Irreducible representation to be convoluted.

        :type arg2: str
        :param args: Any number of additional irreducible representations to be convoluted.

        :return: The result of matching the convolution result to the character table.
        """
        # Obtain convolution results and match these to the character table:
        result = self.match_representation(self.convolution(arg1, arg2, *args))

        # User-friendly way to show which representations were convoluted:
        left_hand_side = ' × '.join([arg1, arg2, *args])
        self.print_result(result, left_hand_side)  # Print nicely
        return result

