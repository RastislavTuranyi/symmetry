"""
Program to manipulate chemically important point groups.
"""

import numpy as np
import pandas as pd

import character_tables


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
            print('The reducible representation must have the same number of elements as a row of the character table')
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
        # Filter out the irreducible representations which don't appear in the reducible reprs, and display nicely:
        to_print = [f'{i} × {j}' for i, j in
                    zip(result['number of appearances'], result.index.values) if i != 0]
        print('  +  '.join(to_print))
        return result['number of appearances']
