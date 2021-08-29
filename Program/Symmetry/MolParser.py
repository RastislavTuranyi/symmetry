import warnings

import numpy as np
import pandas as pd


class CorruptedFileWarning(Warning):
    pass


class MolParser:
    # TODO: add method to change properties from file
    def __init__(self, mol):
        if type(mol) is str:
            with open(fr'C:\Users\TACHYON\Documents\GitHub\symmetry\Program\Data\Molecules\{mol}.mol', 'r') as file:
                file = np.array(list(file))
                self._parse_mol(file)
        else:
            try:
                mol = np.array(mol)
                self._parse_mol(mol)
            except TypeError:
                print('MoleculeError: mol is not a string and is not an iterable. mol should either identify a '
                      'molecule, be a path to a .mol file, or be a str or list representing the contents of a mol file')

    def _parse_mol(self, file):
        # TODO: add compatibility with V3000
        self._assign_counts_block_values(file, self._read_counts_block(file))
        atoms_block_end = 4 + self.natoms

        try:
            self._parse_atoms_block(file[4:atoms_block_end])
        except ValueError:
            warnings.warn('The structure of the molecule could not be set because the coordinates part '
                          f'of the atoms block, specifically line {self._error_line + 1} of the atoms block in the '
                          'provided .mol data contains non-numeric values. Please ensure all the data follows the .mol '
                          'standard. Then you can set the structure attribute again.', CorruptedFileWarning)

        try:
            self._parse_bonds_block(file[atoms_block_end:(atoms_block_end + self.nbonds)])
        except ValueError:
            warnings.warn('The bonds of the molecule could not be set because the data on line '
                          f'{self._error_line + 1} in the provided .mol data contains non-numeric values. '
                          'Please ensure all the data follows the .mol standard. Then you can set the '
                          'structure attribute again.', CorruptedFileWarning)
        # TODO: check_consistency function

    @staticmethod
    def _read_counts_block(file: np.array) -> list:
        """
        Extracts data from the counts block of a .mol file. This data consists of the number of atoms in the
        molecule, the number of bonds in the molecule, and the chirality.

        :param file: The whole .mol file.
        :type file: array-like

        :return: List of values stored in the counts block.
        :rtype: list of ints
        """
        line_block = []
        for index, value in enumerate(file[3].split(' ')[:-1]):  # Ignore last value since it's about file format
            if value:  # If the string is not empty
                try:
                    line_block.append(int(value))
                except ValueError:
                    line_block.append(None)
                    warnings.warn('The counts block of the provided .mol data is corrupted. Please ensure all the data'
                                  ' follows the .mol standard, then you can set the attributes again.',
                                  CorruptedFileWarning)
        return line_block

    def _assign_counts_block_values(self, file: np.array, line_block: list):
        """
        Stores the values parsed from the counts block into instance attributes. If no values could be read, attempts
        to estimate the values using the other blocks.

        :param file: The whole file.
        :type file: array-like
        :param line_block: List of values stored in the counts block.
        :type line_block: array-like
        """
        if line_block[0] is None:  # If the number of atoms failed to be read, try to estimate it
            self._guess_natoms(file)
        else:
            self._natoms = line_block[0]
        if line_block[1] is None:  # If the number of bonds failed to be read, try to estimate it
            self._guess_nbonds(file)
        else:
            self._nbonds = line_block[1]
        if line_block[3] not in [0, 1]:
            warnings.warn('The chirality in the provided .mol data is corrupted. The 4th value in the counts block'
                          'must be either 0 (achiral) or 1 (chiral). Please ensure all data follows the .mol standard,'
                          'then you can restart the process. Alternatively you can change this manually.',
                          CorruptedFileWarning)
            self._chiral = None
        else:
            self._chiral = bool(line_block[3])

    def _guess_natoms(self, file: np.array):
        """
        Estimate the number of atoms using the atoms block. Assumes that the coordinates in the atoms block are floats
        (indicated by '.') while the bonds block has only ints (no'.').

        :param file: The whole .mol file
        :type file: array-like
        """
        for index, value in enumerate(file[4:]):
            if value.find('.') == -1:
                self._natoms = index
                warnings.warn('The molecule\'s number of atoms was estimated from the number of rows in the atoms'
                              ' block. This may not be accurate, so please check and if necessary correct this.')
                break
        else:
            raise CorruptedFileWarning('The number of atoms could not be estimated because the provided data. '
                                       'contains a full stop (".") in every line past 4th.')

    def _guess_nbonds(self, file: np.array):
        """
        Estimates the number of bonds using the bonds block. Assumes that the information following the bonds block
        (either the properties block of END) starts with first character being 'M'.

        :param file: The whole .mol file.
        :type file: array-like
        """
        for index, value in enumerate(file[(4 + self.natoms):]):
            if value[0] == 'M':
                self._nbonds = index
                warnings.warn('The molecule\'s number of bonds was estimated from the number of rows in the bonds'
                              ' block. This may not be accurate, so please check and if necessary correct this.')
                break
        else:
            warnings.warn('The number of bonds could not be estimated because the provided data '
                          'does not contain a line starting wtih "M". Please change the file to follow'
                          'the .mol format or set the number of bonds manually', CorruptedFileWarning)

    def _parse_atoms_block(self, data: np.array):
        """
        Extracts data from the atoms block of a .mol file. This data consists of the coordinate (in Angstrom) of each
        atom in the molecule. Each atom has a separate row. The numeric columns at the end contain extra data.

        :param data: The data contained in a .mol file. This should follow the standards of the format.
        :type data: array-like
        """
        structure = np.zeros((self.natoms, 3))  # Set up table containing coordinates of each atom.
        row_names = []  # Set up list containing the chemical symbols of each atom.

        # Loop over all lines of the atoms block:
        for index, line in enumerate(data):
            # Process the line by removing the separating spaces. Check for numeric in case data is corrupted:
            line = [float(i) if i.isnumeric() else i for i in line.split(' ') if i]
            try:
                structure[index] = line[:3]  # Store the coordinates (first three columns)
            except ValueError:
                self._error_line = index
                raise ValueError
            row_names.append(line[3])  # Store chemical symbol of the atom.

        self._structure = pd.DataFrame(structure, row_names, ['x', 'y', 'z'])  # Create df from the extracted data.

    def _parse_bonds_block(self, data: np.array):
        """
        Extract data from the bonds block of a .mol file. This data shows for each bond which atoms participate in it,
        the type of the bond, and its stereochemistry.

        :param data: The data contained in a .mol file. This should follow the standards of the format.
        :type data: list
        """
        # Set up the table for the data, relying on number of bonds to know how many rows will be needed.
        bonds = np.zeros((self.nbonds, 4), dtype=int)

        # Loop over the lines corresponding to the bonds block:
        for index, line in enumerate(data):
            try:
                line = [int(i) for i in line.split(' ') if i]  # Process each line by splitting it and removing empty.
                bonds[index] = line[:4]  # Relevant information is in the first four columns.
            except ValueError:  # If there is a non-numeric value in the bonds block
                self._error_line = index
                raise ValueError  # Interrupt the parsing

        # Create the row names by taking the the involved atoms' chemical symbols from structure df.
        try:
            row_names = [f'{self.structure.index[bonds[i, 0] - 1]}-{self.structure.index[bonds[i, 1] - 1]}'
                         for i in range(self.nbonds)]
        except AttributeError:
            row_names = [i for i in range(1, self.nbonds + 1)]  # Create dummy row names
            warnings.warn('The bond types could not be identified because structure attribute failed to be set.',
                          CorruptedFileWarning)

        self._bonds = pd.DataFrame(bonds, row_names, ['Atom1', 'Atom2', 'Bond type', 'Stereochemistry'])

    @property
    def natoms(self) -> int:
        """
        :return: The number of atoms in the molecule.
        :rtype: int
        """
        return self._natoms

    @property
    def nbonds(self) -> int:
        """
        :return: The number of bonds in the molecule.
        :rtype: int
        """
        return self._nbonds

    @property
    def chiral(self) -> bool:
        """
        :return: The chirality of the molecule.
        :rtype: bool
        """
        return self._chiral

    @property
    def structure(self) -> pd.DataFrame:
        """
        :return: The coordinates of all atoms in the molecule.
        :rtype: pandas.DataFrame
        """
        return self._structure

    @property
    def bonds(self) -> pd.DataFrame:
        """
        :return: The bonds in the molecule.
        :rtype: pandas.DataFrame
        """
        return self._bonds
