import numpy as np
import pandas as pd


class Molecule:
    def __init__(self, mol):
        if type(mol) is str:
            with open(fr'C:\Users\TACHYON\Documents\GitHub\symmetry\Program\Data\Molecules\{mol}.mol', 'r') as file:
                file = list(file)
                self._parse_mol(file)
        else:
            try:
                mol = list(mol)
                self._parse_mol(mol)
            except TypeError:
                print('MoleculeError: mol is not a string and is not an iterable. mol should either identify a '
                      'molecule, be a path to a .mol file, or be a str or list representing the contents of a mol file')

    def _parse_mol(self, file):
        # TODO: add compatibility with V3000
        try:
            self._parse_counts_block(file)
            self._parse_atoms_block(file)
            self._parse_bonds_block(file)
        except ValueError as e:
            print(e)

    def _parse_counts_block(self, data: list):
        """
        Extracts data from the counts block of a .mol file. This data consists of the number of atoms in the
        molecule, the number of bonds in the molecule, and the chirality.

        :param data: The data contained in a .mol file. This should follow the standards of the format.
        :type data: list
        """
        # Counts block is located on line 4
        line_block = data[3].split(' ')
        # Obtain only the numbers
        line_block = [int(i) for i in line_block[:-1] if i]
        # Store the information contained in the counts block
        self._natoms = line_block[0]
        self._nbonds = line_block[1]
        self._chiral = bool(line_block[3])

    def _parse_atoms_block(self, data: list):
        """
        Extracts data from the atoms block of a .mol file. This data consists of the coordinate (in Angstrom) of each
        atom in the molecule. Each atom has a separate row. The numeric columns at the end contain extra data.

        :param data: The data contained in a .mol file. This should follow the standards of the format.
        :type data: list
        """
        structure = np.zeros((self.natoms, 3))  # Set up table containing coordinates of each atom.
        row_names = []  # Set up list containing the chemical symbols of each atom.

        # Loop over all lines of the atoms block:
        for index, line in enumerate(data[4:(4 + self.natoms)]):
            # Process the line by removing the separating spaces. Check for numeric in case data is corrupted:
            line = [float(i) if i.isnumeric() else i for i in line.split(' ') if i]
            structure[index] = line[:3]  # Store the coordinates (first three columns)
            row_names.append(line[3])  # Store chemical symbol of the atom.
        self._structure = pd.DataFrame(structure, row_names, ['x', 'y', 'z'])  # Create df from the extracted data.

    def _parse_bonds_block(self, data: list):
        """
        Extract data from the bonds block of a .mol file. This data shows for each bond which atoms participate in it,
        the type of the bond, and its stereochemistry.

        :param data: The data contained in a .mol file. This should follow the standards of the format.
        :type data: list
        """
        # Set up the table for the data, relying on number of bonds to know how many rows will be needed.
        bonds = np.zeros((self.nbonds, 4), dtype=int)

        # Loop over the lines corresponding to the bonds block:
        for index, line in enumerate(data[(4 + self.natoms):(4 + self.natoms + self.nbonds)]):
            line = [int(i) for i in line.split(' ') if i]  # Process each line by splitting it and removing empty.
            bonds[index] = line[:4]  # Relevant information is in the first four columns.

        # Create the row names by taking the the involved atoms' chemical symbols from structure df.
        row_names = [f'{self.structure.index[bonds[i, 0] - 1]}-{self.structure.index[bonds[i, 1] - 1]}'
                     for i in range(self.nbonds)]
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
