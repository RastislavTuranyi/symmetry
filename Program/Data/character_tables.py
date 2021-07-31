"""
Character tables for use in the program.

This is raw data and its contetns should be manipulated with caution.

All the data is stored in pands DataFrames, where first the body of the character table is defined, and then
the row names (irreducible representations) are inputted.

All the DataFrames are listed in a dictionary.
"""

import pandas as pd


Cs = pd.DataFrame([[1, 1],
                   [1, 1],
                   [1, -1]])
Cs.index = ['Cs', "A'", "A''"]

C2v = pd.DataFrame([[1, 1, 1, 1],
                    [+1, +1, +1, +1],
                    [+1, +1, -1, -1],
                    [+1, -1, +1, -1],
                    [+1, -1, -1, +1]
                    ])
C2v.index = ['C2v', 'A1', 'A2', 'B1', 'B2']

C3v = pd.DataFrame([[1, 2, 3],
                    [+1, +1, +1],
                    [+1, +1, -1],
                    [+2, -1, 0]
                    ])
C3v.index = ['C3v', 'A1', 'A2', 'E']

C4v = pd.DataFrame([[1, 2, 1, 2, 2],
                    [+1, +1, +1, +1, +1],
                    [+1, +1, +1, -1, -1],
                    [+1, -1, +1, +1, -1],
                    [+1, -1, +1, -1, +1],
                    [2, 0, -2, 0, 0]
                    ])
C4v.index = ['C4v', 'A1', 'A2', 'B1', 'B2', 'E']

C6v = pd.DataFrame([[1, 2, 2, 1, 3, 3],
                    [1, +1, +1, +1, +1, +1],
                    [+1, +1, +1, +1, -1, -1],
                    [+1, -1, +1, -1, +1, -1],
                    [+1, -1, +1, -1, -1, +1],
                    [+2, +1, -1, -2, 0, 0],
                    [+2, -1, -1, +2, 0, 0]
                    ])
C6v.index = ['C6v', 'A1', 'A2', 'B1', 'B2', 'E1', 'E2']

C8v = pd.DataFrame([[1, 2, 2, 2, 1, 4, 4],
                    [+1, +1, +1, +1, +1, +1, +1],
                    [+1, +1, +1, +1, +1, -1, -1],
                    [+1, -1, +1, -1, +1, +1, -1],
                    [+1, -1, +1, -1, +1, -1, +1],
                    [+2, +(2 ** 0.5), 0, -(2 ** 0.5), -2, 0, 0],
                    [+2, 0, -2, 0, +2, 0, 0],
                    [+2, -(2 ** 0.5), 0, +(2 ** 0.5), -2, 0, 0]
                    ])
C8v.index = ['C8v', 'A1', 'A2', 'B1', 'B2', 'E1', 'E2', 'E3']

C2h = pd.DataFrame([[1, 1, 1, 1],
                    [+1, +1, +1, +1],
                    [+1, -1, +1, -1],
                    [+1, +1, -1, -1],
                    [+1, -1, -1, +1]
                    ])
C2h.index = ['C2h', 'Ag', 'Bg', 'Au', 'Bu']

D2h = pd.DataFrame([[1 for i in range(8)],
                    [+1, +1, +1, +1, +1, +1, +1, +1],
                    [1, +1, -1, -1, +1, +1, -1, -1],
                    [+1, -1, +1, -1, +1, -1, +1, -1],
                    [1, -1, -1, +1, +1, -1, -1, +1],
                    [+1, +1, +1, +1, -1, -1, -1, -1],
                    [+1, +1, -1, -1, -1, -1, +1, +1],
                    [+1, -1, +1, -1, -1, +1, -1, +1],
                    [+1, -1, -1, +1, -1, +1, +1, -1]
                    ])
D2h.index = ['D2h', 'Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u']

D3h = pd.DataFrame([[1, 2, 3, 1, 2, 3],
                    [1, +1, +1, +1, +1, +1],
                    [+1, +1, -1, +1, +1, -1],
                    [+2, -1, 0, +2, -1, 0],
                    [1, +1, +1, -1, -1, -1],
                    [+1, +1, -1, -1, -1, +1],
                    [+2, -1, 0, -2, +1, 0]
                    ])
D3h.index = ['D3h', "A'1", "A'2", "E'", "A''1", "A''2", "E''"]

D4h = pd.DataFrame([[1, 2, 1, 2, 2, 1, 2, 1, 2, 2],
                    [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
                    [+1, +1, +1, -1, -1, +1, +1, +1, -1, -1],
                    [+1, -1, +1, +1, -1, +1, -1, +1, +1, -1],
                    [+1, -1, +1, -1, +1, +1, -1, +1, -1, +1],
                    [+2, 0, -2, 0, 0, +2, 0, -2, 0, 0],
                    [1, +1, +1, +1, +1, -1, -1, -1, -1, -1],
                    [+1, +1, +1, -1, -1, -1, -1, -1, +1, +1],
                    [+1, -1, +1, +1, -1, -1, +1, -1, -1, +1],
                    [+1, -1, +1, -1, +1, -1, +1, -1, +1, -1],
                    [+2, 0, -2, 0, 0, -2, 0, +2, 0, 0]
                    ])
D4h.index = ['D4h', 'A1g', 'A2g', 'B1g', 'B2g', 'Eg', 'A1u', 'A2u', 'B1u', 'B2u', 'Eu']

D6h = pd.DataFrame([[1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3],
                    [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
                    [+1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, -1],
                    [+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1],
                    [+1, -1, +1, -1, -1, +1, +1, -1, +1, -1, -1, +1],
                    [+2, +1, -1, -2, 0, 0, +2, +1, -1, -2, 0, 0],
                    [+2, -1, -1, +2, 0, 0, +2, -1, -1, +2, 0, 0],
                    [+1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1],
                    [+1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1],
                    [+1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1],
                    [+1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, -1],
                    [+2, +1, -1, -2, 0, 0, -2, -1, +1, +2, 0, 0],
                    [+2, -1, -1, +2, 0, 0, -2, +1, +1, -2, 0, 0]
                    ])
D6h.index = ['D6h', 'A1g', 'A2g', 'B1g', 'B2g', 'E1g', 'E2g', 'A1u', 'A2u', 'B1u', 'B2u', 'E1u', 'E2u']

D2d = pd.DataFrame([[1, 2, 1, 2, 2],
                    [+1, +1, +1, +1, +1],
                    [+1, +1, +1, -1, -1],
                    [+1, -1, +1, +1, -1],
                    [+1, -1, +1, -1, +1],
                    [+2, 0, -2, 0, 0]
                    ])
D2d.index = ['D2d', 'A1', 'A2', 'B1', 'B2', 'E']

D3d = pd.DataFrame([[1, 2, 3, 1, 2, 3],
                    [1, +1, +1, +1, +1, +1],
                    [+1, +1, -1, +1, +1, -1],
                    [+2, -1, 0, +2, -1, 0],
                    [1, +1, +1, -1, -1, -1],
                    [+1, +1, -1, -1, -1, +1],
                    [+2, -1, 0, -2, +1, 0]
                    ])
D3d.index = ['D3d', 'A1g', 'A2g', 'Eg', 'A1u', 'A2u', 'Eu']

D4d = pd.DataFrame([[1, 2, 2, 2, 1, 4, 4],
                    [+1, +1, +1, +1, +1, +1, +1],
                    [+1, +1, +1, +1, +1, -1, -1],
                    [+1, -1, +1, -1, +1, +1, -1],
                    [+1, -1, +1, -1, +1, -1, +1],
                    [+2, +(2 ** .5), 0, -(2 ** .5), -2, 0, 0],
                    [+2, 0, -2, 0, +2, 0, 0],
                    [+2, -(2 ** .5), 0, +(2 ** .5), -2, 0, 0]
                    ])
D4d.index = ['D4d', 'A1', 'A2', 'B1', 'B2', 'E1', 'E2', 'E3']

D6d = pd.DataFrame([[1, 2, 2, 2, 2, 2, 1, 6, 6],
                    [+1, +1, +1, +1, +1, +1, +1, +1, +1],
                    [+1, +1, +1, +1, +1, +1, +1, -1, -1],
                    [+1, -1, +1, -1, +1, -1, +1, +1, -1],
                    [+1, -1, +1, -1, +1, -1, +1, -1, +1],
                    [+2, +(3 ** .5), +1, 0, -1, -(3 ** .5), -2, 0, 0],
                    [+2, +1, -1, -2, -1, +1, +2, 0, 0],
                    [+2, 0, -2, 0, +2, 0, -2, 0, 0],
                    [+2, -1, -1, +2, -1, -1, +2, 0, 0],
                    [2, -(3 ** .5), +1, 0, -1, +(3 ** .5), -2, 0, 0]
                    ])
D6d.index = ['D6d', 'A1', 'A2', 'B1', 'B2', 'E1', 'E2', 'E3', 'E4', 'E5']

Td = pd.DataFrame([[1, 8, 3, 6, 6],
                   [+1, +1, +1, +1, +1],
                   [1, +1, +1, -1, -1],
                   [+2, -1, +2, 0, 0],
                   [+3, 0, -1, +1, -1],
                   [+3, 0, -1, -1, +1]
                   ])
Td.index = ['Td', 'A1', 'A2', 'E', 'T1', 'T2']

O = pd.DataFrame([[1, 8, 6, 6, 3],
                  [+1, +1, +1, +1, +1],
                  [+1, +1, -1, -1, +1],
                  [+2, -1, 0, 0, +2],
                  [+3, 0, -1, +1, -1],
                  [+3, 0, +1, -1, -1]
                  ])
O.index = ['O', 'A1', 'A2', 'E', 'T1', 'T2']

Oh = pd.DataFrame([[1, 8, 6, 6, 3, 1, 6, 8, 3, 6],
                   [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
                   [+1, +1, -1, -1, +1, +1, -1, +1, +1, -1],
                   [+2, -1, 0, 0, +2, +2, 0, -1, +2, 0],
                   [+3, 0, -1, +1, -1, +3, +1, 0, -1, -1],
                   [+3, 0, +1, -1, -1, +3, -1, 0, -1, +1],
                   [+1, +1, +1, +1, +1, -1, -1, -1, -1, -1],
                   [+1, +1, -1, -1, +1, -1, +1, -1, -1, +1],
                   [+2, -1, 0, 0, +2, -2, 0, +1, -2, 0],
                   [+3, 0, -1, +1, -1, -3, -1, 0, +1, +1],
                   [+3, 0, +1, -1, -1, -3, +1, 0, +1, -1]
                   ])
Oh.index = ['Oh', 'A1g', 'A2g', 'Eg', 'T1g', 'T2g', 'A1u', 'A2u', 'Eu', 'T1u', 'T2u']

character_tables = {'Cs': Cs, 'C2v': C2v, 'C3v': C3v, 'C4v': C4v, 'C6v': C8v,
                    'C2h': C2h,
                    'D2h': D2h, 'D3h': D3h, 'D4h': D4h, 'D6h': D6h,
                    'D2d': D2d, 'D3d': D3d, 'D4d': D4d, 'D6d': D6d,
                    'Td': Td, 'O': O, 'Oh': Oh}