#!/usr/bin/python
# -*- coding:utf-8 -*-
# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dictionary containing ideal internal coordinates and chi angle assignments
 for building amino acid 3D coordinates"""
from typing import Dict


AA_GEOMETRY: Dict[str, dict] = {
    "ALA": {
        "atoms": ["CB"],
        "chi_indices": [],
        "parents": [["N", "C", "CA"]],
        "types": {"C": "C", "CA": "CT1", "CB": "CT3", "N": "NH1", "O": "O"},
        "z-angles": [111.09],
        "z-dihedrals": [123.23],
        "z-lengths": [1.55],
    },
    "ARG": {
        "atoms": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "chi_indices": [1, 2, 3, 4],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "NE"],
            ["CD", "NE", "CZ"],
            ["NH1", "NE", "CZ"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD": "CT2",
            "CG": "CT2",
            "CZ": "C",
            "N": "NH1",
            "NE": "NC2",
            "NH1": "NC2",
            "NH2": "NC2",
            "O": "O",
        },
        "z-angles": [112.26, 115.95, 114.01, 107.09, 123.05, 118.06, 122.14],
        "z-dihedrals": [123.64, 180.0, 180.0, 180.0, 180.0, 180.0, 178.64],
        "z-lengths": [1.56, 1.55, 1.54, 1.5, 1.34, 1.33, 1.33],
    },
    "ASN": {
        "atoms": ["CB", "CG", "OD1", "ND2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["OD1", "CB", "CG"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CG": "CC",
            "N": "NH1",
            "ND2": "NH2",
            "O": "O",
            "OD1": "O",
        },
        "z-angles": [113.04, 114.3, 122.56, 116.15],
        "z-dihedrals": [121.18, 180.0, 180.0, -179.19],
        "z-lengths": [1.56, 1.53, 1.23, 1.35],
    },
    "ASP": {
        "atoms": ["CB", "CG", "OD1", "OD2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["OD1", "CB", "CG"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2A",
            "CG": "CC",
            "N": "NH1",
            "O": "O",
            "OD1": "OC",
            "OD2": "OC",
        },
        "z-angles": [114.1, 112.6, 117.99, 117.7],
        "z-dihedrals": [122.33, 180.0, 180.0, -170.23],
        "z-lengths": [1.56, 1.52, 1.26, 1.25],
    },
    "CYS": {
        "atoms": ["CB", "SG"],
        "chi_indices": [1],
        "parents": [["N", "C", "CA"], ["N", "CA", "CB"]],
        "types": {"C": "C", "CA": "CT1", "CB": "CT2", "N": "NH1", "O": "O", "SG": "S"},
        "z-angles": [111.98, 113.87],
        "z-dihedrals": [121.79, 180.0],
        "z-lengths": [1.56, 1.84],
    },
    "GLN": {
        "atoms": ["CB", "CG", "CD", "OE1", "NE2"],
        "chi_indices": [1, 2, 3],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["OE1", "CG", "CD"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD": "CC",
            "CG": "CT2",
            "N": "NH1",
            "NE2": "NH2",
            "O": "O",
            "OE1": "O",
        },
        "z-angles": [111.68, 115.52, 112.5, 121.52, 116.84],
        "z-dihedrals": [121.91, 180.0, 180.0, 180.0, 179.57],
        "z-lengths": [1.55, 1.55, 1.53, 1.23, 1.35],
    },
    "GLU": {
        "atoms": ["CB", "CG", "CD", "OE1", "OE2"],
        "chi_indices": [1, 2, 3],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["OE1", "CG", "CD"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2A",
            "CD": "CC",
            "CG": "CT2",
            "N": "NH1",
            "O": "O",
            "OE1": "OC",
            "OE2": "OC",
        },
        "z-angles": [111.71, 115.69, 115.73, 114.99, 120.08],
        "z-dihedrals": [121.9, 180.0, 180.0, 180.0, -179.1],
        "z-lengths": [1.55, 1.56, 1.53, 1.26, 1.25],
    },
    "GLY": {
        "atoms": [],
        "chi_indices": [],
        "parents": [],
        "types": {"C": "C", "CA": "CT2", "N": "NH1", "O": "O"},
        "z-angles": [],
        "z-dihedrals": [],
        "z-lengths": [],
    },
    "HIS": {
        "atoms": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["ND1", "CB", "CG"],
            ["CB", "CG", "ND1"],
            ["CB", "CG", "CD2"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD2": "CPH1",
            "CE1": "CPH2",
            "CG": "CPH1",
            "N": "NH1",
            "ND1": "NR1",
            "NE2": "NR2",
            "O": "O",
        },
        "z-angles": [109.99, 114.05, 124.1, 129.6, 107.03, 110.03],
        "z-dihedrals": [122.46, 180.0, 90.0, -171.29, -173.21, 171.99],
        "z-lengths": [1.55, 1.5, 1.38, 1.36, 1.35, 1.38],
    },
    "HSD": {
        "atoms": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["ND1", "CB", "CG"],
            ["CB", "CG", "ND1"],
            ["CB", "CG", "CD2"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD2": "CPH1",
            "CE1": "CPH2",
            "CG": "CPH1",
            "N": "NH1",
            "ND1": "NR1",
            "NE2": "NR2",
            "O": "O",
        },
        "z-angles": [109.99, 114.05, 124.1, 129.6, 107.03, 110.03],
        "z-dihedrals": [122.46, 180.0, 90.0, -171.29, -173.21, 171.99],
        "z-lengths": [1.55, 1.5, 1.38, 1.36, 1.35, 1.38],
    },
    "HSE": {
        "atoms": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "chi_indices": [],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["ND1", "CB", "CG"],
            ["CB", "CG", "ND1"],
            ["CB", "CG", "CD2"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD2": "CPH1",
            "CE1": "CPH2",
            "CG": "CPH1",
            "N": "NH1",
            "ND1": "NR2",
            "NE2": "NR1",
            "O": "O",
        },
        "z-angles": [111.67, 116.94, 120.17, 129.71, 105.2, 105.8],
        "z-dihedrals": [123.52, 180.0, 90.0, -178.26, -179.2, 178.66],
        "z-lengths": [1.56, 1.51, 1.39, 1.36, 1.32, 1.38],
    },
    "HSP": {
        "atoms": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "chi_indices": [],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["ND1", "CB", "CG"],
            ["CB", "CG", "ND1"],
            ["CB", "CG", "CD2"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2A",
            "CD2": "CPH1",
            "CE1": "CPH2",
            "CG": "CPH1",
            "N": "NH1",
            "ND1": "NR3",
            "NE2": "NR3",
            "O": "O",
        },
        "z-angles": [109.38, 114.18, 122.94, 128.93, 108.9, 106.93],
        "z-dihedrals": [125.13, 180.0, 90.0, -165.26, -167.62, 167.13],
        "z-lengths": [1.55, 1.52, 1.37, 1.35, 1.33, 1.37],
    },
    "ILE": {
        "atoms": ["CB", "CG1", "CG2", "CD1"],
        "chi_indices": [1, 3],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CG1", "CA", "CB"],
            ["CA", "CB", "CG1"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT1",
            "CD": "CT3",
            "CG1": "CT2",
            "CG2": "CT3",
            "N": "NH1",
            "O": "O",
        },
        "z-angles": [112.93, 113.63, 113.93, 114.09],
        "z-dihedrals": [124.22, 180.0, -130.04, 180.0],
        "z-lengths": [1.57, 1.55, 1.55, 1.54],
    },
    "LEU": {
        "atoms": ["CB", "CG", "CD1", "CD2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CD1", "CB", "CG"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD1": "CT3",
            "CD2": "CT3",
            "CG": "CT1",
            "N": "NH1",
            "O": "O",
        },
        "z-angles": [112.12, 117.46, 110.48, 112.57],
        "z-dihedrals": [121.52, 180.0, 180.0, 120.0],
        "z-lengths": [1.55, 1.55, 1.54, 1.54],
    },
    "LYS": {
        "atoms": ["CB", "CG", "CD", "CE", "NZ"],
        "chi_indices": [1, 2, 3, 4],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "CE"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD": "CT2",
            "CE": "CT2",
            "CG": "CT2",
            "N": "NH1",
            "NZ": "NH3",
            "O": "O",
        },
        "z-angles": [111.36, 115.76, 113.28, 112.33, 110.46],
        "z-dihedrals": [122.23, 180.0, 180.0, 180.0, 180.0],
        "z-lengths": [1.56, 1.54, 1.54, 1.53, 1.46],
    },
    "MET": {
        "atoms": ["CB", "CG", "SD", "CE"],
        "chi_indices": [1, 2, 3],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "SD"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CE": "CT3",
            "CG": "CT2",
            "N": "NH1",
            "O": "O",
            "SD": "S",
        },
        "z-angles": [111.88, 115.92, 110.28, 98.94],
        "z-dihedrals": [121.62, 180.0, 180.0, 180.0],
        "z-lengths": [1.55, 1.55, 1.82, 1.82],
    },
    "PHE": {
        "atoms": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CD1", "CB", "CG"],
            ["CB", "CG", "CD1"],
            ["CB", "CG", "CD2"],
            ["CG", "CD1", "CE1"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD1": "CA",
            "CD2": "CA",
            "CE1": "CA",
            "CE2": "CA",
            "CG": "CA",
            "CZ": "CA",
            "N": "NH1",
            "O": "O",
        },
        "z-angles": [112.45, 112.76, 120.32, 120.76, 120.63, 120.62, 119.93],
        "z-dihedrals": [122.49, 180.0, 90.0, -177.96, -177.37, 177.2, -0.12],
        "z-lengths": [1.56, 1.51, 1.41, 1.41, 1.4, 1.4, 1.4],
    },
    "PRO": {
        "atoms": ["CB", "CG", "CD"],
        "chi_indices": [1, 2],
        "parents": [["N", "C", "CA"], ["N", "CA", "CB"], ["CA", "CB", "CG"]],
        "types": {
            "C": "C",
            "CA": "CP1",
            "CB": "CP2",
            "CD": "CP3",
            "CG": "CP2",
            "N": "N",
            "O": "O",
        },
        "z-angles": [111.74, 104.39, 103.21],
        "z-dihedrals": [113.74, 31.61, -34.59],
        "z-lengths": [1.54, 1.53, 1.53],
    },
    "SER": {
        "atoms": ["CB", "OG"],
        "chi_indices": [1],
        "parents": [["N", "C", "CA"], ["N", "CA", "CB"]],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "N": "NH1",
            "O": "O",
            "OG": "OH1",
        },
        "z-angles": [111.4, 112.45],
        "z-dihedrals": [124.75, 180.0],
        "z-lengths": [1.56, 1.43],
    },
    "THR": {
        "atoms": ["CB", "OG1", "CG2"],
        "chi_indices": [1],
        "parents": [["N", "C", "CA"], ["N", "CA", "CB"], ["OG1", "CA", "CB"]],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT1",
            "CG2": "CT3",
            "N": "NH1",
            "O": "O",
            "OG1": "OH1",
        },
        "z-angles": [112.74, 112.16, 115.91],
        "z-dihedrals": [126.46, 180.0, -124.13],
        "z-lengths": [1.57, 1.43, 1.53],
    },
    "TRP": {
        "atoms": ["CB", "CG", "CD2", "CD1", "CE2", "NE1", "CE3", "CZ3", "CH2", "CZ2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CD2", "CB", "CG"],
            ["CD1", "CG", "CD2"],
            ["CG", "CD2", "CE2"],
            ["CE2", "CG", "CD2"],
            ["CE2", "CD2", "CE3"],
            ["CD2", "CE3", "CZ3"],
            ["CE3", "CZ3", "CH2"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD1": "CA",
            "CD2": "CPT",
            "CE2": "CPT",
            "CE3": "CAI",
            "CG": "CY",
            "CH2": "CA",
            "CZ2": "CAI",
            "CZ3": "CA",
            "N": "NH1",
            "NE1": "NY",
            "O": "O",
        },
        "z-angles": [
            111.23,
            115.14,
            123.95,
            129.18,
            106.65,
            107.87,
            132.54,
            118.16,
            120.97,
            120.87,
        ],
        "z-dihedrals": [
            122.68,
            180.0,
            90.0,
            -172.81,
            -0.08,
            0.14,
            179.21,
            -0.2,
            0.1,
            0.01,
        ],
        "z-lengths": [1.56, 1.52, 1.44, 1.37, 1.41, 1.37, 1.4, 1.4, 1.4, 1.4],
    },
    "TYR": {
        "atoms": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CD1", "CB", "CG"],
            ["CB", "CG", "CD1"],
            ["CB", "CG", "CD2"],
            ["CG", "CD1", "CE1"],
            ["CE1", "CE2", "CZ"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD1": "CA",
            "CD2": "CA",
            "CE1": "CA",
            "CE2": "CA",
            "CG": "CA",
            "CZ": "CA",
            "N": "NH1",
            "O": "O",
            "OH": "OH1",
        },
        "z-angles": [112.34, 112.94, 120.49, 120.46, 120.4, 120.56, 120.09, 120.25],
        "z-dihedrals": [122.27, 180.0, 90.0, -176.46, -175.49, 175.32, -0.19, -178.98],
        "z-lengths": [1.56, 1.51, 1.41, 1.41, 1.4, 1.4, 1.4, 1.41],
    },
    "VAL": {
        "atoms": ["CB", "CG1", "CG2"],
        "chi_indices": [1],
        "parents": [["N", "C", "CA"], ["N", "CA", "CB"], ["CG1", "CA", "CB"]],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT1",
            "CG1": "CT3",
            "CG2": "CT3",
            "N": "NH1",
            "O": "O",
        },
        "z-angles": [111.23, 113.97, 112.17],
        "z-dihedrals": [122.95, 180.0, 123.99],
        "z-lengths": [1.57, 1.54, 1.54],
    },
}


# our constants
# elements
periodic_table = [ # Periodic Table
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]

# amino acids
aas = [
    ('G', 'GLY'), ('A', 'ALA'), ('V', 'VAL'), ('L', 'LEU'),
    ('I', 'ILE'), ('F', 'PHE'), ('W', 'TRP'), ('Y', 'TYR'),
    ('D', 'ASP'), ('H', 'HIS'), ('N', 'ASN'), ('E', 'GLU'),
    ('K', 'LYS'), ('Q', 'GLN'), ('M', 'MET'), ('R', 'ARG'),
    ('S', 'SER'), ('T', 'THR'), ('C', 'CYS'), ('P', 'PRO')
]

# backbone atoms
backbone_atoms = ['N', 'CA', 'C', 'O']

# backbone bonds
# 1: single bond
# 2: double bond
# 3: triple bond
# 4: conjugate system (e.g. aromatic)
backbone_bonds = [
    ('N', 'CA', 1),
    ('CA', 'C', 1),
    ('C', 'O', 4) # conjugate with adjacent N
]

# side-chain atoms
sidechain_atoms = { symbol: AA_GEOMETRY[aa]['atoms'] for symbol, aa in aas }
# sidechain_atoms = {
#     'G': [],   # -H
#     'A': ['CB'],  # -CH3
#     'V': ['CB', 'CG1', 'CG2'],  # -CH-(CH3)2
#     'L': ['CB', 'CG', 'CD1', 'CD2'],  # -CH2-CH(CH3)2
#     'I': ['CB', 'CG1', 'CG2', 'CD1'], # -CH(CH3)-CH2-CH3
#     'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],  # -CH2-C6H5
#     'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],  # -CH2-C8NH6
#     'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],  # -CH2-C6H4-OH
#     'D': ['CB', 'CG', 'OD1', 'OD2'],  # -CH2-COOH
#     'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],  # -CH2-C3H3N2
#     'N': ['CB', 'CG', 'OD1', 'ND2'],  # -CH2-CONH2
#     'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'],  # -(CH2)2-COOH
#     'K': ['CB', 'CG', 'CD', 'CE', 'NZ'],  # -(CH2)4-NH2
#     'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'],  # -(CH2)-CONH2
#     'M': ['CB', 'CG', 'SD', 'CE'],  # -(CH2)2-S-CH3
#     'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],  # -(CH2)3-NHC(NH)NH2
#     'S': ['CB', 'OG'],  # -CH2-OH
#     'T': ['CB', 'OG1', 'CG2'],  # -CH(CH3)-OH
#     'C': ['CB', 'SG'],  # -CH2-SH
#     'P': ['CB', 'CG', 'CD'],  # -C3H6
# }

# side-chain bonds
sidechain_bonds = {
    'G': [],
    'A': [('CA', 'CB', 1)],
    'V': [('CA', 'CB', 1), ('CB', 'CG1', 1), ('CB', 'CG2', 1)],
    'L': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'CD1', 1), ('CG', 'CD2', 1)],
    'I': [('CA', 'CB', 1), ('CB', 'CG1', 1), ('CB', 'CG2', 1), ('CG1', 'CD1', 1)],
    'F': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'CD1', 4), ('CG', 'CD2', 4), ('CD1', 'CE1', 4), ('CD2', 'CE2', 4), ('CE1', 'CZ', 4), ('CE2', 'CZ', 4)],
    'W': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'CD1', 4), ('CG', 'CD2', 4), ('CD1', 'NE1', 4), ('CD2', 'CE2', 4), ('CD2', 'CE3', 4), ('CE2', 'NE1', 4),
          ('CE2', 'CZ2', 4), ('CZ2', 'CH2', 4), ('CE3', 'CZ3', 4), ('CZ3', 'CH2', 4)],
    'Y': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'CD1', 4), ('CG', 'CD2', 4), ('CD1', 'CE1', 4), ('CD2', 'CE2', 4), ('CE1', 'CZ', 4), ('CE2', 'CZ', 4), ('CZ', 'OH', 1)],
    'D': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'OD1', 4), ('CG', 'OD2', 4)],
    'H': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'ND1', 4), ('CG', 'CD2', 4), ('ND1', 'CE1', 4), ('CD2', 'NE2', 4), ('CE1', 'NE2', 4)],
    'N': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'OD1', 4), ('CG', 'ND2', 4)],
    'E': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'CD', 1), ('CD', 'OE1', 4), ('CD', 'OE2', 4)],
    'K': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'CD', 1), ('CD', 'CE', 1), ('CE', 'NZ', 1)],
    'Q': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'CD', 1), ('CD', 'OE1', 4), ('CD', 'NE2', 4)],
    'M': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'SD', 1), ('SD', 'CE', 1)],
    'R': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'CD', 1), ('CD', 'NE', 1), ('NE', 'CZ', 4), ('CZ', 'NH1', 4), ('CZ', 'NH2', 4)],
    'S': [('CA', 'CB', 1), ('CB', 'OG', 1)],
    'T': [('CA', 'CB', 1), ('CB', 'OG1', 1), ('CB', 'CG2', 1)],
    'C': [('CA', 'CB', 1), ('CB', 'SG', 1)],
    'P': [('CA', 'CB', 1), ('CB', 'CG', 1), ('CG', 'CD', 1), ('CD', 'N', 1)]
}

# atoms for defining chi angles on the side chains
chi_angles_atoms = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

# amino acid smiles
aa_smiles = {
    'G': 'C(C(=O)O)N',
    'A': 'O=C(O)C(N)C',
    'V': 'CC(C)[C@@H](C(=O)O)N',
    'L': 'CC(C)C[C@@H](C(=O)O)N',
    'I': 'CC[C@H](C)[C@@H](C(=O)O)N',
    'F': 'NC(C(=O)O)Cc1ccccc1',
    'W': 'c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N',
    'Y': 'N[C@@H](Cc1ccc(O)cc1)C(O)=O',
    'D': 'O=C(O)CC(N)C(=O)O',
    'H': 'O=C([C@H](CC1=CNC=N1)N)O',
    'N': 'NC(=O)CC(N)C(=O)O',
    'E': 'OC(=O)CCC(N)C(=O)O',
    'K': 'NCCCC(N)C(=O)O',
    'Q': 'O=C(N)CCC(N)C(=O)O',
    'M': 'CSCC[C@H](N)C(=O)O',
    'R': 'NC(=N)NCCCC(N)C(=O)O',
    'S': 'C([C@@H](C(=O)O)N)O',
    'T': 'C[C@H]([C@@H](C(=O)O)N)O',
    'C': 'C([C@@H](C(=O)O)N)S',
    'P': 'OC(=O)C1CCCN1'
}