from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Molecule
from ase.build import molecule

import numpy as np

from Jworkflow.utility import screen_print

dataset_molecule = {
    'Mo': {'uniform_name': 'Mo', 'structure': Molecule(['Mo'], [[0, 0, 0]]), 'is_translate': True, 'rotate': None},
    'N': {'uniform_name': 'N', 'structure': Molecule(['N'], [[0, 0, 0]]), 'is_translate': True, 'rotate': None},
    'H': {'uniform_name': 'H', 'structure': Molecule(['H'], [[0, 0, 0]]), 'is_translate': True, 'rotate': None},
    'O': {'uniform_name': 'O', 'structure': Molecule(['O'], [[0, 0, 0]]), 'is_translate': True, 'rotate': None},
    'N2v': {'uniform_name': 'N2', 'structure': AseAtomsAdaptor.get_molecule(molecule('N2')), 'is_translate': True, 'rotate': None},
    'N2h': {'uniform_name': 'N2', 'structure': AseAtomsAdaptor.get_molecule(molecule('N2')), 'is_translate': True, 'rotate': [['all', np.pi / 2, [1, 0, 0], [0, 0, 0]]]},
    'NH': {'uniform_name': 'NH', 'structure': AseAtomsAdaptor.get_molecule(molecule('NH')), 'is_translate': True, 'rotate': [['all', np.pi, [1, 0, 0], [0, 0, 0]]]},
    'NH2': {'uniform_name': 'NH2', 'structure': AseAtomsAdaptor.get_molecule(molecule('NH2')), 'is_translate': True, 'rotate': [['all', np.pi, [1, 0, 0], [0, 0, 0]]]},
    'NH3': {'uniform_name': 'NH3', 'structure': AseAtomsAdaptor.get_molecule(molecule('NH3')), 'is_translate': True,
            'rotate': [['all', np.pi, [1, 0, 0], [0, 0, 0]], ['all', np.pi, [0, 0, -1], [0, 0, 0]]]},
    'NNHv': {'uniform_name': 'NNH', 'structure': Molecule(['N', 'N', 'H'], [[0, 0, 0], [0, 0, 1.2], [0, 0, 2.2]]), 'is_translate': True, 'rotate': None},
    'NNHi': {'uniform_name': 'NNH', 'structure': Molecule(['N', 'N', 'H'], [[0, 0, 0], [0, 0, 1.2], [0, 0.9, 1.65]]), 'is_translate': True, 'rotate': None},
    'NNHs': {'uniform_name': 'NNH', 'structure': Molecule(['N', 'N', 'H'], [[0, 0.2, 0], [0, 0.66, 0.8], [0, 0.06, 1.7]]), 'is_translate': True, 'rotate': None},
    'NNHh': {'uniform_name': 'NNH', 'structure': Molecule(['N', 'N', 'H'], [[0, 0.6, 0], [0, -0.6, 0], [0, 1.2, 0.7]]), 'is_translate': True, 'rotate': None},
    'NNH2v': {'uniform_name': 'NNH2', 'structure': Molecule(['N', 'N', 'H', 'H'], [[0, 0, 0.1], [0, 0, 1.5], [0, 0.87, -0.6], [0, -0.87, -0.6]]), 'is_translate': True,
              'rotate': [['all', np.pi, [1, 0, 0], [0, 0, 0]]]},
    'NNH2h': {'uniform_name': 'NNH2', 'structure': Molecule(['N', 'N', 'H', 'H'], [[0, 0.7, 0], [0, -0.7, 0], [0.8, 1.3, 0.4], [-0.8, 1.3, 0.4]]), 'is_translate': True,
              'rotate': None},
    'NHNH': {'uniform_name': 'NHNH', 'structure': Molecule(['N', 'N', 'H', 'H'], [[0, 0.03, 0.14], [0, 0.9, 1.06], [0.1, 1.66, 0.54], [-0.1, -0.76, 0.75]]), 'is_translate': True,
             'rotate': None},
    'NHNHh': {'uniform_name': 'NHNH', 'structure': Molecule(['N', 'N', 'H', 'H'], [[0, 0.7, 0], [0, -0.7, 0], [0.1, 1.46, 0.7], [-0.1, -1.46, 0.7]]), 'is_translate': True,
              'rotate': None},
    'NHNH2': {'uniform_name': 'NHNH2', 'structure': Molecule(['N', 'N', 'H', 'H', 'H'], [[0, 0, 0], [0, 1.0, 0.93], [0.87, 1.42, 1.26], [-0.81, 1.42, 1.26], [0, -0.8, 0.7]]),
              'is_translate': True, 'rotate': None},
    'NHNH2h': {'uniform_name': 'NHNH2', 'structure': Molecule(['N', 'N', 'H', 'H', 'H'], [[0, 0.7, 0], [0, -0.7, 0], [0.8, 1.3, 0.4], [-0.8, 1.3, 0.4], [0, -1.5, 0.7]]),
               'is_translate': True, 'rotate': None},
    'NH2NH2': {'uniform_name': 'NH2NH2', 'structure': Molecule(['N', 'N', 'H', 'H', 'H', 'H'],
                                                               [[0, -0.05, -0.05], [0, 0.7, 0.95], [0.8, 1.35, 0.75], [-0.8, 1.35, 0.75], [0.8, -0.6, 0.15], [-0.8, -0.6, 0.15]]),
               'is_translate': True, 'rotate': None},
    'NH2NH2h': {'uniform_name': 'NH2NH2',
                'structure': Molecule(['N', 'N', 'H', 'H', 'H', 'H'], [[0, -0.7, 0], [0, 0.7, 0], [0.8, 1.3, 0.4], [-0.8, 1.3, 0.4], [0.8, -1.3, 0.4], [-0.8, -1.3, 0.4]]),
                'is_translate': True, 'rotate': None},

    'NO3': {'uniform_name': 'NO3', 'structure': Molecule(['N', 'O', 'O', 'O'], [[0, 1.12, 0.643], [0, 2.24, 0], [0, 1.12, 1.86], [0, 0, 0]]), 'is_translate': False, 'rotate': None},
    'NO3c': {'uniform_name': 'NO3', 'structure': Molecule(['N', 'O', 'O', 'O'], [[0, 0, 0.643], [0, 1.12, 0], [0, 0, 1.86], [0, -1.12, 0]]), 'is_translate': True, 'rotate': None},
    'NO3H': {'uniform_name': 'NO3H', 'structure': Molecule(['N', 'O', 'O', 'O', 'H'], [[0, 1.12, 0.643], [0, 2.24, 0], [0, 1.12, 1.86], [0, 0, 0], [0, 1.12, 2.8]]), 'is_translate': False, 'rotate': None},
    'NO3Hc': {'uniform_name': 'NO3H', 'structure': Molecule(['N', 'O', 'O', 'O', 'H'], [[0, 0, 0.643], [0, 1.12, 0], [0, 0, 1.86], [0, -1.12, 0], [0, 0, 2.8]]), 'is_translate': True, 'rotate': None},
    'NO2': {'uniform_name': 'NO2', 'structure': Molecule(['N', 'O', 'O'], [[0, 1.08, 0.648], [0, 2.166, 0], [0, 0, 0]]), 'is_translate': False, 'rotate': None},
    'NO2c': {'uniform_name': 'NO2', 'structure': Molecule(['N', 'O', 'O'], [[0, 0, 0.648], [0, 1.08, 0], [0, -1.08, 0]]), 'is_translate': True, 'rotate': None},
    'NO2h': {'uniform_name': 'NO2', 'structure': Molecule(['N', 'O', 'O'], [[0, 0, 0], [0, -0.62, 1.], [0, 1.3, 0.]]), 'is_translate': False, 'rotate': None},
    'NO2Hh': {'uniform_name': 'NO2H', 'structure': Molecule(['N', 'O', 'O', 'H'], [[0, 0, 0], [0, -0.62, 1.], [0, 1.3, 0.], [0, -0.62, 1.96]]), 'is_translate': False, 'rotate': None},
    'NO': {'uniform_name': 'NO', 'structure': Molecule(['N', 'O'], [[0, 0, 0], [0, 0, 1.23]]), 'is_translate': True, 'rotate': None},
    'NOH': {'uniform_name': 'NOH', 'structure': Molecule(['N', 'O', 'H'], [[0, 0, 0], [0, 0, 1.4], [0, 0.92, 1.7]]), 'is_translate': True, 'rotate': None},

    'CO': {'uniform_name': 'CO', 'structure': Molecule(['C', 'O'], [[0, 0, 0], [0, 0, 1.2]]), 'is_translate': True, 'rotate': None},
    'H2O': {'uniform_name': 'H2O', 'structure': Molecule(['O', 'H', 'H'], [[0, 0, 0], [0, -0.77, 0.59], [0, 0.77, 0.59]]), 'is_translate': True, 'rotate': None},
    'OH': {'uniform_name': 'OH', 'structure': Molecule(['O', 'H'], [[0, 0, 0], [0, 0, 0.97]]), 'is_translate': True, 'rotate': None},
    'OHi': {'uniform_name': 'OH', 'structure': Molecule(['O', 'H'], [[0, 0, 0], [0, 0.7, 0.7]]), 'is_translate': True, 'rotate': None},
    'HCOO': {'uniform_name': 'HCOO', 'structure': Molecule(['H', 'C', 'O', 'O'], [[0, 0, 1.1], [0, 0, 0], [0, -1.1, -0.5], [0, 1.2, -0.5]]), 'is_translate': True, 'rotate': None},
    'COOH': {'uniform_name': 'COOH', 'structure': Molecule(['C', 'O', 'O', 'H'], [[0, 0, 0], [0, -1.1, 0.59], [0, 1.2, 0.62], [0, 1.9, 0.01]]), 'is_translate': True, 'rotate': None},
    'HCOOHc': {'uniform_name': 'HCOOH', 'structure': Molecule(['H', 'C', 'O', 'O', 'H'], [[0., 0.67, 1.94], [0., 0.89, 0.85], [0, 0, 0], [0, 2.16, 0.64], [0, 2.36, -0.34]]), 'is_translate': False, 'rotate': None},
    'HCOOH': {'uniform_name': 'HCOOH', 'structure': AseAtomsAdaptor.get_molecule(molecule('HCOOH')), 'is_translate': True,
              'rotate': [['all', np.pi / 2, [1, 0, 0], [0, 0, 0]], ['all', np.pi / 9, [0, 1, 0], [0, 0, 0]]]},

    'CH3CH2OH': {'uniform_name': 'CH3CH2OH', 'structure': Molecule(['O', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
                                                                   [[0, 0, 0], [0, 2.1, 1.21], [-0.84, 0.9, 0.77], [-0.43, -0.87, -0.04], [-1.24, 0.36, 1.63], [-1.68, 1.23, 0.16],
                                                                    [-0.62, 2.8, 1.74], [0.8, 1.75, 1.88], [0.44, 2.58, 0.35]]), 'is_translate': False, 'rotate': None},
    'CH3CHOH': {'uniform_name': 'CH3CHOH', 'structure': Molecule(['O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'],
                                                                 [[-0.63, -1.23, 0.21], [0, 0, 0], [-0.64, 1.15, 0.74], [-1.68, 1.29, 0.46], [-0.11, 2.08, 0.55],
                                                                  [1.05, -0.15, 0.23], [-1.6, -1.1, 0.18], [-0.6, 0.94, 1.82]]), 'is_translate': True,
                'rotate': [['all', 150 / 180 * np.pi, [0, 0, -1], [0, 0, 0]]]},
    'CH3CHO': {'uniform_name': 'CH3CHO', 'structure': Molecule(['O', 'C', 'C', 'H', 'H', 'H', 'H'],
                                                               [[0, 1.25, -0.34], [0, 0, 0], [-2.82, 6.94, 0.8], [-1.86, 7.26, 0.4], [-2.85, 5.85, 0.87], [3.5, 6.72, 0.19],
                                                                [-2.9, 7.35, 1.81]]), 'is_translate': False, 'rotate': None},
    'CH3COHv': {'uniform_name': 'CH3COH', 'structure': Molecule(['O', 'C', 'C', 'H', 'H', 'H', 'H'],
                                                                [[0, 1.28, 0.51], [0, 0, 0], [-0.1, -1, 1.1], [0, 1.29, 1.49], [-1, -0.86, 1.69], [-0.11, -2, 0.72],
                                                                 [0.77, -0.92, 1.78]]), 'is_translate': True, 'rotate': None},
    'CH3COHm': {'uniform_name': 'CH3COH', 'structure': Molecule(['O', 'C', 'C', 'H', 'H', 'H', 'H'],
                                                                [[0, 1, 0.92], [0, 0, 0], [-0.11, -1.33, 0.69], [0, 1.78, 0.3], [-1, -1.37, 1.31], [-0.14, -2.16, 0],
                                                                 [0.76, -1.49, 1.36]]), 'is_translate': True, 'rotate': None},
    'CH3COOH': {'uniform_name': 'CH3COOH', 'structure': Molecule(['O', 'O', 'C', 'C', 'H', 'H', 'H', 'H'],
                                                                 [[0, 0, 0], [0, 0.75, 2.08], [0, -1.68, 1.65], [0, -0.25, 1.18], [1, -2, 1.76], [-0.51, -1.75, 2.6],
                                                                  [-0.52, -2.3, 0.92], [0, 1.65, 1.73]]), 'is_translate': True, 'rotate': None},
    'CH3CO': {'uniform_name': 'CH3CO', 'structure': Molecule(['O', 'C', 'C', 'H', 'H', 'H'], [[0, 1.19, 0.23], [0, 0, 0], [0, -1, 1.1], [0.14, -0.57, 2.05], [0.63, -1.85, 0.89], [-1.08, -1.46, 1.1]]), 'is_translate': True, 'rotate': None},
    'CH2CO': {'uniform_name': 'CH2CO', 'structure': Molecule(['O', 'C', 'C', 'H', 'H'], [[0, 2.3, 0.95], [0, 0, 0.03], [0, 1.5, 0], [0.93, -0.3, 0.5], [-0.93, -0.3, 0.5]]), 'is_translate': True, 'rotate': None},
    'COOHz': {'uniform_name': 'COOH', 'structure': Molecule(['C', 'O', 'O', 'H'], [[0, 0, 0], [0, -1.24, 0.19], [0, 0.79, 1.1], [0, 1.74, 0.83]]), 'is_translate': True, 'rotate': None},
    'COOHv': {'uniform_name': 'COOH', 'structure': Molecule(['C', 'O', 'O', 'H'], [[0, 0, 0], [0, -1, 0.68], [0, 1.25, 0.58], [0, 1.12, 1.55]]), 'is_translate': True, 'rotate': None},
    'CO2': {'uniform_name': 'CO2', 'structure': Molecule(['C', 'O', 'O'], [[0, 0, 0], [0, 1, 0.64], [0, -1, 0.64]]), 'is_translate': True, 'rotate': None},
    'CH3': {'uniform_name': 'CH3', 'structure': Molecule(['C', 'H', 'H', 'H'], [[0, 0, 0], [0.88, -0.52, 0.36], [0.00, 1.03, 0.36], [-0.90, -0.50, 0.36]]), 'is_translate': True, 'rotate': None},
            
    'CH3OHv': {'uniform_name': 'CH3OH', 'structure': Molecule(['C', 'O', 'H', 'H', 'H', 'H'], [[0, 0, 1.42], [0, 0, 0], [0, -1.05, 1.73], [0, 0.93, -0.29], [0.89, 0.48, 1.84], [-0.89, 0.48, 1.84]]), 'is_translate': False, 'rotate': None},
    'CH3OHi': {'uniform_name': 'CH3OH', 'structure': Molecule(['C', 'O', 'H', 'H', 'H', 'H'], [[0, 1.36, 0.74], [0, 0, 0], [0.56, 2.2, 0.3], [0.29, 1.21, 1.78], [0.96, -0.14, -0.09], [-1.06, 1.56, 0.66]]), 'is_translate': False, 'rotate': None},
    'CH3Ov': {'uniform_name': 'CH3O', 'structure': Molecule(['C', 'O', 'H', 'H', 'H'], [[0, 0, 1.39], [0, 0, 0], [0, 1.06, 1.67], [0.9, -0.46, 1.8], [-0.9, -0.46, 1.8]]), 'is_translate': True, 'rotate': None},
    'CH3Oi': {'uniform_name': 'CH3O', 'structure': Molecule(['C', 'O', 'H', 'H', 'H'], [[0, 1.12, 0.86], [0, 0, 0], [0.89, 1.75, 0.73], [0, 0.71, 1.88], [-0.9, 1.74, 0.74]]), 'is_translate': True, 'rotate': None},
    'CH2O': {'uniform_name': 'CH2O', 'structure': Molecule(['C', 'O', 'H', 'H'], [[0, 1.33, 0.07], [0, 0, 0], [-0.96, 1.78, 0.39], [0.96, 1.78, 0.39]]), 'is_translate': True, 'rotate': None},
    'CHO': {'uniform_name': 'CHO', 'structure': Molecule(['C', 'O', 'H'], [[0, 0, 0], [0, 1.08, 0.54], [0, -0.93, 0.62]]), 'is_translate': True, 'rotate': None},
    
    'C6H5CHO': {'uniform_name': 'C6H5CHO', 'structure': Molecule(['H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'O'], [[-0.1, 3.61, 5.44], [0.03, 4.39, 3.09], [0.08, 2.74, 1.23], [0.1, -1.08, 1.75], [-0.06, -0.49, 4.1], [-0.15, 1.17, 5.96], [-0.08, 2.88, 4.63], [-0.0, 3.32, 3.3], [-0.1, 1.51, 4.93], [-0.01, 1.02, 2.55], [0.02, 2.4, 2.26], [0.03, 0.03, 1.49], [-0.05, 0.58, 3.89], [-0.0, 0.0, -0.0]]), 'is_translate': True, 'rotate': None},
    'C6H5CHOH': {'uniform_name': 'C6H5CHOH', 'structure': Molecule(['H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'O'], [[-0.09, 1.28, 6.11], [0.02, 2.83, 4.16], [0.11, 1.97, 1.88], [0.04, -1.77, 0.99], [-0.06, -2.09, 3.42], [-0.12, -1.19, 5.72], [-0.1, 0.95, 0.22], [-0.06, 0.88, 5.09], [-0.0, 1.75, 4.0], [-0.08, -0.51, 4.87], [0.01, -0.15, 2.45], [0.03, 1.26, 2.7], [0.02, -0.7, 1.17], [-0.04, -1.01, 3.59], [0.0, -0.0, 0.0]]), 'is_translate': True, 'rotate': None},
    'C6H5CH2O': {'uniform_name': 'C6H5CH2O', 'structure': Molecule(['H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'O'], [[-0.11, 3.76, 5.38], [-0.01, 4.44, 2.99], [0.02, 2.67, 1.2], [-0.89, -0.66, 1.82], [-0.14, -0.37, 4.2], [-0.17, 1.36, 6.0], [0.8, -0.61, 1.8], [-0.09, 3.0, 4.6], [-0.04, 3.38, 3.26], [-0.13, 1.65, 4.94], [-0.06, 1.07, 2.61], [-0.02, 2.42, 2.27], [-0.03, 0.03, 1.54], [-0.11, 0.69, 3.95], [0.0, -0.0, -0.0]]), 'is_translate': True, 'rotate': None},
    'C6H5CH2OH': {'uniform_name': 'C6H5CH2OH', 'structure': Molecule(['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'O'], [[-0.1, 1.26, 6.08], [0.02, 2.77, 4.11], [0.1, 1.86, 1.84], [-0.78, -1.54, 1.0], [-0.07, -2.14, 3.43], [-0.12, -1.21, 5.73], [-0.1, 0.94, 0.27], [0.83, -1.54, 1.0], [-0.07, 0.84, 5.08], [-0.01, 1.69, 3.97], [-0.08, -0.54, 4.87], [-0.0, -0.23, 2.45], [0.02, 1.17, 2.68], [0.02, -0.79, 1.12], [-0.05, -1.06, 3.59], [-0.0, -0.0, 0.0]]), 'is_translate': True, 'rotate': None},
    'C6H5COOH': {'uniform_name': 'C6H5COOH', 'structure': Molecule(['H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'O', 'O'], [[-0.23, 3.27, 5.7], [-0.11, 4.2, 3.39], [-0.01, 2.65, 1.43], [-0.11, -0.72, 4.09], [-0.23, 0.81, 6.05], [0.12, -1.77, 0.88], [-0.18, 2.6, 4.84], [-0.11, 3.12, 3.55], [-0.18, 1.22, 5.04], [-0.05, 0.88, 2.65], [-0.05, 2.27, 2.45], [0.02, 0.01, 1.45], [-0.11, 0.36, 3.95], [-0.0, -0.0, -0.0], [0.08, -1.32, 1.76]]), 'is_translate': True, 'rotate': None},


}

dataset_reaction = {
    'HER': {
        'reac': '* + 2xH + 2xe',
        'H': '*H + H + e',
        'prod': '* + H2(g)'
    },
    'NRR': {
        'reac': '* + N2(g) + 6xH + 6xe',
        'N2': '*N2 + 6xH + 6xe',
        'NNH': '*NNH + 5xH + 5xe',
        'NNH2': '*NNH2 + 4xH + 4xe',
        'NHNH': '*NHNH + 4xH + 4xe',
        'N': '*N + 3xH + 3xe + NH3(g)',
        'NHNH2': '*NHNH2 + 3xH + 3xe',
        'NH2NH2': '*NH2NH2 + 2xH + 2xe',
        'NH': '*NH + 2xH + 2xe + NH3(g)',
        'NH2': '*NH2 + H + e + NH3(g)',
        'NH3': '*NH3 + NH3(g)',
        'prod': '* + 2xNH3(g)'
    },
    'NRRa': {
        'reac': '* + N2(g) + 6xH + 6xe',
        'N2': '*N2 + 6xH + 6xe',
        'NNH': '*NNH + 5xH + 5xe',
        'NNH2': '*NNH2 + 4xH + 4xe',
        'NHNH': '*NHNH + 4xH + 4xe',
        'N': '*N + 6xH + 6xe + 0.5xN2(g)',
        'NHNH2': '*NHNH2 + 3xH + 3xe',
        'NH2NH2': '*NH2NH2 + 2xH + 2xe',
        'NH': '*NH + 5xH + 5xe + 0.5xN2(g)',
        'NH2': '*NH2 + 4xH + 4xe + 0.5xN2(g)',
        'NH3': '*NH3 + 3xH + 3xe + 0.5xN2(g)',
        'prod': '* + 2xNH3(g)'
    },
    'NO3RR': {
        'reac': '* + NO3f + 9xH + 9xe',
        'NO3': '*NO3 + 9xH + 9xe',
        'NO3H': '*NO3H + 8xH + 8xe',
        'NO2': '*NO2 + 7xH + 7xe + H2O(l)',
        'NO2H': '*NO2H + 6xH + 6xe + H2O(l)',
        'NO': '*NO + 5xH + 5xe + 2xH2O(l)',
        'NOH': '*NOH + 4xH + 4xe + 2xH2O(l)',
        'N': '*N + 3xH + 3xe + 3xH2O(l)',
        'NH': '*NH + 2xH + 2xe + 3xH2O(l)',
        'NH2': '*NH2 + 1xH + 1xe + 3xH2O(l)',
        'NH3': '*NH3 + 3xH2O(l)',
        'prod': 'NH3(g) + 3xH2O(l)'
    },
    'EOR': {
        'reac': '* + CH3CH2OH(l) + 3xH2O(l)',
        'CH3CH2OH': '*CH3CH2OH + 3xH2O(l)',
        'CH3CHOH': '*CH3CHOH + 3xH2O(l) + H + e',
        'CH3COH': '*CH3COH + 3xH2O(l) + 2xH + 2xe',
        'CH3CO': '*CH3CO + 3xH2O(l) + 3xH + 3xe',
        'CH2CO': '*CH2CO + 3xH2O(l) + 4xH + 4xe',
        'CH3COOH': '*CH3COOH + 2xH2O(l) + 4xH + 4xe',
        'CH3COOH(l)': '* + CH3COOH(l) + 2xH2O(l) + 4xH + 4xe',

        'CO': '*CO + H2O(l) + CO2(l) + 10xH + 10xe',
        'COOH': '*COOH + CO2(l) + 11xH + 11xe',
        'CO2': '*CO2 + CO2(l) + 12xH + 12xe',
        'CO2(l)': '* + 2xCO2(l) + 12xH + 12xe'
        #         'CH3&CO'     : '*CH3_CO + 3xH2O(l) + 3xH + 3xe',
        #         'CH3&COOH'   : '*CH3_COOH + 2xH2O(l) + 4xH + 4xe',
        #         'CH3&CO2'    : '*CH3_CO2 + 2xH2O(l) + 5xH + 5xe',
        #         'CH3&CO2(l)' : '*CH3 + CO2(l) + 2xH2O(l) + 5xH + 5xe',
    },
    'MOR': {
        'reac': '* + CH3OH(g) + H2O(l)',
        'CH3OH': '*CH3OH + H2O(l)',
        'CH3O': '*CH3O + H2O(l) + 1xH + 1xe',
        'CH2O': '*CH2O + H2O(l) + 2xH + 2xe',
        'CHO': '*CHO + H2O(l) + 3xH + 3xe',
        'HCOOH': '*HCOOH + 4xH + 4xe',
        'COOH': '*COOH + 5xH + 5xe',
        'HCOO': '*HCOO + 5xH + 5xe',
        'CO2': '*CO2 + 6xH + 6xe',
        'prod': '* + CO2(g) + 6xH + 6xe'
    }
}

dataset_energy = {
    'H2': -6.77019107, # -6.77104774 20x20x20 520 G1 1E-7 -0.001 0 0.01 PBE-D3
    'H': -6.77019107 / 2,
    'H2O': -14.21961940, # 20x20x20 520 G1 1E-7 -0.001 0 0.01 PBE-D3

    'N2': -16.63315679, # -16.63517878 20x20x20 520 G1 1E-7 -0.001 0 0.01 PBE-D3
    'NH3': -19.53883747, # -19.53943296 20x20x20 520 G1 1E-7 -0.001 0 0.01 PBE-D3
    'HNO3': -28.61173801, # 20x20x20 520 G1 1E-7 -0.001 0 0.01 PEB-D3
    'NO3f': -25.226642475,# E(HNO3(g)) - 0.5 E(H2(g))

    'CH3CH2OH': -46.92672187, # 20x20x20 520 G1 1E-7 -0.001 0 0.01 PBE-D3
    'CH3COOH': -46.77247686, # 20x20x20 520 G1 1E-7 -0.001 0 0.01 PBE-D3
    
    'CH3OH': -30.22541492, # 20x20x20 520 G1 1E-7 -0.001 0 0.01 PBE-D3
    'CO2': -22.95767204, # 24x22.5x21.5 520 G1 1E-7 -0.001 0 0.01 PBE-D3
}

dataset_Gcor = { # 298.15k 1bar
    'H2(g)': -0.121660, # -0.05806830282099462 CCCBDB 0.257961615785502(ZPE)-0.3160299186064966(H-TS)
    'H': -0.121660 / 2,
    'H2O(g)': 0.07749534868389718, # CCCBDB 0.5584250300384028(ZPE)--0.48092968135450564(H-TS)
    'H2O(l)': -0.008634651316102832,

    'N2(g)': -0.352809, # -0.35779568903955816 CCCBDB 0.14444164298284623(ZPE)-0.5022373320224044(H-TS)
    'NH3(g)': 0.388528, # 0.4028615132745161 CCCBDB 0.8944843204289648(ZPE)-0.49162280715444867(H-TS)
    'HNO3(g)': -0.005, # -0.004954969638454765 CCCBDB 0.6966054687546107(ZPE)-0.7015604383930655(H-TS) 
    'NO3f': -0.3679658485895027,# G(HNO3(g)) - 0.5 G(H2(g)) - Gcorrect(0.392) https://doi.org/10.1002/adfm.202008533

    'CH3CH2OH(g)': 1.3800252076253468, # CCCBDB 2.1038142272876637(ZPE)-0.7237890196623169(H-TS)
    'CH3COOH(g)': 0.8681174785783672, # CCCBDB 1.6031162607452374(ZPE)-0.7349987821668702(H-TS)
    
    'CH3OH(g)': 0.7223861532545338, # CCCBDB 1.345043059141122(ZPE)-0.6226569058865882(H-TS)
    'CO2(g)': -0.252672095218762 # CCCBDB 0.3109524812025565(ZPE)-0.5636245764213185(H-TS)
}


def adsb_to_reaction(adsb):
    '''
    Function to find which reaction includs the specified adsbrate

    Parameter:
        - adsb: uniform name of the adsb / str
    Return:
        - List of reactions that include the adsb
    '''
    reactions = []
    for reaction in dataset_reaction.keys():
        if adsb in dataset_reaction[reaction].keys():
            reactions.append(reaction)
    return reactions


def reaction_energy(reaction, product, energy_dict={}, Gcor_dict={}, U=0, PH=0, refer_state='reac', print_info=False):
    '''
    Function to calculate adsorptioen energy use reactions stroed in dataset

    Parameters:
        - reaction: Reaction type / str, e.g. 'NRR', 'EOR'
        - product: Main product name / str, e.g. 'NHNH', 'CH3&CO'
        - energy_dict: Dict of all total energies / dict, e.g. {'*':-200,'H':-3,'e':0,'NH3(g)':-21,'*NH':-220}
        - Gcor_dict: Dict of all free energy correction terms / dict, e.g. {'H':-1.2,'e':0,'NH3(g)':-0.6,'*NH':-2}
        - U: Applied voltage, make {'e': -1 * U} / float, default 0
        - PH: PH correction, make {'H': - 0.06 * PH} / float, default 0
        - refer_state: In which referential state to calculate the energy / str, default 'reac'
        - print_info: Whether to print information for inspection or not / bool, default False
    Return:
        - The energy of a particular reaction product with respect to the reference state
    Cautions:
        - You should pass an empty dict {} to Gcor_dict to perform energy calculation only
    '''
    # gain refer and product state from dataset_reaction
    refer = dataset_reaction[reaction][refer_state]
    products = dataset_reaction[reaction][product]
    # obtain all variable in reaction and form a dict
    all_species = {}
    for reaction in [refer, products]:
        for pair in reaction.split('+'):
            varia = sep_coeff_varia(pair)[1]
            all_species[varia] = 0
    # dupate energyies for each variable according to dict and dataset
    # update energy from energy_dict
    if len(energy_dict) > 0:
        for varia, value in energy_dict.items():
            all_species[varia] += value
    # update energy from Gcor_dict
    if len(Gcor_dict) > 0:
        for varia, value in Gcor_dict.items():
            all_species[varia] += value
    # update energy from dataset_energy and dataset_Gcor
    for varia, value in dataset_Gcor.items():
        if varia in all_species.keys() and all_species[varia] == 0:
            if len(energy_dict) > 0:
                all_species[varia] += dataset_energy[varia.split('(')[0]]
            if len(Gcor_dict) > 0:
                all_species[varia] += value
    # update energy with applied U and PH
    if 'e' in all_species.keys() and U != 0:
        all_species['e'] += -1 * U
    if 'H' in all_species.keys() and PH != 0:
        all_species['H'] += -0.06 * PH
    # calculate reaction energy
    step = sum(coeff_varia[0] * float(all_species[coeff_varia[1]]) for coeff_varia in [sep_coeff_varia(pair) for pair in products.split('+')]) \
           - sum(coeff_varia[0] * float(all_species[coeff_varia[1]]) for coeff_varia in [sep_coeff_varia(pair) for pair in refer.split('+')])
    # print info
    if print_info:
        screen_print('Calculation_Inspection')
        screen_print('React', refer)
        screen_print('Product', products)
        screen_print('Variable', all_species)
        screen_print('Reaction energy', str(step))
        screen_print('Inspection_End')
    # return
    return step


def sep_coeff_varia(coeff_varia_pair):
    '''
    Function for splitting the coefficient and variable in a formula according to the position of the 'x' sign

    Parameters:
        - coeff_varia_pair: Formulas like '3 x N2'
    Return:
        - (coefficient, variable)
    '''
    sep = coeff_varia_pair.split('x')
    if len(sep) == 1:
        return (float(1), sep[0].replace(' ', ''))
    elif len(sep) == 2:
        return (float(sep[0].replace(' ', '')), sep[1].replace(' ', ''))


def uniform_name_dict():
    '''
    Function return all uniform and specific names of molecules in dataset

    Return:
        - A dict use uniform names as keys and their specific names as values
    '''
    uniform_name_dict = {}
    for molecule_specific in dataset_molecule.keys():
        uniform_name = dataset_molecule[molecule_specific]['uniform_name']
        if uniform_name not in uniform_name_dict.keys():
            uniform_name_dict[uniform_name] = [molecule_specific]
        elif uniform_name in uniform_name_dict.keys():
            uniform_name_dict[uniform_name].append(molecule_specific)
    return uniform_name_dict


def uniform_adsb(adsb_name):
    '''
    Function to return the uniform adsorbate name from one specific adsorbate name

    Parameter"
        - adsb_name: The specific adsorbate name / str
    Return:
        - The uniform adsorbate name
    '''
    return dataset_molecule[adsb_name]['uniform_name']


def load_molecule(molecule_name, rotation=0, axis=[0, 0, -1], anchor=[0, 0, 0]):
    '''
    Function to load molecule structure from dataset

    Parameters:
        - molecule_name: Molecule name that is used to find and load molecule structure from dataset / str
        - rotation: Rotation angle of molecule in unit ° / float or (3) list, default 0
        - axis: The axis of rotation (3) or (3,3) list, default [0,0,-1]
        - anchor: The point around which the rotation revolves in cartesian / (3) list, default [0,0,0]
    Return:
        - molecule structure, a bool of is_translate
    Cautions:
        - The default rotation direction follows the left hand rule
        - bool of the is_translate parameter in the database will be used in AdsorbateSiteFinder.add_adsorbate()
    '''
    # load molecule from dataset
    molecule_structure = dataset_molecule[molecule_name]['structure'].copy()
    if dataset_molecule[molecule_name]['rotate'] is not None:
        for rotate_operation in dataset_molecule[molecule_name]['rotate']:
            if rotate_operation[0] == 'all':
                indices = range(len(molecule_structure))
            else:
                indices = rotate_operation[0]
            molecule_structure.rotate_sites(indices=indices, theta=rotate_operation[1], axis=rotate_operation[2], anchor=rotate_operation[3])
    # apply rotation
    if rotation != 0 and type(rotation) != list:
        molecule_structure.rotate_sites(indices=range(len(molecule_structure)), theta=rotation / 180 * np.pi, axis=axis, anchor=anchor)
    elif rotation != 0 and type(rotation) == list:
        axises = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        for _, rotate in enumerate(rotation):
            molecule_structure.rotate_sites(indices=range(len(molecule_structure)), theta=rotate / 180 * np.pi, axis=axises[_], anchor=anchor)
        return molecule_structure, False
    return molecule_structure, dataset_molecule[molecule_name]['is_translate']
