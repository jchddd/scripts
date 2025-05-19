import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from scipy.ndimage import center_of_mass

from Jworkflow.dataset import sep_coeff_varia, dataset_energy, dataset_Gcor
from Jworkflow.utility import phase_name_convert


def get_reaction_energy_Gcor(data_name):
    '''
    Obtain the pre-input data for Pourbaix diagram drawing

    Parameter:
        - data_name (str). The following data can be selected at present:
          'ZnO-PWmat' from http://www.pwmat.com:8080/upload/module/pdf/Pourbaix_diagram.pdf
          'ZnO-ASE' from https://wiki.fysik.dtu.dk/ase/ase/phasediagram/phasediagram.html
          'Mo2C*O*OH' from https://pubs.acs.org/doi/10.1021/acscatal.4c04878
          'Fe@4N-Gr', 'Cr@4N-Gr', 'Mn@4N-Gr', and 'Co@4N-Gr' from https://pubs.acs.org/doi/10.1021/acscatal.3c04801
    Example:
        - reaction, energy, correction = get_reaction_energy_Gcor('ZnO-ASE')
    '''
    dataset = {
        'ZnO-PWmat':( # http://www.pwmat.com:8080/upload/module/pdf/Pourbaix_diagram.pdf, https://zhuanlan.zhihu.com/p/89907884
            {
                'Zn':     ('Zn', 'Zn'),
                'Zn+2':   ('Zn', 'Znzz + 2xe'),
                'ZnO':    ('Zn + H2O(l)', 'ZnO + 2xH + 2xe'),
                'ZnO2':   ('Zn + 2xH2O(l)', 'ZnO2 + 4xH + 4xe'),
                'ZnO2-2': ('Zn + 2xH2O(l)', 'ZnO2ff + 4xH + 2xe'),
                'ZnOH+':  ('Zn + H2O(l)', 'ZnOHz + 1xH + 2xe'),
                'HZnO2-': ('Zn + 2xH2O(l)', 'HZnO2f + 3xH + 2xe'),
            },
            {
                'Zn': -5430.28, 
                'Znzz': -5430.28 -1.525,
                'ZnO': -5865.86,
                'ZnO2': -6297.86,
                'ZnO2ff': -5430.28 -4.042 -865.44, # dG=ZnO2ff - Zn - O2 = -4.042, ZnO2ff = -4.042 + Zn + O2
                'ZnOHz': -5430.28 -3.518 -865.44/2 -31.89/2,
                'HZnO2f': -5430.28 -4.810 -865.44 -31.89/2,
                'H2O': -466.96 -0.11,
                'H': -15.795,    
            },
            {
                'Zn': 0, 
                'Znzz': 0,
                'ZnO': 0,
                'ZnO2': 0,
                'ZnO2ff': 0,
                'ZnOHz': 0,
                'HZnO2f': 0,
                'H2O(l)': 0,
                'H': 0,
            }
        ),
        'ZnO-ASE':( # https://wiki.fysik.dtu.dk/ase/ase/phasediagram/phasediagram.html
            {
                'Zn':     ('Zn', 'Zn'),
                'Zn+2':   ('Zn', 'Znzz(aq) + 2xe'),
                'ZnO':    ('Zn + H2O(l)', 'ZnO + 2xH + 2xe'),
                'ZnO2':   ('Zn + 2xH2O(l)', 'ZnO2(aq) + 4xH + 4xe'),
                'ZnO2-2': ('Zn + 2xH2O(l)', 'ZnO2ff(aq) + 4xH + 2xe'),
                'ZnOH+':  ('Zn + H2O(l)', 'ZnOHz(aq) + 1xH + 2xe'),
                'HZnO2-': ('Zn + 2xH2O(l)', 'HZnO2f(aq) + 3xH + 2xe'),         
            },
            {
                'Zn': 0,
                'Znzz': -1.5264164573008816,
                'ZnO': -3.323,
                'ZnO2': -2.921,
                'ZnO2ff': -4.045437252886342,
                'ZnOHz': -3.5207315956891634,
                'HZnO2f': -4.801273583873681,
                'H2O': -2.45831105012463,
                'H': 0.0                
            },
            {
                'Zn': 0,
                'Znzz(aq)': 0,
                'ZnO': 0,
                'ZnO2(aq)': 0,
                'ZnO2ff(aq)': 0,
                'ZnOHz(aq)': 0,
                'HZnO2f(aq)': 0,
                'H2O(l)': 0.0,
                'H': 0.0                
            }
        ),
        'Mo2C*O*OH':( # https://pubs.acs.org/doi/10.1021/acscatal.4c04878
            {
                'Mo2C*O9':     ('Mo2C-9*O', 'Mo2C-9*O'),
                'Mo2C*O8*OH':  ('Mo2C-9*O + 1xH + 1xe', 'Mo2C-8*O-1*OH'),
                'Mo2C*O7*OH2': ('Mo2C-9*O + 2xH + 2xe', 'Mo2C-7*O-2*OH'),
                'Mo2C*O6*OH3': ('Mo2C-9*O + 3xH + 3xe', 'Mo2C-6*O-3*OH'),
                'Mo2C*O5*OH4': ('Mo2C-9*O + 4xH + 4xe', 'Mo2C-5*O-4*OH'),
                'Mo2C*O4*OH5': ('Mo2C-9*O + 5xH + 5xe', 'Mo2C-4*O-5*OH'),
                'Mo2C*O3*OH6': ('Mo2C-9*O + 6xH + 6xe', 'Mo2C-3*O-6*OH'),
                'Mo2C*O2*OH7': ('Mo2C-9*O + 7xH + 7xe', 'Mo2C-2*O-7*OH'),
                'Mo2C*O1*OH8': ('Mo2C-9*O + 8xH + 8xe', 'Mo2C-1*O-8*OH'),
                'Mo2C*OH9':    ('Mo2C-9*O + 9xH + 9xe', 'Mo2C-9*OH'),
            },
            {
                'Mo2C-9*O':     0,
                'Mo2C-8*O-1*OH':  -0.35,
                'Mo2C-7*O-2*OH': -0.27,
                'Mo2C-6*O-3*OH':  0.11,
                'Mo2C-5*O-4*OH':  0.29,
                'Mo2C-4*O-5*OH':  1.06,
                'Mo2C-3*O-6*OH':  1.83,
                'Mo2C-2*O-7*OH':  2.60,
                'Mo2C-1*O-8*OH':  3.54,
                'Mo2C-9*OH':     4.55,
                'H':0                
            },
            {
                'Mo2C-9*O':     0,
                'Mo2C-8*O-1*OH':  0,
                'Mo2C-7*O-2*OH': 0,
                'Mo2C-6*O-3*OH': 0,
                'Mo2C-5*O-4*OH': 0,
                'Mo2C-4*O-5*OH': 0,
                'Mo2C-3*O-6*OH': 0,
                'Mo2C-2*O-7*OH': 0,
                'Mo2C-1*O-8*OH': 0,
                'Mo2C-9*OH':    0,
                'H':0
            }
        ),
        'Fe@4N-Gr':( # https://pubs.acs.org/doi/10.1021/acscatal.3c04801
            {
                '*':       ('*', '*'),
                '*H':      ('* + 1xH + 1xe', '*H'),
                '*O':      ('* + H2O(l)', '*O + 2xH + 2xe'),
                '*OH':     ('* + H2O(l)', '*OH + 1xH + 1xe'),
                '*OOH':    ('* + 2xH2O(l)', '*OOH + 3xH + 3xe'),
                'Fe(s)':   ('*', 'Fe(s)'),
                'Fe+2':    ('*', 'Fe(2z) + 2xe'),
                'Fe+3':    ('*', 'Fe(3z) + 3xe'),
                'Fe(OH)2': ('* + 2xH2O(l)', 'Fe(OH2) + 2xH + 2xe'),
                'Fe(OH)3': ('* + 3xH2O(l)', 'Fe(OH3) + 3xH + 3xe'),
                'FeO4-2':  ('* + 4xH2O(l)', 'Fe(O42f) + 8xH + 6xe'),
            },
            {
                '*': 0,
                '*H': 0.37,
                '*O': 1.49,
                '*OH': 0.67,
                '*OOH': 3.74,
                'H': 0,
                'H2O': 0,
                'Fe': --7.4 - 4.28, # -Eb - E_coh               
            },
            {
                '*': 0,
                '*H': 0,
                '*O': 0,
                '*OH': 0,
                '*OOH': 0,
                'H': 0,
                'H2O(l)': 0,
                'Fe(s)': 0,
                'Fe(2z)': 2 * -0.45,
                'Fe(3z)': 3 * -0.04,
                'Fe(OH2)': 2 * -0.05,
                'Fe(OH3)': 3 * 0.06,
                'Fe(O42f)': 6 * 1.08,                
            }
        ),
        'Cr@4N-Gr':(
            {
                '*':       ('*', '*'),
                '*H':      ('* + 1xH + 1xe', '*H'),
                '*O':      ('* + H2O(l)', '*O + 2xH + 2xe'),
                '*OH':     ('* + H2O(l)', '*OH + 1xH + 1xe'),
                '*OOH':    ('* + 2xH2O(l)', '*OOH + 3xH + 3xe'),
                'Cr(s)':   ('*', 'Cr(s)'),
                'Cr+2':    ('*', 'Cr(2z) + 2xe'),
                'Cr+3':    ('*', 'Cr(3z) + 3xe'),
                'Cr(OH)3': ('* + 3xH2O(l)', 'Cr(OH3) + 3xH + 3xe'),
                'HCrO4-':  ('* + 4xH2O(l)', 'Cr(HO4f) + 7xH + 6xe'),
            },
            {
                '*': 0,
                '*H': 0.32,
                '*O': -0.08,
                '*OH': -0.04,
                '*OOH': 3.28,
                'H': 0,
                'H2O': 0,
                'Cr': --7.2 -4.09,                
            },
            {
                '*': 0,
                '*H': 0,
                '*O': 0,
                '*OH': 0,
                '*OOH': 0,
                'H': 0,
                'H2O(l)': 0,
                'Cr(s)': 0,
                'Cr(2z)': 2 * -0.91,
                'Cr(3z)': 3 * -0.74,
                'Cr(OH3)': 3 * -0.64,
                'Cr(HO4f)': 6 * 0.31,
            }
        ),
        'Mn@4N-Gr':(
            {
                '*':       ('*', '*'),
                '*H':      ('* + 1xH + 1xe', '*H'),
                '*O':      ('* + H2O(l)', '*O + 2xH + 2xe'),
                '*OH':     ('* + H2O(l)', '*OH + 1xH + 1xe'),
                '*OOH':    ('* + 2xH2O(l)', '*OOH + 3xH + 3xe'),
                'Mn(s)':   ('*', 'Mn(s)'),
                'Mn+2':    ('*', 'Mn(2z) + 2xe'),
                'Mn+3':    ('*', 'Mn(3z) + 3xe'),
                'Mn(OH)2': ('* + 2xH2O(l)', 'Mn(OH2) + 2xH + 2xe'),
                'Mn(OH)3': ('* + 3xH2O(l)', 'Mn(OH3) + 3xH + 3xe'),
                'Mn(OH)4': ('* + 4xH2O(l)', 'Mn(OH4) + 4xH + 4xe'),
                'MnO4-2':  ('* + 4xH2O(l)', 'Mn(HO42f) + 8xH + 6xe'), 
            },
            {
                '*': 0,
                '*H': 0.53,
                '*O': 1.05,
                '*OH': 0.63,
                '*OOH': 3.83,
                'H': 0,
                'H2O': 0,
                'Mn': --6.8 -2.92,                
            },
            {
                '*': 0,
                '*H': 0,
                '*O': 0,
                '*OH': 0,
                '*OOH': 0,
                'H': 0,
                'H2O(l)': 0,
                'Mn(s)': 0,
                'Mn(2z)': 2 * -1.19,
                'Mn(3z)': 3 * -0.04,
                'Mn(OH2)': 2 * -0.73,
                'Mn(OH3)': 3 * -0.16,
                'Mn(OH4)': 4 * 0.21,
                'Mn(HO42f)': 6 * 0.74, 
            }
        ),
        'Co@4N-Gr':(
            {
                '*':       ('*', '*'),
                '*H':      ('* + 1xH + 1xe', '*H'),
                '*O':      ('* + H2O(l)', '*O + 2xH + 2xe'),
                '*OH':     ('* + H2O(l)', '*OH + 1xH + 1xe'),
                '*OOH':    ('* + 2xH2O(l)', '*OOH + 3xH + 3xe'),
                'Co(s)':   ('*', 'Co(s)'),
                'Co+2':    ('*', 'Co(2z) + 2xe'),
                'Co+3':    ('*', 'Co(3z) + 3xe'),
                'Co(OH)2': ('* + 2xH2O(l)', 'Co(OH2) + 2xH + 2xe'),
                'Co(OH)3': ('* + 3xH2O(l)', 'Co(OH3) + 3xH + 3xe'),     
            },
            {
                '*': 0,
                '*H': 0.13,
                '*O': 2.64,
                '*OH': 1.08,
                '*OOH': 4.33,
                'H': 0,
                'H2O': 0,
                'Co': --7.8 -4.43,
            },
            {
                '*': 0,
                '*H': 0,
                '*O': 0,
                '*OH': 0,
                '*OOH': 0,
                'H': 0,
                'H2O(l)': 0,
                'Co(s)': 0,
                'Co(2z)': 2 * -0.28,
                'Co(3z)': 3 * -0.74,
                'Co(OH2)': 2 * 0.10,
                'Co(OH3)': 3 * 0.31,                
            }
        )
    }
    
    return dataset[data_name]
    

kB = 8.617330337217213e-05
CONST = kB * np.log(10)  # Nernst constant
water_refer = -2.4583 # Default chemical potentials for water
U_STD_AGCL = 0.222  # Standard redox potential of AgCl electrode
U_STD_SCE = 0.244   # Standard redox potential of SCE electrode


class Pourbaix():
    def __init__(self, reaction, energies=None, Gcors=None, reference='SHE', T=298.15, conc=1.0e-6, column_names=['sys', 'energy', 'Gcor']):
        '''
        Parameters:
            - reaction (str or dict): If str, search from dataset_pourbais in pourbais.py
              key for reaction is the material (phase) name which only use for display. Write the sign of the charged number in front, like 'ZnO2+2'
              value is a (2, ) Tuple including reactant and product, e.g. {'ZnO2-2': ('Zn + 2xH2O(l)', 'ZnO2ff + 4xH + 2xe')}
            - energies (dict or str): Energies for items in reaction, or a name of .xlsx or .csv file to load data.
            - Gcors (dict or str): Correction energies for items in reaction, or a name of .xlsx or .csv file to load data.
            - reference (str): The reference electrode. Default = 'SHE'
              energy and Gcor for an item will first search from the input dicts, then default molecule from dataset.py
            - T (float): Temperature in Kelvin. Default: 298.15 K
            - conc (float): Concentration of the ionic species. Default: 1e-6 mol/L.
            - column_names ((3, ) list): Column names of material, energy, and corrected energy in DataFrame. Default = ['sys', 'energy', 'Gcor']
        '''
        self.T = T
        self.conc = conc
        self.reference = reference
        
        self.reaction = dataset_pourbais[reaction] if type(reaction) is str else reaction

        self.dataset_energy_cal, self.dataset_Gcor_cal = self._get_energy_Gcor(energies, Gcors, column_names)
        self._generate_reac_matrix()

    def _get_energy_Gcor(self, energies, Gcors, column_names):
        '''
        Get energy and correction energy dict from dataframe (load from a .xlsx or .csv file)
        '''
        # read energy and cor data from dataframe
        if type(energies) is str:
            if   energies[-5:] == '.xlsx':
                dfe = pd.read_excel(energies)
            elif energies[-4:] == '.csv':
                dfe = pd.read_csv(energies)
            dataset_energy_cal = {}
            for (sys, e) in zip(dfe[column_names[0]], dfe[column_names[1]]):
                dataset_energy_cal[str(sys)] = float(e)
        else:
            dataset_energy_cal = energies
            
        if type(Gcors) is str:
            if   Gcors[-5:] == '.xlsx':
                dfg = pd.read_excel(Gcors)
            elif Gcors[-4:] == '.csv':
                dfg = pd.read_csv(Gcors)
            dataset_Gcor_cal = {}
            for (sys, g) in zip(dfg[column_names[0]], dfg[column_names[2]]):
                dataset_Gcor_cal[str(sys)] = float(g)
        else:
            dataset_Gcor_cal = Gcors

        return dataset_energy_cal, dataset_Gcor_cal
        
    def _generate_reac_matrix(self, phases=None, verbose=False):
        '''
        Generate a matrix for the delta G calculation

        Parameters:
            - phase (list or None): Phases that are considered. Default = None
              None will choose all phases in reaction.keys() and update self.matrix
            - verbose (bool): Print detailed information. Default = False
              Use an not  None phases and verbose=True to check calculation process
        '''
        if phases is None:
            phases = self.reaction.keys()
            update_matrix = True
            self.matrix = np.zeros((0, 3))
        else:
            update_matrix = False

        for phase in phases:
            print(f'------ Phase {phase}') if verbose else None
            a, b, c = 0, 0, 0
            for i, reaction in enumerate(self.reaction[phase]):
                print(f'---{ {0: "reactat", 1: "product"}[i]}: {reaction}') if verbose else None
                sign = -1 if i == 0 else 1
                for pair in reaction.split('+'):
                    coefficient, item = sep_coeff_varia(pair)
                    coefficient = coefficient * sign
                    if   item == 'e':
                        c += -1 * coefficient
                        print(f'item: {item:10s} - coef: {int(coefficient):3d} - U_term: {c:6.3f}') if verbose else None
                    elif item == 'H':
                        b += -1 * coefficient * CONST * self.T
                        G = self._get_G(item)
                        a += coefficient * G
                        print(f'item: {item:10s} - coef: {int(coefficient):3d} - energy: {G:8.3f} - G_term: {a:6.3f} - PH_term: {b:6.3f}') if verbose else None
                    else:
                        G = self._get_G(item)
                        a += coefficient * G
                        print(f'item: {item:10s} - coef: {int(coefficient):3d} - energy: {G:8.3f} - G_term: {a:6.3f}') if verbose else None

            gibbs_corr, pH_corr = self._get_ref_correction(-1 * c)
            print(f'refer: {self.reference:6s}, n_e: {-1 * int(c):3d}, Gcor: {gibbs_corr:6.3f}, pH_cor: {pH_corr:6.3f}') if verbose else None
            vector = np.array([a + gibbs_corr, b + pH_corr, c])
            print(f'--- final vector: {vector[0]:12.6f}, {vector[1]:12.6f}, {vector[2]:12.6f}') if verbose else None
                        
            self.matrix= np.vstack((self.matrix, vector)) if update_matrix else None

    def _get_ref_correction(self, n_e):
        '''
        Correct the constant and pH contributions to the reaction free energy based on the reference electrode of choice and the temperature
        '''
        gibbs_corr = 0.0
        pH_corr = 0.0
        if self.reference in ['RHE', 'Pt']:
            pH_corr += n_e * CONST * self.T
            if self.reference == 'Pt' and n_e < 0:
                gibbs_corr += n_e * 0.5 * water_refer
        if self.reference == 'AgCl':
            gibbs_corr -= n_e * U_STD_AGCL
        if self.reference == 'SCE':
            gibbs_corr -= n_e * U_STD_SCE

        return gibbs_corr, pH_corr
            
    def get_pourbaix_energy(self, U, PH, verbose=False):
        '''
        Find the most stable phase at U and PH

        Parameters:
            - U (float).
            - PH (float).
            - verbose (bool): Print energy for each phase. Default = False
        '''
        es = np.array([1, PH, U]) @ self.matrix.T
        phase = list(self.reaction.keys())[np.argmin(es)]
        print(f'{phase} is the most stable.')
        if verbose:
            for i, e in enumerate(es):
                print(f'energy of {list(self.reaction.keys())[i]} is {e:.3f} eV.')

    def _get_G(self, item):
        '''
        Calculate the free energy of an item
        '''
        # find energy
        if   item.split('(')[0] in self.dataset_energy_cal.keys():
            energy = self.dataset_energy_cal[item.split('(')[0]]
        elif item.split('(')[0] in dataset_energy.keys():
            energy = dataset_energy[item.split('(')[0]]
        else:
            raise KeyError(f'Energy value of {item.split("(")[0]} not found!')
            
        # fing correction energy
        if   item in self.dataset_Gcor_cal.keys():
            Gcor = self.dataset_Gcor_cal[item]
        elif item in dataset_Gcor.keys():
            Gcor = dataset_Gcor[item]
        else:
            raise KeyError(f'Gcor value for {item} not found!')
            
        # solvent correct by \mu = \mu_0 + RTlnC
        if '(' in item:
            if 'aq' in item.split('(')[1]:
                Gsol = CONST * self.T * np.log10(self.conc)
            else:
                Gsol = 0
        else:
            Gsol = 0
            
        return energy + Gcor + Gsol

    def compute_phase(self, PH_range=[0, 14], U_range=[-2, 2], resolution=1000):
        '''
        Calculate delta G and confirm the stable phase

        Parameters:
            - PH_range ((2, ) list). Default = [0, 14]
            - U_range ((2, ) list). Default = [-2, 2]
            - resolution (int): Grid density. Default = 1000
        '''
        # generate grid
        x = np.linspace(PH_range[0], PH_range[1], resolution)
        y = np.linspace(U_range[0], U_range[1], resolution)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # initial min value and phase array
        min_e = np.full_like(x_grid, np.inf)
        phase = np.zeros_like(x_grid, dtype=int)
        
        # Find the minimum formation energy and phase for each point 
        for i, (A, B, C) in enumerate(self.matrix):
            formation_energy = A + B * x_grid + C * y_grid
            mask = formation_energy < min_e
            min_e[mask] = formation_energy[mask]
            phase[mask] = i

        self.x_grid = x_grid
        self.y_grid = y_grid
        self.min_e = min_e
        self.phase = phase
        
    def plot(self, typ='phase', colors='viridis', legend=True, colorbar=True, text=True, redox_lines=True, convert_label=True):
        '''
        Show the Pourbaix diagrams

        Parameters:
            - typ (str): Plot type, choose from 'phase' (phase diagram) or 'min_e' (heatmap of delta G)
            - colors (str or (num_reaction, ) list): Use str to choose a colormap from matplotlib or give in a (num_reaction, ) list. Default = 'viridis'
            - legend (bool): Add the legend if plotting a  phase diagram. Default = True
            - colorbar (bool): Add a colorbar if plotting a heatmap. Default = True
            - text (bool): Add phase name at the center of the phase zone. Default = True
            - redox_lines (bool): Add water redox potentials. Default = True
            - convert_label (bool): Use phase_name_convert to convert label. Default = True
        Return:
            - The figure
        '''
        # set figure
        figure = plt.figure(figsize=(6, 6))
        plt.xlabel('PH')
        plt.ylabel('U (V)')

        # initial parameters
        n = len(self.matrix)
        extent = [self.x_grid.min(), self.x_grid.max(), self.y_grid.min(), self.y_grid.max()]
        aspect = (self.x_grid.max() - self.x_grid.min())/(self.y_grid.max()-self.y_grid.min())
        im = self.min_e if typ == 'min_e' else self.phase
        if   typ == 'min_e':
            cmap = plt.colormaps[colors]
        elif typ == 'phase' and type(colors) is str:
            cmap = ListedColormap(plt.colormaps[colors](np.linspace(0, 1, n)))
        elif typ == 'phase' and type(colors) is list:
            cmap = ListedColormap(colors)

        # plot 
        plt.contour(self.x_grid, self.y_grid, self.phase, levels=np.arange(0.5, n), colors='black', linewidths=1.5, zorder=2)
        plt.imshow(im, extent=extent, origin='lower', aspect=aspect, cmap=cmap, alpha=0.8, zorder=1)

        # labels and annotations
        if typ == 'min_e' and colorbar:
            plt.colorbar(label='$\Delta$ G (eV)')
            
        if typ == 'phase' and legend:
            legend_elements = [
                Patch(facecolor=cmap.colors[i], label=phase_name_convert(list(self.reaction.keys())[i], convert_label))
                for i in range(n)
            ]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.26, 1))
            
        if text:
            centers = {}
            for label in np.unique(self.phase):
                y_center, x_center = center_of_mass(self.phase == label)
                centers[label] = (x_center, y_center)
            for label, (x, y) in centers.items():
                xx = x/self.x_grid.shape[0] * (self.x_grid.max() - self.x_grid.min()) + self.x_grid.min()
                yy = y/self.y_grid.shape[0] * (self.y_grid.max() - self.y_grid.min()) + self.y_grid.min()
                plt.text(xx, yy, phase_name_convert(list(self.reaction.keys())[int(label)], convert_label), ha='center', va='center', color='black')

        if redox_lines:
            plt.xlim(self.x_grid.min(), self.x_grid.max())
            plt.ylim(self.y_grid.min(), self.y_grid.max())
            pH = np.array([self.x_grid.min(), self.x_grid.max()])
            const = -0.5 * water_refer
            corr = {
                'SHE': 0,
                'RHE': 0,
                'Pt': 0,
                'AgCl': -U_STD_AGCL,
                'SCE': -U_STD_SCE,
            }
            kwargs = {
                'c': 'k',
                'ls': '--',
                'zorder': 3
            }
            if self.reference in ['SHE', 'AgCl', 'SCE']:
                slope = -1 * CONST * self.T
                plt.plot(pH, slope * pH + corr[self.reference], **kwargs)
                plt.plot(pH, slope * pH + const + corr[self.reference], **kwargs)
            elif self.reference in ['Pt', 'RHE']:
                plt.axhline(0 + corr[self.reference], **kwargs)
                plt.axhline(const + corr[self.reference], **kwargs)
            else:
                raise ValueError('The specified reference electrode doesnt exist')
                
        plt.close()

        return figure