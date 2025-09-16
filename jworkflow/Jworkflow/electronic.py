from pymatgen.electronic_structure.core import Spin, OrbitalType, Orbital
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.io.vasp.outputs import Locpot, Chgcar, Procar
from pymatgen.io.lobster import Doscar
from pymatgen.io.vasp import Vasprun
from pymatgen.core import Element
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_locpot_z_distrib(Locpot_file):
    '''
    Function used to obtain the electrostatic potential in the z direction to calculate the work function

    Parameter:
        - Locpot_file: Path to the LOCPOT file / str, path
    Return:
        - (n) array represent the electrostatic potential in z direction
    '''
    locpot = Locpot.from_file(Locpot_file)
    return locpot.get_average_along_axis(2)


class CHG_Process():
    def __init__(self):
        self.chgcar = None

    def read_chgcar(self, chgcar):
        '''
        Read in chgcar file

        Parameter:
            - chgcar (path)
        '''
        self.chgcar = Chgcar.from_file(chgcar)

    def get_chgcar_distrib(self, axis):
        '''
        Function to obtain the distribtion of deformation charge density along x, y or z axis
    
        Parameter:
            - axis (int): The axis for projecting / int, 0, 1 or 2, represent x, y and z axis
        Return:
            - A dict include axis grid and charge density difference
        '''
        return {'grid': self.chgcar.get_axis_grid(axis), 'chg_diff': self.chgcar.get_average_along_axis(axis)}
        
    def smooth_chgcar(self, smooth_chgcar, xt=[0, 1], yt=[0, 1], zt=[0, 1]):
        '''
        Function to smooth the CHGCAR
    
        Parameters:
            - smooth_chgcar (path): Path to the smoothed CHGCAR /str, path
            - xt (list): X-axis threshold in percentage, charge out of this threshold will be set to 0 / (2) list, default [0, 1]
            - yt (list): X-axis threshold in percentage / (2) list, default [0, 1]
            - zt (list): Z-axis threshold in percentage / (2) list, default [0, 1]
        Accomplish:
            - Write a smoothed CHGCAR at smooth_chgcar
        '''
        gridx = self.chgcar.data['total'].shape[0]
        gridy = self.chgcar.data['total'].shape[1]
        gridz = self.chgcar.data['total'].shape[2]
        for x in range(gridx):
            for y in range(gridy):
                for z in range(gridz):
                    if (x <= gridx * xt[0] or x >= gridx * xt[1]) or (y <= gridy * yt[0] or y >= gridy * yt[1]) or (z <= gridz * zt[0] or z >= gridz * zt[1]):
                        self.chgcar.data['total'][x][y][z] = 0.
        self.chgcar.write_file(smooth_chgcar)
        
    def get_elec_in_sphere(self, target_atom, radius):
        '''
        The value of the charge within a series of radii of an atom

        Parameters:
            - target_atom (int)
            - radius (list)
        Return:
            - a list of charge number
        '''
        xt, yt, zt = self.chgcar.structure.cart_coords[target_atom]
        # get dV
        dx = self.chgcar.xpoints[1] - self.chgcar.xpoints[0]
        dy = self.chgcar.ypoints[1] - self.chgcar.ypoints[0]
        dz = self.chgcar.zpoints[1] - self.chgcar.zpoints[0]
        dV = dx * dy * dz
        # get ele in sphere
        elecs = []
        for r in radius:
            llattice = self.chgcar.structure.lattice
            x, y, z = np.meshgrid(self.chgcar.xpoints*llattice.a, self.chgcar.ypoints*llattice.b, self.chgcar.zpoints*llattice.c, indexing='ij')
            distances = np.sqrt((x - xt)**2 + (y - yt)**2 + (z - zt)**2)
            mask = distances <= r
            eles.append(np.sum(self.chgcar.data['total'][mask]) * dV)
        return elecs

def find_orbital_component(procar, target_atom, target_orbital, proj_threshold=0.0):
    '''
    Search for the bands where the projection of an atom's orbit is greater than a certain value

    Parameters:
        - procar (path to file): Procar file
        - target_atom (int)
        - target_orbital (str): orbital type list s, px, py, pz, dxy, dxz, dyz, dx2, dz2 ..
        - proj_threshold (float). Default = 0.0
    Return:
        - a list of list (spin, kpoint, band, atom, orbital, proj_weight)
    '''
    procar = Procar(procar)
    target_orbital = procar.orbitals.index(target_orbital)
    results = []
    for spin in [Spin.up, Spin.down]:
        if spin in procar.data.keys():
            for kpoint in range(procar.nkpoints):
                for band in range(procar.nbands):
                    proj_weight = procar.data[spin][kpoint][band][target_atom][target_orbital]
                    if proj_weight > proj_threshold:
                        results.append([spin, kpoint, band, target_atom, target_orbital, proj_weight])
    return results
    
name_to_value_orbital = {
    's': 0,
    'py': 1,
    'pz': 2,
    'px': 3,
    'dxy': 4,
    'dyz': 5,
    'dz2': 6,
    'dxz': 7,
    'dx2': 8,
    'f_3': 9,
    'f_2': 10,
    'f_1': 11,
    'f_0': 12,
    'f1': 13,
    'f2': 14,
    'f3': 15,
}
name_to_value_orbitaltype = {
    's': 0,
    'p': 1,
    'd': 2,
    'f': 3,
}

class DOS_Process():
    '''
    Class to read in vasprun.xml and calculate DOS related properties

    Available Functions:
        - read_xml: Read vasprun.xml file
        - D_band_center: Calculate d band center of specified atoms
        - D_band_width: Calculate the d-band width, d-band-center need to calculate at first
        - upper_D_band_edge: Calculate the upper d-band edge
        - fermi_softness: Calculate the fermi softness
        - D_band_filling: Calculate the d band filling
        - get_E_DOS: Get E and DOS data for the specified atom
        - all_band_properities: Use pymatgen to calculate all band properities
    Internal Functions:
        - integral: Perform Intergration
    '''

    def __init__(self):
        '''
        Attributes:
            - xml: Read in vasprun.xml file / pymatgen.io.vasp.Vasprun
            - ispin: Whether ispin is on / bool, default False
            - orbital_ds: d oribtals / list, default [Orbital.dxy,Orbital.dyz,Orbital.dz2,Orbital.dxz,Orbital.dx2]
            - orbital_ps: p orbitals / list, default [Orbital.px,Orbital.py,Orbital.pz]
            - orbital_ss: s orbitals / list, default [Orbital.s]
            - DBC: The calculated d-band-center / list
        '''
        self.xml = None
        self.ispin = False
        self.orbital_ds = [Orbital.dxy, Orbital.dyz, Orbital.dz2, Orbital.dxz, Orbital.dx2]
        self.orbital_ps = [Orbital.px, Orbital.py, Orbital.pz]
        self.orbital_ss = [Orbital.s]
        self.DBC = None

    def read_xml(self, xml_file):
        '''
        Function to read vasprun.xml file

        Parameter:
            - xml_file: Path to the vasprun.xml file / str, path
        '''
        self.xml = Vasprun(xml_file, parse_potcar_file=False)
        self.ispin = self.xml.is_spin

    def D_band_center(self, atom_index, erange=None, max_inter=10, relative_to_fermi=True, orbitals=None):
        '''
        Function to calculate d band center of specified atoms

        Parameters
            - atom_index: Index of the atom / int
            - erange: Energy range of the DOS / (2)D-array, default [EMIN-0.6,EMAX+0.6]
            - max_inter: The maximum integral value at which the integration will stop and the result will be calculated / float, default 10
            - relative_to_fermi: Whether the energy is relative to Fermi level / bool, default True
            - orbitals: List of orbitals to cumulative DOS / (n)D-array, pymatgen.Orbital, default all d orbitals
        Return:
            - The d-band-center
        '''
        if orbitals == None:
            orbitals = self.orbital_ds
        if erange == None:
            E, DOS = self.get_E_DOS(1, self.orbital_ds, relative_to_fermi)
            erange = [min(E) - 0.6, max(E) + 0.6]

        condition = [erange[0], erange[1], max_inter, 0]
        E, DOS = self.get_E_DOS(atom_index, orbitals, relative_to_fermi)
        DBC_up = self.integral('DOS*E', E, DOS[0], condition) / self.integral('DOS', E, DOS[0], condition)
        if self.ispin:
            DBC_down = self.integral('DOS*E', E, DOS[1], condition) / self.integral('DOS', E, DOS[1], condition)
        if not self.ispin:
            self.DBC = [DBC_up]
            return round(DBC_up, 2)
        elif self.ispin:
            self.DBC = [DBC_up, DBC_down]
            return [round(DBC_up, 2), round(DBC_down, 2)]

    def D_band_width(self, atom_index, erange=None, max_inter=10, relative_to_fermi=True, orbitals=None):
        '''
        Function to calculate the d-band width, d-band-center need to calculate at first

        Parameters
            - atom_index: Index of the atom, start from index 1 not 0 / int
            - erange: Energy range of the DOS / (2)D-array, default [EMIN-0.6,EMAX+0.6]
            - max_inter: The maximum integral value at which the integration will stop and the result will be calculated / float, default 10
            - relative_to_fermi: Whether the energy is relative to Fermi level / bool, default True
            - orbitals: List of orbitals to cumulative DOS / (n)D-array, pymatgen.Orbital, default all d orbitals
        Return:
            - The d-band width
        '''
        if orbitals == None:
            orbitals = self.orbital_ds
        if erange == None:
            E, DOS = self.get_E_DOS(1, self.orbital_ds, relative_to_fermi)
            erange = [min(E) - 0.6, max(E) + 0.6]

        condition = [erange[0], erange[1], max_inter, 0]
        E, DOS = self.get_E_DOS(atom_index, orbitals, relative_to_fermi)
        DW_up = self.integral('DOS*(E-DBC)**2', E, DOS[0], condition) / self.integral('DOS', E, DOS[0], condition)
        DW_up = DW_up ** (1 / 2)
        if self.ispin:
            condition = [erange[0], erange[1], max_inter, 1]
            DW_down = self.integral('DOS*(E-DBC)**2', E, DOS[1], condition) / self.integral('DOS', E, DOS[1], condition)
            DW_down = DW_down ** (1 / 2)
        if not self.ispin:
            return round(DW_up, 2)
        elif self.ispin:
            return [round(DW_up, 2), round(DW_down, 2)]

    def upper_D_band_edge(self, atom_index, orbitals=None):
        '''
        Function to calculate the upper d-band edge

        Parameters:
            - atom_index: Index of the atom / int
            - orbitals: List of orbitals to cumulative DOS / (n)D-array, pymatgen.Orbital, default all d orbitals
        Return:
            - The upper d-band edge
        '''
        if orbitals == None:
            orbitals = self.orbital_ds

        E, DOS = self.get_E_DOS(atom_index, orbitals, True)
        UDE_up = E[np.argmax(np.imag(hilbert(abs(DOS[0]))))]
        if self.ispin:
            UDE_down = E[np.argmax(np.imag(hilbert(abs(DOS[1]))))]
        if not self.ispin:
            return round(UDE_up, 2)
        elif self.ispin:
            return [round(UDE_up, 2), round(UDE_down, 2)]

    def fermi_softness(self, atom_index, kT=0.3, orbitals=None):
        '''
        Function to calculate the fermi softness

        Parameters:
            - atom_index: Index of the atom / int
            - kT: The kT value in Fermi-Dirac function / float, default 0.3
            - orbitals: List of orbitals to cumulative DOS / (n)D-array, pymatgen.Orbital, default all d orbitals
        Return:
            - The fermi softness
        '''
        if orbitals == None:
            orbitals = self.orbital_ds

        E, DOS = self.get_E_DOS(atom_index, orbitals, True)
        condition = [min(E) - 0.6, max(E) + 0.6, 10, kT]
        FS_up = self.integral('DOS*Fermi-Dirac*-1', E, DOS[0], condition)
        if self.ispin:
            FS_down = self.integral('DOS*Fermi-Dirac*-1', E, DOS[1], condition)
        if not self.ispin:
            return round(FS_up, 2)
        elif self.ispin:
            return [round(FS_up, 2), round(FS_down, 2)]

    def D_band_filling(self, atom_index, erange=None, orbitals=None):
        '''
        Function to calculate the d band filling

        Parameters
            - atom_index: Index of the atom / int
            - erange: Energy range of the DOS / (2)D-array, default [EMIN-0.6,EMAX+0.6]
            - orbitals: List of orbitals to cumulative DOS / (n)D-array, pymatgen.Orbital, default all d orbitals
        Return:
            - The d band filling
        '''
        if orbitals == None:
            orbitals = self.orbital_ds
        if erange == None:
            E, DOS = self.get_E_DOS(1, self.orbital_ds, True)
            erange = [min(E) - 0.6, max(E) + 0.6]

        E, DOS = self.get_E_DOS(atom_index, orbitals, True)
        condition1 = [erange[0], 0, 10, 0]
        condition2 = [erange[0], erange[1], 10, 0]
        DF = self.integral('DOS', E, DOS[0], condition1) / self.integral('DOS', E, DOS[0], condition2)
        if self.ispin:
            DF = (self.integral('DOS', E, DOS[0], condition1) + self.integral('DOS', E, DOS[1], condition1)) \
                 / (self.integral('DOS', E, DOS[0], condition2) + self.integral('DOS', E, DOS[1], condition2))
        return round(DF, 2)

    def all_band_properities(self, atom_list, orbitals=OrbitalType.d, erange=None, spin=None):
        '''
        Function to use pymatgen to calculate all band properities

        Parameters:
            - atom_index: List of indexes of the atom / (n)D-list
            - orbitals: List of orbitals to cumulative properities / (n)D-array, pymatgen.Orbital, default OrbitalType.d
            - erange: Energy range of the Band / (2)D-array, default None
            - spin: Spin channel to use. If None, both spin channels will be combined. / Spin or None, default None
        Return:
            - List of band properities
        '''
        band_properities = {}
        structure = self.xml.final_structure
        complete_dos = self.xml.complete_dos
        sites = [structure[i] for i in atom_list]
        band_properities['band center'] = complete_dos.get_band_center(sites=sites, band=orbitals, erange=erange, spin=spin)
        band_properities['band width'] = complete_dos.get_band_width(sites=sites, band=orbitals, erange=erange, spin=spin)
        band_properities['band kurtosis'] = complete_dos.get_band_kurtosis(sites=sites, band=orbitals, erange=erange, spin=spin)
        band_properities['band skewness'] = complete_dos.get_band_skewness(sites=sites, band=orbitals, erange=erange, spin=spin)
        band_properities['band filling'] = complete_dos.get_band_filling(sites=sites, band=orbitals, spin=spin)
        band_properities['upper band edge'] = complete_dos.get_upper_band_edge(sites=sites, band=orbitals, erange=erange, spin=spin)
        return band_properities

    def get_E_DOS(self, atom_index, orbitals=None, relative_to_fermi=True):
        '''
        Function to get E and DOS data for the specified atom

        Parameters:
            - atom_index: Index of the atom / int
            - orbitals: List of orbitals to cumulative DOS / (n)D-array, pymatgen.Orbital, default all d orbitals
            - relative_to_fermi: Whether the energy is relative to Fermi level / bool, default True
        Return
            - E: (n)D-array, energy
            - DOS: (2,n)D-array, DOS of up and down spin respectively
        '''
        if orbitals == None:
            orbitals = [Orbital.dxy, Orbital.dyz, Orbital.dz2, Orbital.dxz, Orbital.dx2]
        if relative_to_fermi:
            E = self.xml.tdos.energies - self.xml.efermi
        else:
            E = self.xml.tdos.energies
        DOS_up = np.zeros(len(E))
        DOS_down = np.zeros(len(E))
        for orbital in orbitals:
            DOS_up += self.xml.pdos[atom_index][orbital][Spin.up]
            if self.ispin:
                DOS_down += self.xml.pdos[atom_index][orbital][Spin.down]
        DOS = [DOS_up, DOS_down]
        return E, DOS

    def integral(self, formula, E, DOS, condition):
        '''
        Functin to perform Intergration

        Paramters
            - formula: The formula for integrating objects / str, 'DOS', 'DOS*E', 'DOS*(E-DBC)**2', 'DOS*Fermi-Dirac*-1'
            - E: Array of energy / (n)D-array
            - DOS: Array of density of states / (n)D-array
            - condition : List of extra information / (4)D-array
                       4 values are EMIN, EMAX, MaxIntegralElectronic and ExtraValue (atom_index in DOS*(E-DBC)**2, kT value in DOS*Fermi-*-Dirac1)
        Return:
            - The integral result
        '''
        EMIN = condition[0]
        EMAX = condition[1]
        MaxIntegralElectronic = condition[2]
        ExtraValue = condition[3]

        delta_E = E[1] - E[0]
        IntegralElectronic = 0
        result = 0
        for i in range(len(E)):
            IntegralElectronic += delta_E * abs(DOS[i])
            if E[i] >= EMIN and E[i] <= EMAX and IntegralElectronic <= MaxIntegralElectronic:
                if formula == 'DOS':
                    result += delta_E * abs(DOS[i])
                elif formula == 'DOS*E':
                    result += delta_E * abs(DOS[i]) * E[i]
                elif formula == 'DOS*(E-DBC)**2':
                    result += delta_E * abs(DOS[i]) * (E[i] - self.DBC[ExtraValue]) ** 2
                elif formula == 'DOS*Fermi-Dirac*-1':
                    result += delta_E * abs(DOS[i]) * (np.exp(E[i] / ExtraValue) + 1) ** (-2) * np.exp(E[i] / ExtraValue) * 1 / ExtraValue
        return result
        
    def get_dos(self, dos_typ, sigma=None):
        '''
        Quickly grab the desired density state density
    
        Parameters:
            - dos_typ: The state density feature you want to grab. If there is a spin, it's automatically divided into up and down / list
              Use the following 7 mode strings to define the required state density information:
              'tot': total dos; 'tot-p': total p dos; 'Pd_Ni-tot': tot dos of Ni and Pd element; 'Ni-d': d dos of Ni element
              '6_7_8_9_10-tot': total dos of site 6 ~ 10; '6-f': f dos of site 6; '0_6-dyz' dyz dos of site 6 and 0
              input including a required dos infor and a label name of it, e.g. [(7_8-d, Ni-d)]
            - sigma: / float or None, default = None
        '''
        complete_dos = self.xml.complete_dos
        structure = self.xml.final_structure
    
        self.dos_name = []
        self.dos_data = {}
        self.dos_data['energies'] = complete_dos.energies - complete_dos.efermi
        
        for (di, dn) in dos_typ:
            if '-' in di:
                obj, ort = di.split('-')
            doses = []
            if   di == 'tot':
                doses.append(complete_dos) 
            elif obj == 'tot':
                doses.append(complete_dos.get_spd_dos()[OrbitalType(name_to_value_orbitaltype[ort])])
            elif obj[0].isalpha() and ort == 'tot':
                for ele in obj.split('_'):
                    doses.append(complete_dos.get_element_dos()[Element(ele)])
            elif obj[0].isalpha() and ort.isalpha():
                for ele in obj.split('_'):
                    doses.append(complete_dos.get_element_spd_dos(Element(ele))[OrbitalType(name_to_value_orbitaltype[ort])])
            elif obj[0].isdigit() and ort == 'tot':
                for site in obj.split('_'):
                    doses.append(complete_dos.get_site_dos(structure[int(site)]))
            elif obj[0].isdigit() and len(ort) == 1:
                for site in obj.split('_'):
                    doses.append(complete_dos.get_site_spd_dos(structure[int(site)])[OrbitalType(name_to_value_orbitaltype[ort])])
            elif obj[0].isdigit() and len(ort) > 1:
                for site in obj.split('_'):
                    doses.append(complete_dos.get_site_orbital_dos(structure[int(site)], Orbital(name_to_value_orbital[ort])))
    
            for i in range(len(doses)):
                doses[i] = doses[i].densities if sigma is None else doses[i].get_smeared_densities(sigma)
            if self.ispin:
                dos_up = [dos[Spin.up] for dos in doses]
                dos_up = np.sum(dos_up, axis=0)
                dos_down = [dos[Spin.down] for dos in doses]
                dos_down = -1 * np.sum(dos_down, axis=0)
                self.dos_data[dn + '-up'] = dos_up
                self.dos_data[dn + '-down'] = dos_down
            else:
                dos = [dos[Spin.up] for dos in doses]
                dos = np.sum(dos, axis=0)
                self.dos_data[dn] = dos
            self.dos_name.append(dn)
    
    def plot_dos(self):
        '''
        Plot DOS. It can only be used after get_dos has been executed.
        '''
        plt.figure(figsize=(12, 6))
        energies = self.dos_data['energies']
    
        ymax = 0
        ymin = 0
        for dos_name in self.dos_name:
            if self.ispin:
                dos = np.concatenate([self.dos_data[dos_name + '-up'], np.flip(self.dos_data[dos_name + '-down'])])
                plt.plot(np.concatenate([energies, np.flip(energies)]), dos, label=dos_name)
            else:
                dos = self.dos_data[dos_name]
                plt.plot(energies, dos, label=dos_name)
            if np.max(dos) > ymax:
                ymax = np.max(dos)
            if np.min(dos) <ymin:
                ymin = np.min(dos)
        plt.plot([np.min(energies), np.max(energies)], [0, 0], color='k', zorder=6)
        y_length = ymax - ymin
        plt.plot([0, 0], [ymin - y_length *0.06, ymax + y_length * 0.06], color='k', ls='--', zorder=-6)
        plt.ylim([ymin - y_length *0.06, ymax + y_length * 0.06])
        plt.legend()
        plt.xlabel('Energy - Fermi level (eV)')
        plt.ylabel('DOS (States/eV)')
        plt.show()
    
    def write_csv(self, csv):
        '''
        Write DOS data to a csv file. It can only be used after get_dos has been executed.
    
        Parameters:
            - csv: the csv file / file path
        '''
        df = pd.DataFrame.from_dict(self.dos_data)
        df.to_csv(csv)
    
    
class Lobster_PProcess():
    '''
    Class to draw data from Lobster outputs by pymatgen

    Available Functions:
        - read_car: Function to read in lobster outputs
        - show_bonds: Function to show the bonds in COXPCAR.lobster file
        - get_coxp: Function to get COXP of Specified label and orbitals
        - get_icohp: Function to get the ICOHP for a specified bond
        - get_band_properities: Function to use pymatgen to calculate all band properities
        - show_spd_typ: Function to show oribtals on a specified site
        - get_dos: Function to get DOS data
    '''

    def __init__(self):
        '''
        Attributes:
            - COXP: The COXP data / pymatgen.electronic_structure.cohp.CompleteCohp
            - DOS: The DOS data / pymatgen.io.lobster.Doscar.completedos
        '''
        self.COXP = None
        self.DOS = None

    def read_car(self, car, structure, typ='coxp'):
        '''
        Function to read in lobster outputs

        Parameters:
            - car: Path to the output file / str, path
            - structure: Path to the structure / str, path
            - typ: Read in file type / str in 'coxp' or 'dos', default 'coxp'
        Cautions:
            - Can only read in COHPCAR.lobster, COOPCAR.lobster and DOSCAR.lobster
        '''
        if typ == 'coxp':
            self.COXP = CompleteCohp.from_file(fmt="LOBSTER", filename=car, structure_file=structure)
            self.COXP.energies = self.COXP.energies - self.COXP.efermi
        elif typ == 'dos':
            self.DOS = Doscar(doscar=car, structure_file=structure).completedos
            self.DOS.efermi = self.COXP.efermi

    def show_bonds(self):
        '''
        Function to show the bonds in COXPCAR.lobster file

        Accomplish:
            - Show each bond`s label and atom index
        Cautions:
            - This atom index count from 1
        '''
        for i, k in enumerate(list(self.COXP.bonds.keys())):
            bond_label = str(self.COXP.structure.index(self.COXP.bonds[k]['sites'][0]) + 1) + str(self.COXP.bonds[k]['sites'][0].species_string) \
                         + '-' + str(self.COXP.structure.index(self.COXP.bonds[k]['sites'][1]) + 1) + str(self.COXP.bonds[k]['sites'][1].species_string)
            print(('%2s %12s') % (str(i + 1), bond_label))

    def get_coxp(self, label, divisor=1, orbitals=None):
        '''
        Function to get COXP of specified label and orbitals

        Parameters:
            - label: Bond indexes in lobster calculation / int, str or list
            - divisor: The divisor, when specifies more than 1 labels / float, default 1
            - orbitals: Select orbitals / list, default None
        Return:
            - Dict of the COXP data
        Cautions:
            - Label corresponding to the return of show_bonds function
            - The return COXP values are another dict in the returned dict
            - If select 1 label, orbitals show be (2, 2) list like [[4, Orbital.s], [4, Orbital.py]]
            - If select more than 1 labels, orbitals show be the same length as label
        '''
        if type(label) is list:
            label = [str(l) for l in label]
            if not orbitals:
                return self.COXP.get_summed_cohp_by_label_list(label, divisor).as_dict()
            else:
                return self.COXP.get_summed_cohp_by_label_and_orbital_list(label, orbitals, divisor).as_dict()
        else:
            label = str(label)
            if not orbitals:
                return self.COXP.get_cohp_by_label(label).as_dict()
            else:
                return self.COXP.get_orbital_resolved_cohp(label, orbitals).as_dict()
    
    def get_icohp(self, label):
        '''
        Function to get the ICOHP for a specified bond.
        
        Parameters:
            - label: Bond indexes / int
        Return:
            - The ICOHP
        '''
        label = str(label)
        cohp = self.COXP.get_cohp_by_label(label).as_dict()
        for i, energy in enumerate(cohp['energies']):
            if energy > 0:
                icohp = cohp['ICOHP']['1'][i]
                break
        return icohp

    def show_spd_typ(self, site):
        '''
        Function to show oribtals contained in the density of states on a specified site

        Parameter:
            - site: The site / int
        Cautions:
            - This atom index count from 0
        '''
        for k in list(self.DOS.get_site_spd_dos(self.DOS.structure[site]).keys()):
            print(k)

    def get_dos(self, site, orbitals='d', spin=Spin.up):
        '''
        Function to get DOS data

        Parameters:
            - site: The site / int
            - orbitals: Orbitals appear in show_spd_typ function / list or str, you can use 'd', 'p', 's' to represent corresponding orbitals
            - spin: Spin direction / pymatgen.electronic_structure.core.Spin, default Spin.up
        Return:
            - A dict that stores the DOS data
        Cautions:
            - This atom index count from 0
        '''
        density_list = self.DOS.get_site_spd_dos(self.DOS.structure[site])
        orbital_list = list(density_list.keys())
        if orbitals == 'd':
            orbitals = [o for o in orbital_list if 'd' in o]
        elif orbitals == 'p':
            orbitals = [o for o in orbital_list if 'p' in o]
        elif orbitals == 's':
            orbitals = [o for o in orbital_list if 's' in o]

        for i, orbital in enumerate(orbitals):
            if i == 0:
                density = density_list[orbital].densities[spin]
            else:
                density = density + density_list[orbital].densities[spin]
        return {'energies': self.DOS.energies, 'dos': np.array(density)}
