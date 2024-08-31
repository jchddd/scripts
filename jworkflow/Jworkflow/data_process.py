from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure

import pandas as pd
import numpy as np
import os
import re

from Jworkflow.slab import Layer_Divide_Process
from Jworkflow.utility import screen_print, save_structure, get_file_or_subdirection



def get_vasp_results(result_type, path=None):
    '''
    Function for extracting the calculation result of VASP
    
    Parameters:
        - result_type: Type of the extract result / str
            'E': total energy
            'C': bool value about whether the calculation is compeleted
            'L': Total number of ion step
            'M': Total magnetism in OSZICAR
            'N': Max number of electronic step
            'G': Gibbs free energy correction from ther_info
            'Z': Zero point energy correction from ther_info
            'H': Enthalpy correction from ther_info
            'S': Entropy correction from ther_info
        - path: Path to where VASP output file exist / str, path
    Return:
        - The result value
    '''
    # init
    if path == None:
        path = os.getcwd()
    # extract value
    if result_type == 'E':
        f = open(path + '\\OUTCAR')
        for line in f:
            match = re.findall('  energy  without entropy=*', line)
            if match != []:
                result = float(line.split('=')[2])
        f.close()
    elif result_type == 'C':
        result = False
        f = open(path + '\\OUTCAR')
        for line in f:
            match = re.findall(' reached required accuracy - stopping structural energy minimisation*', line)
            if match != []:
                result = True
        f.close()
    elif result_type == 'T':
        result = -1
        f = open(path + '\\OUTCAR')
        for line in f:
            match = re.findall('Elapsed time*', line)
            if match != []:
                result = float(line.split(':')[1])
        f.close()
    elif result_type == 'L':
        f = open(path + '\\OSZICAR')
        for line in f:
            match = re.findall('F=*', line)
            if match != []:
                result = int(line.split('F')[0])
        f.close()
    elif result_type == 'M':
        f = open(path + '\\OSZICAR')
        for line in f:
            match = re.findall('F=*', line)
            if match != [] and 'mag' in line:
                result = float(line.split('mag=')[1])
            else:
                result = 0
        f.close()
    elif result_type == 'N':
        es = 0
        f = open(path + '\\OSZICAR')
        for line in f:
            match = re.findall(':', line)
            if match != []:
                es += 1
        result = es
        f.close()
    elif result_type == 'G':
        f = open(path + '\\ther_info')
        for line in f:
            match = re.findall('Thermal correction to G(T)*', line)
            if match != []:
                result = float(line.split('mol')[1].split('eV')[0])
        f.close()
    elif result_type == 'Z':
        f = open(path + '\\ther_info')
        for line in f:
            match = re.findall('Zero-point energy E_ZPE*', line)
            if match != []:
                result = float(line.split('mol')[1].split('eV')[0])
        f.close()
    elif result_type == 'H':
        f = open(path + '\\ther_info')
        for line in f:
            match = re.findall('Thermal correction to H(T)*', line)
            if match != []:
                result = float(line.split('mol')[1].split('eV')[0])
        f.close()
    elif result_type == 'S':
        f = open(path + '\\ther_info')
        for line in f:
            match = re.findall('Entropy S *', line)
            if match != []:
                result = float(line.split(')')[1].split('eV')[0])
        f.close()
    elif result_type == 'formula':
        s = Structure.from_file(path + '\\CONTCAR')
        result = s.formula
    # assert if value not exists
    try:
        return result
    except UnboundLocalError:
        screen_print('Error', result_type + ' in ' + path + ' Value not found !!!', start='\n')
        return 666


class Adss_DataExtract_PostProcess():
    '''
    Class to Extract VASP results and do some simple post-processing with structures
    
    Available Functions:
        - extract: Extract VASP results from a path that include multiple folders which store the outputs
        - Poscar_Process: Function to postprocess POSCARs
    Internal Functions:
        - locate_index: Get Element type according to their atom type
        - get_atom_shift: Get the shift of atoms between POSCAR and CONTCAR
        - get_bond_change: Get the bond length change between POSCAR and CONTCAR
        - fix_slab: Fix slab atom
        - prepare_chgcar_diff: Prepare input file for chgcar differ calculation
    '''

    def __init__(self):
        '''
        Attributes:
            - ele_adsb: Elements that will be treated as adsorbate / list, pymatgen.Element
            - ele_skel: Elements that will be treated as skeleton of the adsorbate / list, pymatgen.Element
            - height_threshold: The height criterion for judging the adsorbed molecules / int or float
            - height_type: Methods for judging height / str in 'layer' or 'z'
            - poscar: POSCAR name / str, default 'POSCARoo'
            - print_info: Whether to print running information / bool, default True
            - interval: The interval at which the progress bar is updated / int, default 6
        '''
        self.ele_adsb = [Element.N, Element.H, Element.C, Element.O]
        self.ele_skel = [Element.N]
        self.height_threshold = None
        self.height_type = 'layer'
        self.poscar = 'POSCARoo'

        self.print_info = True
        self.interval = 6

    def reset_type_element(self, ele_adsb, ele_skel):
        '''
        Function to reset elements for adsorbate and molecular skeleton

        Parameters:
            - ele_adsb: elements for adsorbate / (n) list
            - ele_skel: elements for molecular skeleton / (n) list
        Accomplish:
            - Update ele_adsb and ele_skel in Attributes
        Example:
            - DEP.reset_element(['N','H'],['N'])
        '''
        self.ele_adsb = [Element(e) for e in ele_adsb]
        self.ele_skel = [Element(e) for e in ele_skel]
        if self.print_info:
            screen_print('Reset type element')
            screen_print('Adsorbate elements', ele_adsb)
            screen_print('Skeleton elements', ele_skel)
            screen_print('End')

    def set_height_filter(self, height_threshold, height_type='layer'):
        '''
        Function to open and set parameter for height filter when screen out atoms

        Parameters:
            - height_threshold: threshold that used to filter atom by height / int, float, default None
                    atoms with height > threshold will be choose
            - height_type: The type of threshold / str in 'layer', 'z', default 'layer'
        Accomplish:
            - The height criterion is used in addition to the element criterion in the selection of atoms
        '''
        self.height_threshold = height_threshold
        self.height_type = height_type
        if self.print_info:
            screen_print('Set_height_filter')
            screen_print('Height threshold', str(height_threshold))
            screen_print('Height type', height_type)
            screen_print('End')

    def extract(self, path, task_type, write_file=False, file_name='auto', write_file_path=None):
        '''
        Function that extract VASP results from a path that include multiple folders which store the outputs

        Parameters:
            - path: Path that store multiple folders which store the outputs / str, path
            - task_type: Calculation result type / str in 'adss', 'slab', 'Gcor', 'sta'
            - write_file: Whether to write the extract DataFrame to Excel / bool, default False
            - file_name: Name of the Excel file / 'str', default 'auto'
            - write_file_path: Path where to store the output Excel file / path, directory two levels ahead path
        Return:
            - The DataFrame that stores all results
        Cautions:
            - Make sure there are not folders in directions that store VASP results
        '''
        # init
        if self.print_info:
            screen_print('Extract')
            screen_print('Direction', path)
            screen_print('Task type', task_type)
        path_iterate = []
        for i in os.walk(path):
            if i[1] == []: path_iterate.append(i[0])
        data_number = len(path_iterate)
        # decide extract value type
        if task_type == 'adss':
            res_list = ['ads_sys', 'system', 'adsb', 'site', 'rotate', 'energy', 'converg', 'mtransla_skel', 'mupgrade_skel', 'mtransla_adsb', 'mdista_adsb', 'mshift_slab',
                        'mtransla_slab', 'mupgrade_slab', 'Etime', 'setp']
        elif task_type == 'slab':
            res_list = ['system', 'energy', 'converg', 'mshift_slab', 'ashift_slab', 'mtransla_slab', 'mupgrade_slab', 'Etime', 'setp']
        elif task_type == 'Gcor':
            res_list = ['ads_sys', 'system', 'adsb', 'G', 'ZPE', 'H', 'S', 'TS']
        elif task_type == 'sta':
            res_list = ['system', 'formula', 'energy', 'mag', 'Etime', 'setp', 'estep']
        res_dict = dict()
        for res_key in res_list:
            res_dict[res_key] = []
        # loop path and extract result
        for i, path in enumerate(path_iterate):
            ads_sys = path.split('\\')[-1]
            for key in res_dict:
                if key == 'ads_sys':
                    res_dict[key].append(ads_sys)
                elif key == 'system':
                    res_dict[key].append(ads_sys.split('_')[0])
                elif key == 'adsb':
                    res_dict[key].append(ads_sys.split('_')[1])
                elif key == 'site':
                    res_dict[key].append(ads_sys.split('_')[-1])
                elif key == 'rotate':
                    if len(ads_sys.split('_')) == 3:
                        res_dict[key].append('0')
                    else:
                        res_dict[key].append(ads_sys.split('_')[-2])
                elif key == 'mshift_slab':
                    res_dict[key].append(self.get_atom_shift(path, 'slab', 'max', 'xyz'))
                elif key == 'ashift_slab':
                    res_dict[key].append(self.get_atom_shift(path, 'slab', 'adverage', 'xyz'))
                elif key == 'mtransla_slab':
                    res_dict[key].append(self.get_atom_shift(path, 'slab', 'max', 'xy'))
                elif key == 'mupgrade_slab':
                    res_dict[key].append(self.get_atom_shift(path, 'slab', 'max', 'z'))
                elif key == 'mtransla_skel':
                    res_dict[key].append(self.get_atom_shift(path, 'skel', 'max', 'xy'))
                elif key == 'mupgrade_skel':
                    res_dict[key].append(self.get_atom_shift(path, 'skel', 'max', 'z'))
                elif key == 'mtransla_adsb':
                    res_dict[key].append(self.get_atom_shift(path, 'adsb', 'max', 'xy'))
                elif key == 'mdista_adsb':
                    res_dict[key].append(self.get_bond_change(path=path))
                elif key == 'energy':
                    res_dict[key].append(get_vasp_results('E', path))
                elif key == 'converg':
                    res_dict[key].append(get_vasp_results('C', path))
                elif key == 'Etime':
                    res_dict[key].append(get_vasp_results('T', path))
                elif key == 'setp':
                    res_dict[key].append(get_vasp_results('L', path))
                elif key == 'mag':
                    res_dict[key].append(get_vasp_results('M', path))
                elif key == 'estep':
                    res_dict[key].append(get_vasp_results('N', path))
                elif key == 'G':
                    res_dict[key].append(get_vasp_results('G', path))
                elif key == 'ZPE':
                    res_dict[key].append(get_vasp_results('Z', path))
                elif key == 'H':
                    res_dict[key].append(get_vasp_results('H', path))
                elif key == 'S':
                    res_dict[key].append(get_vasp_results('S', path))
                elif key == 'formula':
                    res_dict[key].append(get_vasp_results('formula', path))
                elif key == 'TS':
                    res_dict[key].append(0)
            if i == 0 or i % self.interval == 0:
                progress = (i + 1) / data_number * 100
                screen_print('Extract progress', str(round(progress, 2)) + '%', end='', start='\r') if self.print_info else None
        screen_print('Extract progress', 'Compeleted', start='\r', end='\n') if self.print_info else None
        # to DataFrame
        result = pd.DataFrame(res_dict)
        if task_type == 'adss':
            result.sort_values(by=['system', 'adsb', 'site', 'rotate'], inplace=True)
        else:
            result.sort_values(by=['system'], inplace=True)
        result.reset_index(drop=True, inplace=True)
        if task_type == 'Gcor':
            result['H'] = result['H'] - result['ZPE']
            result['TS'] = result['S'] * 298.15
        screen_print('DataFrame sort', 'Compeleted') if self.print_info else None
        # write file
        if write_file:
            if file_name == 'auto':
                file_name = task_type + '_' + str(len(result.index)) + '.xlsx'
            else:
                file_name = file_name
            if write_file_path is None:
                write_file_path = os.path.dirname(os.path.dirname(path))
            file = os.path.join(write_file_path, file_name)
            result.to_excel(file)
            screen_print('Write file', file_name) if self.print_info else None
        else:
            screen_print('Write file', 'False') if self.print_info else None
        screen_print('End') if self.print_info else None

        return result

    def get_atom_shift(self, path, atom_type, cal_mode, dimention):
        '''
        Function to get the shift of atoms between POSCAR and CONTCAR
        
        Parameters:
            - path: Direction path where POSCAR and CONTCAR exist / path
            - atom_type: Which type of atom to calculate / in 'adsb', 'skel', 'slab'
            - cal_mode: How to deal with all the shifts of atoms / str in 'adverage', 'max'
            - dimention: Which dimention to calculate the shift / str in 'xy', 'xyz', 'z'
        Return:
            - The shift of atoms
        '''
        # init
        path_pos = os.path.join(path, self.poscar)
        stru_pos = Structure.from_file(path_pos)
        path_cont = os.path.join(path, 'CONTCAR')
        try:  # assert if CONTCAR is empty
            stru_cont = Structure.from_file(path_cont)
        except IndexError:
            screen_print('Error', path + ' CONTCAR is empty !!!', start='\n')
            return 666
        scale = {'xy': np.array([1, 1, 0]), 'z': np.array([0, 0, 1]), 'xyz': np.array([1, 1, 1])}[dimention]
        # merge POSCAR and CONTCAR
        farc_coords_unite = np.array(list(stru_pos.frac_coords) + list(stru_cont.frac_coords)) * scale
        species_unite = stru_pos.species + stru_cont.species
        stru_unite = Structure(lattice=stru_pos.lattice, species=species_unite, coords=farc_coords_unite)
        # determine which atoms to be calculate
        atom_cal_index = self.locate_index(stru_pos, atom_type)
        # calculate shifts
        shift_all = []
        atom_number = len(stru_pos.species)
        for atom_index in atom_cal_index:
            shift_atom = stru_unite.get_distance(atom_index, atom_index + atom_number)
            if dimention == 'z' and stru_unite.frac_coords[atom_index + atom_number][2] < stru_unite.frac_coords[atom_index][2]:
                shift_atom = shift_atom * (-1)  # turn into negative value if the atom move downshift if dimention is 'z'
            shift_all.append(shift_atom)
        if len(shift_all) == 0:  # add 0 value if not atom is used to calculate
            shift_all.append(0)
        # statistical treatment
        if cal_mode == 'adverage':
            result = sum(shift_all) / atom_number
        elif cal_mode == 'max':
            shift_all = np.array(shift_all)
            shift_abs = abs(shift_all)
            result = shift_all[list(shift_abs).index(max(shift_abs))]
        # return
        return np.round(result, 3)

    def get_bond_change(self, path, atom_type='adsb', max_bond=1.5, return_type='max'):
        '''
        Function to get the bond length change between POSCAR and CONTCAR

        Parameters:
            - path: Direction path where POSCAR and CONTCAR exist / str, path
            - atom_type: Which type of atoms to get calculate bonds / str, 'adsb', 'skel', 'slab'
            - max_bond: Max length of bond. Bonds beyond this distance in POSCAR will be ignored / float, default 1.5
            - self.poscar: Name of the POSCAR / str, default 'POSCARoo'
            - return_type: Determine ways to retun / str in 'max' or 'dict', default 'max'
                    'max' will only return one value corresponding to max bond change.
                    'dict' will return a dict include all bonds and their distance changes.
        Return:
            - The result
        '''
        # init
        path_pos = os.path.join(path, self.poscar)
        stru_pos = Structure.from_file(path_pos)
        path_cont = os.path.join(path, 'CONTCAR')
        try:  # check if CONTCAR is empty
            stru_cont = Structure.from_file(path_cont)
        except IndexError:
            screen_print(p3_word1='Error', p4_word2=path + ' CONTCAR is empty !!!', p6_start='\n')
            return 666
        # determine which atoms to be calculate
        atom_cal_index = self.locate_index(stru_pos, atom_type)
        # find bonds between those atoms
        bonds = []
        for i in atom_cal_index:
            for j in atom_cal_index:
                if i != j and stru_pos.get_distance(i, j) < max_bond:
                    bond = {i, j}
                    if bond not in bonds:
                        bonds.append(bond)
        # calculate bond length change
        if len(bonds) > 0:
            distances_pos = [stru_pos.get_distance(list(i)[0], list(i)[1]) for i in bonds]
            distances_cont = [stru_cont.get_distance(list(i)[0], list(i)[1]) for i in bonds]
            changes = [distances_cont[i] - distances_pos[i] for i in range(len(distances_pos))]
            changes_abs = abs(np.array(changes))
            max_change = changes[list(changes_abs).index(max(changes_abs))]
        else:
            max_change = 0.0
        # return
        if return_type == 'max':
            return np.round(max_change, 3)
        elif return_type == 'dict':
            return {'bonds': bonds, 'changes': changes}

    def locate_index(self, structure, atom_type):
        '''
        Function to get Element type according to their atom type
        
        Parameters:
            - structure: Input structure / pymatgen.core.Structure
            - atom_type: type of located atoms / str in 'adsb', 'slab', 'skel'
        Return:
            - List of located index
        '''
        # extract calculate element type
        ele_all = set(structure.species)
        if atom_type == 'adsb':
            ele_cal = [e for e in self.ele_adsb if e in ele_all]
        elif atom_type == 'skel':
            ele_cal = [e for e in self.ele_skel if e in ele_all]
        elif atom_type == 'slab':
            ele_cal = list(ele_all - set(self.ele_adsb))
        # choose index according to element
        atom_cal_index_by_e = []
        for index, ele in enumerate(structure.species):
            if ele in ele_cal:
                atom_cal_index_by_e.append(index)
        cal_index = atom_cal_index_by_e
        # filter index according to height
        if self.height_threshold is not None and atom_type != 'slab':
            atom_cal_index_by_h = []
            # init Layer_Divide_Process
            if self.height_type == 'layer':
                LDP = Layer_Divide_Process()
                LDP.print_info = False
                LDP.load_slab(structure)
                LDP.divide_layer()
            # loop and filter
            for atom in atom_cal_index_by_e:
                if self.height_type == 'layer' and LDP.identify_layer([atom])[0] > self.height_threshold:
                    atom_cal_index_by_h.append(atom)
                elif self.height_type == 'z' and structure.cart_coords[atom][2] > self.height_threshold:
                    atom_cal_index_by_h.append(atom)
            cal_index = atom_cal_index_by_h
        # return
        return cal_index

    def process_POSCAR(self, path, deal_type):
        '''
        Function to post process POSCARs
        
        Parameters:
            - path: Path that store multiple folders which store the outputs / str, path
            - deal_type: Post-processing type / str in 'fre' or 'chargediff'
                    'fre': fix all slab atoms for vibration calculation
                    'chargediff': seperate adsorbate and slab for calculating charge difference
        Accomplish:
            - Make a new direction in upper path and sotre new POSCARs there
        '''
        # init
        if self.print_info:
            screen_print('Process POSCAR')
            screen_print('Direction', path)
            screen_print('Deal type', deal_type)
        # make direction
        upper_path = os.path.dirname(path)
        if deal_type == 'fre':
            store_path = os.path.join(upper_path, 'fre')
            if not os.path.exists(store_path):
                os.mkdir(store_path)
        elif deal_type == 'chargediff':
            store_path_adsb = os.path.join(upper_path, 'chargediff', 'adsb')
            store_path_slab = os.path.join(upper_path, 'chargediff', 'slab')
            store_path_all = os.path.join(upper_path, 'chargediff', 'all')
            if not os.path.exists(os.path.join(upper_path, 'chargediff')):
                os.mkdir(os.path.join(upper_path, 'chargediff'))
            if not os.path.exists(store_path_adsb):
                os.mkdir(store_path_adsb)
            if not os.path.exists(store_path_slab):
                os.mkdir(store_path_slab)
            if not os.path.exists(store_path_all):
                os.mkdir(store_path_all)
        screen_print('Created direction', deal_type) if self.print_info else None
        files = get_file_or_subdirection(path, 'file')
        length = len(files)
        for j, file in enumerate(files):
            if deal_type == 'fre':
                stru_fix = self.fix_slab(os.path.join(path, file))
                save_structure(stru_fix, os.path.join(store_path, file))
            elif deal_type == 'chargediff':
                slab, adsb, stru_all = self.prepare_chgcar_diff(os.path.join(path, file))
                save_structure(slab, os.path.join(store_path_slab, file))
                save_structure(adsb, os.path.join(store_path_adsb, file))
                save_structure(stru_all, os.path.join(store_path_all, file))
        screen_print('END') if self.print_info else None

    def fix_slab(self, contcar):
        '''
        Function to fix slab atom
        
        Parameter:
            - contcar: Path and file name to a CONTCAR file / str, path
        Return:
            - The fixed structure
        '''
        stru_cont = Structure.from_file(contcar)
        sd_list = []
        adsb_list = self.locate_index(stru_cont, 'adsb')
        for i in range(len(stru_cont)):
            if i not in adsb_list:
                sd_list.append([False, False, False])
            else:
                sd_list.append([True, True, True])
        stru_cont.add_site_property("selective_dynamics", sd_list)
        return stru_cont

    def prepare_chgcar_diff(self, contcar):
        '''
        Function to prepare input file for chgcar differ calculation
        
        Parameter:
            - contcar: Path and file name to a CONTCAR file / str, path
        Return:
            - slab structure, adsorbate structure, origin structure
        '''
        stru_cont = Structure.from_file(contcar)
        species = stru_cont.species
        lattice = stru_cont.lattice.matrix
        frac_coords_raw = [stru_cont.sites[k].frac_coords for k in range(len(stru_cont.sites))]
        spe_slab = []
        spe_adsb = []
        coord_slab = []
        coord_adsb = []
        adsb_list = self.locate_index(stru_cont, 'adsb')
        for i in range(len(stru_cont)):
            if i in adsb_list:
                spe_adsb.append(species[i]);
                coord_adsb.append(frac_coords_raw[i])
            elif i not in adsb_list:
                spe_slab.append(species[i]);
                coord_slab.append(frac_coords_raw[i])
        slab = Structure(lattice=lattice, species=spe_slab, coords=coord_slab)
        adsb = Structure(lattice=lattice, species=spe_adsb, coords=coord_adsb)
        return slab, adsb, stru_cont
