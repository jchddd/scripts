from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list_pbc
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Structure

from matplotlib import pyplot as plt
from math import floor, ceil
import numpy as np
import random
import os

import Jworkflow.dataset as dataset
import Jworkflow.utility as utility
from Jworkflow.plot import view_structure_VASP, plot_slab


def random_substitute(struc, substi_element, substi_atom, atom_identify_way='layer'):
    '''
    Function to substitute selected atoms with specified elements in equal proportion 

    Parameters:
        - struc: The structure of the atoms that need to be replaced / pymatgen.core.Structure
        - substi_element: Elements to substitute / (n) list
        - substi_atom: Atoms to be substituted / (n) list, int
        - atom_identify_way: The way to select atoms / str in 'layer' and 'index', default 'layer'
    Return:
        - Structure after substitution
    '''
    # load Layer Divide Process
    LDP = Layer_Divide_Process()
    LDP.print_info = False
    LDP.load_slab(struc)
    LDP.Divide_Layer()
    # read input infor
    if atom_identify_way == 'layer':
        atom_list = np.array(LDP.identify_layer(substi_atom, 'layer'))
    elif atom_identify_way == 'index':
        atom_list = np.array(substi_atom)
    elements = substi_element
    # initialize parameter
    structure = struc.copy()
    element_number = len(elements)
    atom_number = len(atom_list)
    # calculate substitute amount for each element
    substi_base = atom_number // element_number
    substi_rema = atom_number % element_number
    substi_counts = [substi_base] * element_number
    for _ in range(substi_rema):
        substi_counts[_] += 1
    # apply random to atom index
    index_random = list(range(atom_number))
    random.shuffle(index_random)
    # substitute atom
    index_start = 0
    for _, element in enumerate(elements):
        substi_list = index_random[index_start: index_start + substi_counts[_]]
        index_start += substi_counts[_]
        for substi_atom in atom_list[substi_list]:
            structure[int(substi_atom)] = element
    return structure


def Z_average(structure):
    '''
    Function to calculate average z coordinate for all atoms

    Parameter:
        - structure: the structure / pymatgen.core.Structure
    '''
    frac_coords = structure.frac_coords
    return np.sum(frac_coords, axis=0)[2] / frac_coords.shape[0]


def get_vector(structure, atom_from, atom_to, is_frac=True):
    '''
    Function to get a vector between two atomic coordinates

    Parameters:
        - structure, pymatgen.core.Structure: The structure file
        - atom_from, int: Start atom of the vector
        - atom_to, int: End atom of the vector
        - is_frac, bool, default True: Use fractional coordinates or Cartesian coordinates
    Return:
        - The vector
    Cautions:
        - Make sure the coordinates are in cell to get coorrect result
    '''
    # get coords for two atoms
    if is_frac:
        coord_from = structure.frac_coords[atom_from]
        coord_to = structure.frac_coords[atom_to]
    else:
        coord_from = structure.cart_coords[atom_from]
        coord_to = structure.cart_coords[atom_to]
    # calculate the vector
    vector = [coord_to[0] - coord_from[0], coord_to[1] - coord_from[1], coord_to[2] - coord_from[2]]
    return vector


class Layer_Divide_Process():
    '''
    Class for layering structures and fixing or moving atoms for each layer individually

    Available Functions:
        - load_slab: Function to load slab structure
        - divide_layer: Functions that perform layered operations on the slab
        - identify_layer: Function for Determining atomic layers or obtaining atoms of a specific layer
        - fix_layer: Function to fix coordinates according to the atomic layer
        - move_layer: Function for moving atoms for specific layers
        - delete_layer: Function to delete layer atoms
        - rotate_layer: Function to rotate layers
        - reset_structure: Function to reset the processed structure to inital structure
        - save_structure: Function to save the processed structure to POSCAR file
        - view_structure: Function to view the processed structure
    '''

    def __init__(self):
        '''
        Attributes:
            - work_path: Path to where store the output files / str,path
            - structure_init: The initial structure / pymatgen.core.Structure
            - structure_process: Structures after fixed or mobile processing / pymatgen.core.Structure
            - devide_method: Names of layering methods and their key parameters / (2) list, [method name, the key parameter]
            - layer_refer: reference z coordinates for help with layering / (l) list, l is the number of layers
            - layer_number: Total number of layer according to the devide mode / int
            - layer_list: Layer for each atom. Note that the layer start from 0 / (n) list, n is the atom number in slab
            - layer_bar: Histogram data that counts the number of atoms in each layer / (l) list, l is the number of layers
            - print_info: Whether to pring running information / bool, default True
        '''
        self.work_path = None
        self.structure_init = None
        self.structure_process = None
        self.devide_method = []
        self.layer_refer = []
        self.layer_number = 0
        self.layer_list = []
        self.layer_bar = []

        self.print_info = True

    def load_slab(self, structure):
        '''
        Function to load slab structure

        Parameter:
            - structure: Input slab structure / path or pymatgen.Structure
        Accomplish:
            - Read structure to structure_init and copy it to structure_process
        '''
        # read structure and set work path
        if type(structure) == type('a'):
            self.structure_init = Structure.from_file(structure)
            self.work_path = os.path.split(os.path.abspath(structure))[0]
        else:
            self.structure_init = structure
            self.work_path = os.getcwd()
        self.structure_process = self.structure_init.copy()
        # print info
        if self.print_info:
            utility.screen_print('Load Slab')
            utility.screen_print('Load structure  ', 'Compeleted')
            utility.screen_print('Slab atom number', str(len(self.structure_init)))
            utility.screen_print('End')

    def divide_layer(self, identify_method='threshold', method_parameter=0.36):
        '''
        Functions that perform layered operations on the slab

        Parameters:
            - identify_method: Method to devide atomic layer / str in 'round' or 'threshold', default 'threshold'
                The 'round' method uses round() to take an approximation of the z coordinates. After the
                approximation, atoms with the same coordinates are considered to be at the same layer.
                The 'threshold' will search refer z In sequence. Atoms with z closes to one specific refer
                z with a difference less than a threshold value will be considerd to be at the same layer.
            - method_parameter: method parameter / int or float, default 0.36
                In 'round' method, it is the int parameter in round(), like 2 or 3
                In 'threshold', it is the threshold value, a value like 0.3 is rational
        '''
        # build layer refer
        self.devide_method = [identify_method, method_parameter]
        # round method
        if identify_method == 'round':
            atomic_zs_frac = [round(coord[2], method_parameter) for coord in self.structure_init.frac_coords]
            unique_z_frac = list(set(atomic_zs_frac))
            unique_z_frac.sort()
            self.layer_refer = unique_z_frac
        # threshold method
        elif identify_method == 'threshold':
            atomic_zs_cart = [coord[2] for coord in self.structure_init.cart_coords]
            refer_z_cart = [atomic_zs_cart[0]]
            for atomic_z in atomic_zs_cart:
                to_refer = True
                for refer_z in refer_z_cart:
                    if abs(atomic_z - refer_z) <= method_parameter:
                        to_refer = False
                if to_refer:
                    refer_z_cart.append(atomic_z)
            refer_z_cart.sort()
            self.layer_refer = refer_z_cart
        # collect info
        self.layer_number = len(self.layer_refer)
        self.layer_bar = [0] * self.layer_number
        self.layer_list = self.identify_layer(range(len(self.structure_init)))
        for atom_index in range(len(self.structure_init)):
            self.layer_bar[self.layer_list[atom_index]] += 1
        # print info
        if self.print_info:
            utility.screen_print('Devide Layer')
            utility.screen_print('Layer number', str(self.layer_number))
            utility.screen_print('Layer refer ', [round(i, 2) for i in self.layer_refer])
            utility.screen_print('Layer bar   ', self.layer_bar)
            utility.screen_print('End')

    def identify_layer(self, identify_list, by='atom'):
        '''
        Function for Determining atomic layers or obtaining atoms of a specific layer

        Parameters:
            - identify_list: List of data that need to used at identified layers / (n) list
            - by: Input atom indexes to identify layers or input layers to find atoms / str, 'atom' or 'layer', default 'atom'
        Return:
            - 'atom', return a (n) list of layers corresponding to each atoms
            - 'layer', return a (n) list of atoms that locate on these layers
        '''
        # init
        identiyf_list = []
        # identify by atom
        if by == 'atom':
            for atom_index in identify_list:
                if self.devide_method[0] == 'round':
                    atomic_z = round(self.structure_init.frac_coords[atom_index][2], self.devide_method[1])
                    layer = self.layer_refer.index(atomic_z)
                    identiyf_list.append(layer)
                elif self.devide_method[0] == 'threshold':
                    atomic_z = self.structure_init.cart_coords[atom_index][2]
                    for refer_z in self.layer_refer:
                        if abs(atomic_z - refer_z) <= self.devide_method[1]:
                            layer = self.layer_refer.index(refer_z)
                            break
                    identiyf_list.append(layer)
        # identify by layer
        elif by == 'layer':
            for atom_index, atomic_layer in enumerate(self.layer_list):
                if atomic_layer in identify_list:
                    identiyf_list.append(atom_index)
        # return
        return identiyf_list

    def fix_layer(self, layers, relax_dynamics=[True, True, True], fix_dynamics=[False, False, False]):
        '''
        Function to fix coordinates according to the atomic layer

        Parameters:
            - layers: Atomic layers to be fixed / (x) list
            - relax_dynamics: Realx atom dynamics / (3) list, default [True, True, True]
            - fix_dynamics: Fix atom dynamics / (3) list, default [False, False, False]
        Accomplish:
            - Add selective dynamics site property to structure_process
        '''
        # init
        fix_list = self.identify_layer(layers, 'layer')
        SD_list = []
        # add selective dynamics
        for atomic_index in range(len(self.structure_init)):
            if atomic_index in fix_list:
                SD_list.append(fix_dynamics)
            elif atomic_index not in fix_list:
                SD_list.append(relax_dynamics)
        self.structure_process.add_site_property("selective_dynamics", SD_list)

    def move_layer(self, layers, vector, is_frac=True, to_unit_cell=True):
        '''
        Function for moving atoms for specific layers

        Parameters:
            - layers: Atomic layers to be moved / (x) list
            - vector: Translation vector for sites / (3) array
            - is_frac: Whether the vector corresponds to fractional or Cartesian coordinates / bool, default True
            - to_unit_cell: Whether new sites are transformed to unit cell / bool, default True
        Accomplish:
            - Move specified layers on structure_process
        '''
        move_list = self.identify_layer(layers, 'layer')
        self.structure_process.translate_sites(move_list, vector, is_frac, to_unit_cell)

    def delete_layer(self, layers):
        '''
        Function to delete layer atoms

        Parameter:
            - layers: List of which layers to delete / list
        Accomplish:
            - Delete specified layers on structure_process
        Caution:
            - It is not recommended to continue other operations after deleting unless you re divide_layer
        '''
        delete_list = self.identify_layer(layers, 'layer')
        self.structure_process.remove_sites(delete_list)

    def rotate_layer(self, layer, theta, anchor, axis=[0, 0, -1], to_unit_cell=True):
        '''
        Function to rotate layers

        Parameters:
            - layer: List of which layers to rotate / (x) list
            - theta: Rotation angle of layers in unit ° / float
            - anchor: The point around which the rotation revolves in cartesian / (3) array
            - axis: The axis of rotation / 3D-array-like, default [0,0,-1]
            - to_unit_cell: Whether new sites are transformed to unit cell / bool, default True
        Accomplish:
            - Rotate specified layers on structure_process
        '''
        rotate_list = self.identify_layer(layer, 'layer')
        self.structure_process.rotate_sites(rotate_list, theta / 180 * np.pi, axis, anchor, to_unit_cell)

    def reset_structure(self):
        '''
        Function to reset the processed structure to inital structure
        '''
        self.structure_process = self.structure_init.copy()

    def save_structure(self, file):
        '''
        Function to save the processed structure to POSCAR file

        Parameters"
            - file, str: Saved file name
        '''
        utility.save_structure(self.structure_process, os.path.join(self.work_path, file))
        if self.print_info:
            utility.screen_print('Save structure')
            utility.screen_print('Saved file', os.path.join(self.work_path, file))
            utility.screen_print('END')

    def view_structure(self):
        '''
        Function to view the processed structure
        '''
        view_structure_VASP(self.structure_process)


class Slab_Adsorption():
    '''
    Class for cleaving facets and adding adsorption molecules

    Available Functions:
        - read_structure: Function to read structures
        - deal_name: Function to deal names in self.strus_name
        - P2C: Function to Turn Primitive Cell to Convention Unit Cell for structures in self.strus_bulk
        - make_supercell: Function to scale slabs in self.strus_slab
        - surface_check: Function to check surface information
        - cleave_surface: Function to cleave surface
        - fix_layers: Function to fix layers for all slabs on self.stru_slab
        - redefine_surface: Function to redefine surface and subsurface
        - find_adsorption_site: Function to find adsorption site
        - add_molecule: Function to add molecule to adsorption site
        - show_slab: Function to show slab or site information
        - view_slab: Function to view slab structures
        - save_file: Function to save slab or adsorption files
        - add_adsorption_sites: Function to add adsorption sites
    '''

    def __init__(self):
        '''
        Attributes:
            - work_direction: The working direction, Files will be saved herestr / path
            - strus_bulk: List that stores the read in bulk structures / (n) list, pymatgen.Structure
            - strus_name: List that stores the read in structure names / (n) list, str
            - strus_slab: List that stores the cleaved or read in slab structures / (n) list, pymatgen.Structure
            - strus_ads: List that stores the adsorption structures split by individual slab
            / (n,x) list, pymatgen.Structure, n slab number, x ads structure number
            - strus_ads_name: List of adsorption names without slab name / (x) list, str
            - unit_layer: The number of layers of repeating units / int
            - miller_index: The miller index / (3) purple
            - ads_sites_fra_all: List of all adsorption site coords
            / (n,3,x,3) list, n slab number, 3 three type of sites, x site number, 3 xyz coords
            - ads_sites_fra_unique: List of unique adsorption site coords / (n,3,x,3) list
            - diff_refer_name: Slab names that have different symmetry operation or site number / list
            - devide_para: Parameters used to determine the layering method / (2) list, default ['round', 2]
            - print_info: Whether to print running information / bool, default True
            - interval: The interval at which the progress bar is updated / int, default 6
        '''
        self.work_direction = ''

        self.strus_bulk = []
        self.strus_name = []
        self.strus_slab = []
        self.strus_ads = []
        self.strus_ads_name = []

        self.unit_layer = 0
        self.miller_index = None

        self.ads_sites_fra_all = []
        self.ads_sites_fra_unique = []

        self.diff_refer_name = []

        self.devide_para = ['threshold', 0.3]
        self.print_info = True
        self.interval = 6

    def read_structure(self, path, file_type='bulk'):
        '''
        Function to read structures

        Parameters:
            - path: Path to where the structures are stored. Parent directory will be set to work_direction / str(path)
            - file_type: File types of the read in structure / str, 'bulk' or 'slab', default 'bulk'
        Accomplish:
            - Initialize and update strus_bulk, strus_slab, strus_name and work_direction
        Caution:
            - The file name should be like 'Ru2Cu.cif' which contain the slab/substrate name and file type
        '''
        # init
        self.strus_bulk = []
        self.strus_name = []
        self.strus_slab = []
        self.work_direction = os.path.dirname(path)
        file_list = utility.get_file_or_subdirection(path)
        # read file
        for file in file_list:
            structure = Structure.from_file(os.path.join(path, file))
            name = file.split('.')[0]
            self.strus_name.append(name)
            if file_type == 'bulk':
                self.strus_bulk.append(structure)
            elif file_type == 'slab':
                self.strus_slab.append(structure)
        # print info
        if self.print_info:
            utility.screen_print('Read Structure')
            utility.screen_print('Structure number', str(len(file_list)))
            utility.screen_print('File type', file_type)
            utility.screen_print('END')

    def deal_name(self, mode, len_char_pos):
        '''
        Function to deal names in self.strus_name

        Parameters:
            - mode: Deal type / str, 'del'-delete, 'add', 'sep'-seperate by '_'
            - len_char_pos: if 'del', it means how many characters to delete from the back / int
                            if 'add', it means what characters to add after the names / str
                            if 'sep', it meats which part will be remain / list
        '''
        if mode == 'del':
            for i in range(len(self.strus_name)):
                self.strus_name[i] = self.strus_name[i][:-1 * len_char_pos]
        elif mode == 'add':
            for i in range(len(self.names)):
                self.strus_name[i] = self.strus_name[i] + len_char_pos
        elif mode == 'sep':
            for i in range(len(self.strus_name)):
                self.strus_name[i] = '_'.join([self.strus_name[i].split('_')[j] for j in len_char_pos])

    def P2C(self):
        '''
        Function to turn Primitive Cell to Convention Unit Cell for structures in self.strus_bulk
        '''
        for i in range(len(self.strus_bulk)):
            self.strus_bulk[i] = utility.convert_pri_con(self.strus_bulk[i], 'ptc')

    def make_supercell(self, scaling_matrix=[2, 2, 1]):
        '''
        Function to scale slabs in self.strus_slab

        Parameter:
            - scaling_matrix : 3D-list or (3, 3) tuple, default [2,2,1]
             You can use (3, 3) tuple to change lattice vector like ((-1, 1, 0), (1, 1, 0), (0, 0, 1))
        Caution:
            - All numbers must be integer.
        '''
        for slab in self.strus_slab:
            slab.make_supercell(scaling_matrix)

    def surface_check(self, miller_index, print_layer_species=True, structure_index=0, primitive=False, xnp=1):
        '''
        Function to check surface information accordint to miller index and structure index

        Parameters:
            - miller_index: The miller index will be stored and can be ignored in Cleave_Surface / 3D-tuple, like (1,1,1)
            - print_layer_species: Whether to print species for each layer / bool ,default True
            - structure_index: Decide which structure is used to check surface information / int, default 0
            - primitive: Whether to reduce any generated slabs to a primitive cell / bool, default False
            - xnp: Max normal search parameters, Increase this value if no orthogonal lattice can be found / int, default 1 
        '''
        self.miller_index = miller_index
        # check termination number, height
        sg = SlabGenerator(self.strus_bulk[structure_index], miller_index, 1, 10, max_normal_search=max(miller_index) + xnp, primitive=primitive)
        unit_termination = len(sg.get_slabs())
        unit_height = sg._proj_height
        unit_ceil = sg.oriented_unit_cell
        if abs(unit_ceil.lattice.angles[2] - 90) > 0.666 or abs(unit_ceil.lattice.matrix[2][0]) > 0.666 \
        or abs(unit_ceil.lattice.matrix[2][1] > 0.666) or abs(unit_ceil.lattice.matrix[2][3]) < 0.666:
                unit_ceil = sg.get_slabs()[0]
                if abs(unit_ceil.lattice.angles[2] - 90) > 0.666:
                    utility.screen_print('Warning !!', 'Max normal search failed! please increase xnp')
        # check layers of the unit slab
        LD = Layer_Divide_Process()
        LD.print_info = False
        LD.load_slab(unit_ceil)
        LD.divide_layer(self.devide_para[0], self.devide_para[1])
        if utility.check_numbers_close_to_0_and_1(unit_ceil.frac_coords[:,2]):
            self.unit_layer = LD.layer_number - 1
        else:
            self.unit_layer = LD.layer_number
        # check repeat layer
        min_slab_size = 1.5 * unit_height
        min_vacuum_size = 1.5 * unit_height
        sg = SlabGenerator(self.strus_bulk[structure_index], miller_index, min_slab_size, \
                           min_vacuum_size, max_normal_search=max(miller_index) + xnp, center_slab=True, primitive=primitive)  # 初始化Slab生成器
        slab = sg.get_slabs()[0]
        LD = Layer_Divide_Process()
        LD.print_info = False
        LD.load_slab(slab)
        LD.divide_layer(self.devide_para[0], self.devide_para[1])
        
        continue_repeat_check = True
        check_time = 0
        repeat_layer_numbers = []
        repeat_layer_lists = []
        while(continue_repeat_check):
            repeat_layer_number = 0
            repeat_layer_list = []
            for layer_n in range(LD.layer_number):
                if layer_n == 0:
                    layer_atominfo = set(
                        [(round(slab.frac_coords[i][0], 2) % 1, round(slab.frac_coords[i][1], 2) % 1, slab[i].specie.symbol) for i in LD.identify_layer([layer_n + check_time], 'layer')])
                    layer_atominfo_refer = layer_atominfo
                else:
                    layer_atominfo = set(
                        [(round(slab.frac_coords[i][0], 2) % 1, round(slab.frac_coords[i][1], 2) % 1, slab[i].specie.symbol) for i in LD.identify_layer([layer_n+ check_time], 'layer')])
                if layer_atominfo == layer_atominfo_refer and layer_n != 0:
                    break
                else:
                    repeat_layer_number += 1
                    repeat_layer_list.append([atominfo[2] for atominfo in layer_atominfo])
            if repeat_layer_number not in repeat_layer_numbers:
                repeat_layer_numbers.append(repeat_layer_number)
                repeat_layer_lists.append(repeat_layer_list)
                check_time += 1
            else:
                continue_repeat_check = False
        repeat_layer_number = repeat_layer_numbers[repeat_layer_numbers.index(max(repeat_layer_numbers))]
        repeat_layer_list = repeat_layer_lists[repeat_layer_numbers.index(max(repeat_layer_numbers))]
        # print info
        if self.print_info:
            utility.screen_print('Surface_Check')
            utility.screen_print('System', self.strus_name[structure_index])
            utility.screen_print('Miller index', str(miller_index))
            utility.screen_print('Terminations', str(unit_termination))
            utility.screen_print('Unit cell layers', str(self.unit_layer))
            utility.screen_print('Periodicity layers', str(repeat_layer_number))
            if print_layer_species:
                for c, rl in enumerate(repeat_layer_list):
                    if c == 0:
                        utility.screen_print('Layer species', str(rl))
                    else:
                        utility.screen_print('', str(rl))
            utility.screen_print('END')
        self.strucs_slab = slab

    def cleave_surface(self, layers=4, vacuum_length=15, scaling_matrix=[1, 1, 1], fix_layers=2, miller_index=None, select_shift=None, center_slab=True, primitive=False, to=None, xnp=1, mdis=0.06):
        '''
        Function to cleave surface

        Parameter:
            - layers: Decide the layer number of slab / int, default 4
            - vacuum_length : Decide the vacuum length / float, default 15
            - scaling_matrix: To scale the slab / 3D-list, default [1,1,1]
            - fix_layers: Decide how many or which layers will be fixed / int or list, default 2
            - miller_index: Miller index of the surface / 3D-tuple, default self.miller_index that defined from surface_check
            - select_shift: Integer to control exposed facet / int, 0
            - center_slab: Whether to center the slab. If not, a 0.06 distance will be kept from the 0 z coordinate / bool, default True
            - primitive: Whether to reduce any generated slabs to a primitive cell / bool, default False
            - to: Decide how many slab to cleave / int, default len(self.strus_bulk), set from 1 to default
            - xnp: Max normal search parameters, Increase this value if no orthogonal lattice can be found / int, default 1
            - mdis: Min distance between bottom atoms and boundary if center_slab is False / float, default 0.06
        '''
        utility.screen_print('Cleave_Surface') if self.print_info else None
        self.strus_slab = []
        # initial parameters
        if not to:
            to = len(self.strus_bulk)
        if miller_index == None and self.miller_index != None:
            miller_index = self.miller_index
        if self.unit_layer == 0:
            sg = SlabGenerator(self.strus_bulk[0], miller_index, 1, 10, max_normal_search=max(miller_index) + xnp, primitive=primitive)
            unit_ceil = sg.oriented_unit_cell
            if abs(unit_ceil.lattice.angles[2] - 90) > 0.666 or abs(unit_ceil.lattice.matrix[2][0]) > 0.666 \
            or abs(unit_ceil.lattice.matrix[2][1] > 0.666) or abs(unit_ceil.lattice.matrix[2][3]) < 0.666:
                unit_ceil = sg.get_slabs()[0]
                if abs(unit_ceil.lattice.angles[2] - 90) > 0.666:
                    utility.screen_print('Warning !!', 'Max normal search failed! please increase xnp')
            LD = Layer_Divide_Process()
            LD.print_info = False
            LD.load_slab(unit_ceil)
            LD.divide_layer(self.devide_para[0], self.devide_para[1])
            if utility.check_numbers_close_to_0_and_1(unit_ceil.frac_coords[:,2]):
                self.unit_layer = LD.layer_number - 1
            else:
                self.unit_layer = LD.layer_number
        # delete parameters and n_shift
        if layers % self.unit_layer != 0:
            delete = True
            delete_layer = self.unit_layer - layers % self.unit_layer
        else:
            delete = False
            delete_layer = 0
        n_shift = 0  # default
        if delete:
            n_shift = n_shift + delete_layer
        if select_shift != None:
            n_shift = select_shift
        # print cleave parameters
        if self.print_info:
            utility.screen_print('Miller index', str(miller_index))
            utility.screen_print('Layer number', str(layers))
            utility.screen_print('Vacuum length', str(vacuum_length))
            utility.screen_print('Scaling matrix', str(scaling_matrix))
            utility.screen_print('Layer shift n', str(n_shift))
        length = len(self.strus_bulk[0:to])
        # loop and cleave surfaces
        for i, structure in enumerate(self.strus_bulk[0:to]):
            # cleave an initial slab
            sg = SlabGenerator(structure, miller_index, 1, 1, max_normal_search=max(miller_index) + xnp, primitive=primitive)
            unit_height = sg._proj_height
            min_slab_size = (ceil(layers / self.unit_layer)) * unit_height
            sg = SlabGenerator(structure, miller_index, min_slab_size, 1, max_normal_search=max(miller_index) + xnp, center_slab=True, primitive=primitive)
            # shifts = sg._calculate_possible_shifts() # pymatgen old version
            # slab_raw = sg.get_slab(shifts[n_shift]) # pymatgen old version
            slab_raw = sg.get_slabs()[n_shift % len(sg.get_slabs())] # pymatgen new version
            
            LD = Layer_Divide_Process()
            LD.print_info = False
            LD.load_slab(slab_raw)
            LD.divide_layer(self.devide_para[0], self.devide_para[1])
            if delete or LD.layer_number - delete_layer != layers:
                LD.delete_layer([l for l in range(delete_layer)])
                if LD.layer_number - delete_layer != layers:
                    LD.reset_structure()
                    delete_layer = LD.layer_number - layers
                    LD.delete_layer([l for l in range(delete_layer)])
            slab_raw = LD.structure_process
            # get coords and element type from raw slab
            species = slab_raw.species
            cart_coords_raw = slab_raw.cart_coords
            cart_coords_z_raw = [coord[2] for coord in cart_coords_raw]
            max_z_raw = max(cart_coords_z_raw)
            min_z_raw = min(cart_coords_z_raw)
            # recleave slab
            lattice_raw = slab_raw.lattice.matrix
            if center_slab:
                z_change = (vacuum_length / 2) - min_z_raw
                cart_coords_new = [[coord[0], coord[1], coord[2] + z_change] for coord in cart_coords_raw]
                max_z_new = max([coord[2] for coord in cart_coords_new])
                lattice_new = [list(lattice_raw[0]), list(lattice_raw[1]), [0.0, 0.0, max_z_new + vacuum_length / 2]]
            elif not center_slab:
                mdis = 0.06
                cart_coords_new = [[coord[0], coord[1], coord[2] - min_z_raw + mdis] for coord in cart_coords_raw]
                max_z_new = max([coord[2] for coord in cart_coords_new])
                lattice_new = [list(lattice_raw[0]), list(lattice_raw[1]), [0.0, 0.0, max_z_new + vacuum_length - mdis]]
            slab = Structure(lattice=lattice_new, species=species, coords=cart_coords_new, coords_are_cartesian=True)
            slab.make_supercell(scaling_matrix)
            # store slab
            self.strus_slab.append(slab)
            if i == 0 or i % self.interval == 0:
                progress = (i + 1) / length
                utility.screen_print('Surface cleavage', '{:5}'.format(str(round(progress * 100, 2))) + '%', '\r', '') if self.print_info else None
        utility.screen_print('Surface cleavage', 'compeleted', '\r') if self.print_info else None
        utility.screen_print('Slab atom number', str(len(self.strus_slab[0].species))) if self.print_info else None
        # fix layer
        self.fix_layers(fix_layers, True)
        utility.screen_print('END') if self.print_info else None

    def fix_layers(self, fix_layer, internal=False):
        '''
        Function to fix layers for all slabs on self.stru_slab

        Parameter:
            - fix_layer: Decide how many or which layers will be fixed / int or list, default 2
            - internal: Whether its use internal and not print start and END information / bool, default False
        '''
        layer_number_refer = 0
        layer_differ_list = []
        for i, slab in enumerate(self.strus_slab):
            LD = Layer_Divide_Process()
            LD.print_info = False
            LD.load_slab(slab)
            LD.divide_layer(self.devide_para[0], self.devide_para[1])
            if type(fix_layer) == int:
                fix_list = [f for f in range(fix_layer)]
            elif type(fix_layer) == list:
                fix_list = fix_layer
            LD.fix_layer(fix_list)
            self.strus_slab[i] = LD.structure_process
            # check if layer number is different
            layer_number = LD.layer_number
            if i == 0:
                layer_number_refer = layer_number
            if layer_number != layer_number_refer:
                layer_differ_list.append(self.strus_name[i])
        if self.print_info:
            utility.screen_print('Fix_Layer') if not internal else None
            utility.screen_print('Layer number', str(layer_number_refer))
            utility.screen_print('Fixed layer', str(fix_list))
            if len(layer_differ_list) != 0:
                utility.screen_print('Warning !!!', 'The number of layers on the surface is inconsistent')
                utility.screen_print('Inconsistent list', str(layer_differ_list))
            utility.screen_print('END') if not internal else None
    
    def redefine_surface(self, surface_layers):
        '''
        Function to redefine surfaces and subsurfaces
        
        Parameter:
            - surface_layers: Which layer will be confirm as surface, others to be subsurface / list
        '''
        
        for i, slab in enumerate(self.strus_slab):
            atom_list = range(len(slab))
            LD = Layer_Divide_Process()
            LD.print_info = False
            LD.load_slab(slab)
            LD.divide_layer(self.devide_para[0], self.devide_para[1])
            atom_layers = LD.identify_layer(atom_list)
            site_properties = slab.site_properties
            site_properties['surface_properties'] = []
            for atom in atom_list:
                if atom_layers[atom] in surface_layers:
                    site_properties["surface_properties"].append('surface')
                else:
                    site_properties["surface_properties"].append('subsurface')
            self.strus_slab[i] = slab.copy(site_properties = site_properties)
            
    def find_adsorption_site(self, distance=2.0, refer_layer=None, site_type_precision=[3, 3, 3], no_obtuse_hollow=False):
        '''
        Function to find adsorption site

        parameters
            - distance: distance between sires and surface / float, default 2.0
            - refer_layer: Find sites on which layer, upper layers will be delete / int, default None(the top one)
            - site_type_precision: The round precision of ontop, bridge and hollow sites that uses to sort sites / 3D-list, default [3,3,3]
            - no_obtuse_hollow: Whether to include obtuse triangular ensembles in hollow sites / bool, default False
        '''
        # init
        utility.screen_print('Find_Adsorption_Site') if self.print_info else None
        utility.screen_print('Site distance', str(distance)) if self.print_info else None
        self.ads_sites_fra_all = []
        self.ads_sites_fra_unique = []
        diff_refer_index = []
        self.diff_refer_name = []
        length = len(self.strus_slab)
        # loop and find adsorption side for each slab
        for i in range(length):
            # get slab
            slab = self.strus_slab[i]
            if refer_layer:
                LDP = Layer_Divide_Process()
                LDP.print_info = False
                LDP.load_slab(slab)
                LDP.divide_layer(self.devide_para[0], self.devide_para[1])
                LDP.delete_layer(np.arange(refer_layer + 1, LDP.layer_number, 1))
                slab = LDP.structure_process
            # get all adsorption sites
            asf = AdsorbateSiteFinder(slab)
            ads_sites_sym0 = asf.find_adsorption_sites(symm_reduce=0, distance=distance, no_obtuse_hollow=no_obtuse_hollow)
            sym_op = set(SpacegroupAnalyzer(slab).get_symmetry_operations())
            # classify sites with coordination and sort sites with distance to center
            self.ads_sites_fra_all.append([])
            self.ads_sites_fra_unique.append([])
            site_number = []
            site_number_unique = []
            for k in ['ontop', 'bridge', 'hollow']:
                rp = site_type_precision[['ontop', 'bridge', 'hollow'].index(k)]
                coords_car = ads_sites_sym0[k]
                if len(coords_car) != 0:
                    coords_fra = [slab.lattice.get_fractional_coords(coord) for coord in coords_car]
                    ran = np.array(coords_fra)
                    ran2 = np.abs(ran - 0.5)
                    ran3 = np.array([[round(min(ran2[i, :2]), rp), round(max(ran2[i, :2]), rp), round(ran[i][0], rp), round(ran[i][1], rp)] for i in range(len(ran2))])
                    idex = np.lexsort((ran3[:, 3], ran3[:, 2], ran3[:, 0], ran3[:, 1]))
                    coords_fra_sorted = ran[idex, :]
                elif len(coords_car) == 0:
                    coords_fra_sorted = []
                self.ads_sites_fra_all[i].append(coords_fra_sorted)
                site_number.append(len(coords_fra_sorted))
                # distinguish unique and not unique sites
                unique_coords_fra = []
                for coord in coords_fra_sorted:
                    incoord = False
                    for op in sym_op:
                        if in_coord_list_pbc(unique_coords_fra, op.operate(coord), atol=1e-4):  # e-6
                            incoord = True
                            break
                    if not incoord:
                        unique_coords_fra.append(coord)
                self.ads_sites_fra_unique[i].append(unique_coords_fra)
                site_number_unique.append(len(unique_coords_fra))
            # Compare space groups and site numbers
            if i == 0:
                sym_op_refer = sym_op
                site_number_refer = site_number
                site_number_unique_refer = site_number_unique
            elif i != 0:
                if sym_op != sym_op_refer or site_number != site_number_refer or site_number_unique != site_number_unique_refer:
                    diff_refer_index.append(i)
            if i == 0 or i % self.interval == 0:
                progress = (i + 1) / length
                utility.screen_print('Find adsorption site', '{:5}'.format(str(round(progress * 100, 2))) + '%', '\r', '') if self.print_info else None
        # print info
        if self.print_info:
            utility.screen_print('Find adsorption site', 'compeleted', '\r')
            utility.screen_print('Unique site number',
                                 'ontop: ' + str(site_number_unique_refer[0]) + ', bridge: ' + str(site_number_unique_refer[1]) + ', hollow: ' + str(site_number_unique_refer[2]))
            utility.screen_print('Total site number', 'ontop: ' + str(site_number_refer[0]) + ', bridge: ' + str(site_number_refer[1]) + ', hollow: ' + str(site_number_refer[2]))
        # unequal symmetry operation or site number
        if len(diff_refer_index) != 0:
            for j in diff_refer_index:
                self.diff_refer_name.append(self.strus_name[j])
            if self.print_info:
                utility.screen_print('Warning !!!', 'There is unequal in symmetry operation or site number')
                utility.screen_print('Warning !!!', 'You can check self.diff_refer_name for detail')
        # unequal adsorption site coordinate
        different_matrix = np.array(self.ads_sites_fra_all, dtype=object)
        uneqal_site_list = [[], [], []]
        for str_index in range(len(self.strus_slab)):
            for site_type_index in range(3):
                different_matrix[str_index][site_type_index] = different_matrix[str_index][site_type_index] - np.array(self.ads_sites_fra_all[0][site_type_index])
        for str_index in range(len(self.strus_slab)):
            for site_type_index in range(3):
                for site_index in range(len(different_matrix[str_index][site_type_index])):
                    if max(np.abs(different_matrix[str_index][site_type_index][site_index])) >= 0.06:
                        uneqal_site_list[site_type_index].append(site_index)
        uneqal_site_list[0].sort()
        uneqal_site_list[1].sort()
        uneqal_site_list[2].sort()
        if self.print_info:
            utility.screen_print('Unequal  top   site', str(set(uneqal_site_list[0])))
            utility.screen_print('Unequal bridge site', str(set(uneqal_site_list[1])))
            utility.screen_print('Unequal hollow site', str(set(uneqal_site_list[2])))
            utility.screen_print('END')

    def add_molecule(self, parameters):
        '''
        Function to add molecule to adsorption site

        Parameter:
                                          molecule  rotation    site
            - parameters : (X,3,n)-D list, like： [ [['CH3COOH'], [0,90], ['1h','0t']],
                          X and n are arbitrary   [['CH3','CH2'], [0,0], ['0t','8b']] ]
                molecule: moelecule name list can be found in Jworkflow.dataset.Uniform_Name_Dict().values()
                rotation: rotation angle in unit °. Positive and negative represent clockwise and counterclockwise
                        A (3) list can also use to control the rotation angle in three dimentions
                site: site consists of a index and site type. Index is refer to all sites, not unique sites.
                        site type include 't'-ontop, 'b'-bridge and 'h'-hollow
                execution way: if number of molecule == rotation == site,
                        Adsorption structure will be construced using all corresponding molecule, rotation and site
                        if numberof molecule==1 and rotation == site,
                        Adsorption structures will be construced for the molecule using corresponding rotation and site
                        if numberof molecule==1 and rotation != site,
                        Adsorption structures will be construced for the molecule using each combination of rotation and site
        '''
        utility.screen_print('Add_Molecule') if self.print_info else None
        # init
        self.strus_ads = []
        self.strus_ads_name = []
        type_sqeuence = ['t', 'b', 'h']
        length = len(self.strus_slab) * len(parameters)
        # loop and add molecule
        for i, slab in enumerate(self.strus_slab):
            ads_structures = []
            for j, adsb_stru_info in enumerate(parameters):
                slab_pure = slab.copy()
                ml = adsb_stru_info[0]  # molecule list
                rl = adsb_stru_info[1]  # rotate list
                sl = adsb_stru_info[2]  # site list
                rls = ['='.join([str(it) for it in r]) if type(r) == list else r for r in rl]
                # case 1 : ml == rl == sl add all molecule in one slab
                if len(ml) == len(rl) and len(rl) == len(sl) and len(sl) == len(ml):
                    for k in range(len(ml)):
                        asf = AdsorbateSiteFinder(slab_pure)
                        molecule, is_translate = dataset.load_molecule(ml[k], rl[k])
                        site_type = sl[k][-1]
                        site_index = int(sl[k][:-1])
                        site_car_coord = slab.lattice.get_cartesian_coords(self.ads_sites_fra_all[i][type_sqeuence.index(site_type)][site_index])
                        slab_pure = asf.add_adsorbate(molecule, site_car_coord, translate=is_translate)
                    ads_structures.append(slab_pure)
                    if i == 0:
                        ads_name = ''
                        ads_name += '-'.join([m for m in ml]) + '_' + '-'.join([str(r) for r in rls]) + '_' + '-'.join([s for s in sl])
                        self.strus_ads_name.append(ads_name)
                # case 2 : ml == 1 , rl == sl add molecule based on the rotation and site of the same index
                elif len(ml) == 1 and len(rl) == len(sl):
                    for k in range(len(rl)):
                        asf = AdsorbateSiteFinder(slab_pure)
                        molecule, is_translate = dataset.load_molecule(ml[0], rl[k])
                        site_type = sl[k][-1]
                        site_index = int(sl[k][:-1])
                        site_car_coord = slab.lattice.get_cartesian_coords(self.ads_sites_fra_all[i][type_sqeuence.index(site_type)][site_index])
                        ads_structure = asf.add_adsorbate(molecule, site_car_coord, translate=is_translate)
                        ads_structures.append(ads_structure)
                        if i == 0:
                            ads_name = ''
                            ads_name = ml[0] + '_' + str(rls[k]) + '_' + sl[k]
                            self.strus_ads_name.append(ads_name)
                # case 3 : ml == 1 rl != sl，add molecule for each combination of rotation and site
                elif len(ml) == 1 and len(rl) != len(sl):
                    for ks in range(len(sl)):
                        for kr in range(len(rl)):
                            asf = AdsorbateSiteFinder(slab_pure)
                            molecule, is_translate = dataset.load_molecule(ml[0], rl[kr])
                            site_type = sl[ks][-1]
                            site_index = int(sl[ks][:-1])
                            site_car_coord = slab.lattice.get_cartesian_coords(self.ads_sites_fra_all[i][type_sqeuence.index(site_type)][site_index])
                            ads_structure = asf.add_adsorbate(molecule, site_car_coord, translate=is_translate)
                            ads_structures.append(ads_structure)
                            if i == 0:
                                ads_name = ''
                                ads_name = ml[0] + '_' + str(rls[kr]) + '_' + sl[ks]
                                self.strus_ads_name.append(ads_name)
                if i == 0 or (i * len(parameters) + j + 1) % self.interval == 0:
                    progress = (i * len(parameters) + j + 1) / length
                    utility.screen_print('Add Molecule', '{:5}'.format(str(round(progress * 100, 2))) + '%', '\r', '') if self.print_info else None
            self.strus_ads.append(ads_structures)
        if self.print_info:
            utility.screen_print('Add Molecule', 'compeleted', '\r')
            utility.screen_print('Ads structure / slab', str(len(self.strus_ads_name)))
            utility.screen_print('Total Ads structure', str(len(self.strus_ads_name) * len(self.strus_name)))
            utility.screen_print('END')

    def show_slab(self, slab_type='slab', slab_index=0, ads_list=None):
        '''
        Function to show slab or site information

        Parameters:
            - slab_type: Which type of slab to plot / str, 'slab', 'unique', 'site', 'adss', default 'slab'
                        slab: Only one pure slab | unique: slab with its unique adsorption sites
                        site: slab with its all adsorption sites | adss: adsorption structures
            - slab_index: Which slab to plot / int, default 0
            - ads_list: Decide which adsorption structures to show / list, int, default all

        '''
        xy = np.abs(self.strus_slab[slab_index].lattice.matrix[:2, :2]).sum(axis=0)
        xvy = xy[0] / xy[1]
        if slab_type == 'slab':
            slab = self.strus_slab[slab_index]
            asf = AdsorbateSiteFinder(slab)
            fig = plt.figure(figsize=(10 * xvy, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(self.strus_name[slab_index], fontsize=36, pad=10)
            ax.axis('off')
            plot_slab(ax, slab, decay=0.06)
            for site in asf.surface_sites:
                ax.text(site.coords[0], site.coords[1], site.species, zorder=6667, ha='center', va='center', fontsize=30)
        elif slab_type == 'unique':
            siten = ['ontop', 'bridge', 'hollow']
            slab = self.strus_slab[slab_index]
            fig = plt.figure(figsize=(30 * xvy, 10))
            for i, site_type in enumerate(siten):
                ax = fig.add_subplot(1, 3, i + 1)
                ax.axis('off')
                if len(self.ads_sites_fra_unique[slab_index][i]) != 0:
                    ax.set_title(site_type, fontsize=30, pad=10)
                    plot_slab(ax, slab, decay=0.06)
                    for j, site in enumerate(self.ads_sites_fra_unique[slab_index][i]):
                        car_coord = slab.lattice.get_cartesian_coords(site)
                        ax.scatter(car_coord[0], car_coord[1], s=999, zorder=6667, c='red', alpha=0.66)
                        ax.text(car_coord[0], car_coord[1], str(j), ha='center', va='center', fontsize=36, zorder=6668)
        elif slab_type == 'site':
            siten = ['ontop', 'bridge', 'hollow']
            slab = self.strus_slab[slab_index]
            fig = plt.figure(figsize=(30 * xvy, 10))
            for i, site_type in enumerate(siten):
                ax = fig.add_subplot(1, 3, i + 1)
                ax.axis('off')
                if len(self.ads_sites_fra_all[slab_index][i]) != 0:
                    ax.set_title(site_type, fontsize=36, pad=10)
                    plot_slab(ax, slab, decay=0.06, cover_outside=None)
                    for j, site in enumerate(self.ads_sites_fra_all[slab_index][i]):
                        car_coords = slab.lattice.get_cartesian_coords(site)
                        ax.text(car_coords[0], car_coords[1], str(j), zorder=6667, ha='center', va='center', fontsize=30)
        elif slab_type == 'adss':
            if ads_list == None:
                ads_list = range(len(self.strus_ads_name))
            stru_num = len(ads_list)
            row = 3
            column = ceil(stru_num / row)
            fig = plt.figure(figsize=(30 * xvy, 10 * column))
            for i, stru_index in enumerate(ads_list):
                stru_ads = self.strus_ads[slab_index][stru_index]
                ax = fig.add_subplot(column, 3, i + 1)
                ax.axis('off')
                plot_slab(ax, stru_ads, decay=0.06)
                ax.set_title(self.strus_ads_name[stru_index], fontsize=36, pad=10)

    def view_slab(self, slab_type='slab', ads_list=None, slab_index=0):
        '''
        Function to view slab structures

        Parameters:
            - slab_type: Which type of slab to view / str, 'slab', 'adss', default 'slab'
            - ads_list: Decide which adsorption structures to show / list, int, default all
            - slab_index: Which slab to plot / int, default 0
        '''
        if slab_type == 'slab':
            view_structure_VASP(self.strus_slab[slab_index])
        elif slab_type == 'adss':
            if ads_list is None:
                ads_list = range(len(self.strus_ads_name))
            for i in ads_list:
                view_structure_VASP(self.strus_ads[slab_index][i])

    def save_file(self, file_type='adss', path=None, dir_name=None):
        '''
        Function to save slab or adsorption files

        Parameters:
            - file_type: Decide whether to save slab or adsorption files / str, 'slab 'or 'adss', default 'adss'
            - path: Path to where save structures / str, default work path
            - dir_name: direction name / str, default Pure_slab or Ads_structures
        '''
        count = 0
        # creat save path
        if path is None:
            path = self.work_direction
        if dir_name is None:
            dir_name = {'slab': 'Pure_slab', 'adss': 'Ads_structures'}[file_type]
        path_save = os.path.join(path, dir_name)
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        # save file
        if file_type == 'slab':
            for i, slab in enumerate(self.strus_slab):
                utility.save_structure(slab, os.path.join(path_save, self.strus_name[i] + '.vasp'))
                count += 1
        elif file_type == 'adss':
            for i, adsses in enumerate(self.strus_ads):
                for j, adss in enumerate(adsses):
                    utility.save_structure(adss, os.path.join(path_save, self.strus_name[i] + '_' + self.strus_ads_name[j] + '.vasp'))
                    count += 1
        if self.print_info:
            utility.screen_print('Save_File')
            utility.screen_print('Save File', str(path_save))
            utility.screen_print('File number', str(count))
            utility.screen_print('END')

    def add_adsorption_sites(self, sites):
        '''
        Function to add additonal adsorption sites in the form of fractional coordinates

        Parameter:
            - sites: sites that you want to add / (n, 2) list, like [[[0.48,0.37,0.72], 'top'], [[0.36,0.36,0.72], 'hollow']]
        Cautions:
            - You can use 'x' in z axis to use the Pre-generated z coordinates
        '''
        site_to_index = {'top': 0, 'bridge':1, 'hollow': 2}
        slab_number = len(self.strus_slab)
        if len(self.ads_sites_fra_all) == 0:
            for i in range(slab_number):
                self.ads_sites_fra_all.append([np.array([]), np.array([]), np.array([])])
        for site in sites:
            i_site = site_to_index[site[1]]
            for i_slab in range(slab_number):
                if len(self.ads_sites_fra_all[i_slab][i_site]) == 0:
                    self.ads_sites_fra_all[i_slab][i_site] = np.array([site[0]])
                else:
                    if site[0][-1] == 'x':
                        site[0][-1] = self.ads_sites_fra_all[i_slab][i_site][0][-1]
                    self.ads_sites_fra_all[i_slab][i_site] = np.concatenate((self.ads_sites_fra_all[i_slab][i_site], [site[0]]), axis=0)
