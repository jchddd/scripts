from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from ase.calculators.emt import EMT
from ase import Atoms

import matplotlib.pyplot as plt

from shutil import copy
import pandas as pd
import numpy as np
import os

# ====== Part 1: base tools =========================================================

def move_file(file_list, move_from, move_to, file_suffix='.vasp', return_ignore=False):
    '''
    Function that moves specified files from one direction to another

    Parameters:
        - file_list: The names of the file to move. Files that are not found will be ignored / (n) list
        - move_from: The folder to search for the target files to remove it / str(path)
        - move_to: The folder to which to move the file in. a directory will be created if it does not exist / str(path)
        - file_suffix: Suffixes of files, will be added to the file names in file_list / str, default '.vasp'
        - return_ignore: Whether to retuen a list of ignored files / bool, default False
    Return:
        - List of file names that are ignored from moving if return_ignore is True
    '''
    # init
    ignore_list = []
    if not os.path.exists(move_to):
        os.mkdir(move_to)
    # loop and move file
    for file in file_list:
        file_name = file + file_suffix
        file_source = os.path.join(move_from, file_name)
        file_copy = os.path.join(move_to, file_name)
        if os.path.exists(file_source):
            copy(file_source, file_copy)
        else:
            ignore_list.append(file)
    # return ignore
    if return_ignore:
        return ignore_list

def get_file_or_subdirection(path, typ='file'):
    '''
    Function that gets all subdirectories or files in a specified directory

    Parameters:
        - path: Directory where the command is executed / str(path)
        - typ: Return files or subdirectories / str, 'file' or 'dir', default 'file'
    Return:
        - List of file or direction names
    '''
    for curDir, dirs, files in os.walk(path):
        if curDir == path and typ == 'file':
            return files
        elif curDir == path and typ == 'dir':
            return dirs

def write_list_to_file(list_of_strings, file, file_system='Linux'):
    '''
    Function to write values in a list into a file

    Parameters:
        - list_of_strings: the list of strings / list
        - file: write file / path (str)
        - file_system: which system the file is used to / str, 'Linux' or 'Windows'
    '''
    if   file_system == 'Linux':
        line_ending = '\n'
    elif file_system == 'Windows':
        line_ending = '\r\n'
    else:
        raise ValueError("file_system must be 'Linux' or 'Windows'")

    with open(file, 'w', encoding='utf-8', newline='') as file:
        for item in list_of_strings:
            file.writelines(item + line_ending)
    file.close()

# ====== Part 2: tools for structure transformation ========================================

def convert_pri_con(structure, convert_type='ptc'):
    '''
    Function convert pymatgen.structure between primitive and conventional cell

    Parameters:
        - structure: Structure that needs to convert / pymatgen.Structure
        - convert_type: primitive to conventional, or conventional to primitive / str, 'ptc' or 'ctp', default 'ptc'
    Return:
        - The corresponding conventional or primitive cell in pymatgen.Structure
    '''
    if convert_type == 'ptc':
        return SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    elif convert_type == 'ctp':
        return structure.get_primitive_structure()

def save_structure(structure, file):
    '''
    Function to save structure to POSCAR file with sort species

    Parameter:
        - structure: Structure that need to write to a POSCAR / pymatgen.core.Structure
        - file: File that the structure write to / str(path)
    '''
    poscar = Poscar(structure, sort_structure=True)
    poscar.write_file(file)

def convert_atomstru(atsr, to='sta'):
    '''
    Function to convert between pymatgen.Structure and ase.atoms class
    
    Parameters:
        - atsr: atoms or Structure item / default None
        - to: Conversion Type / 'sta' or 'ats'
    Return:
        - Structure or atoms
    '''
    
    if to == 'sta':
        return AseAtomsAdaptor.get_atoms(atsr)
    elif to == 'ats':
        return AseAtomsAdaptor.get_structure(atsr)

# ====== Part 3: Calculation and unit conversion ===============================================

def calculate_surface_energy(E_slab,N_slab,E_bulk,N_bulk,A,E_slab_unrelax,method=0,unit='J/m^2'):
    '''
    Function to calculate surface energy
    
    Parameters:
        - E_slab: Slab energy in eV / float
        - N_slab: Slab atom number / int
        - E_bulk: Bulk energy in eV / float
        - N_bulk: Bulk atom number / int
        - A: Slab surface area in A^2 / float
        - E_slab_unrelax: Unrelaxed slab energt in eV / float
        - method: 0 or 1 / int
          0 -> E = 1/2A*(Eslab-Ebulk/Nbulk*Nslab) for slab that relax both side and fix center part
          1 -> E = 1/2A*(EslabUnrelax-Ebulk/Nbulk*Nslab)+1/A*(Eslab-EslabUnrelax) for slab that only relax one side
        - unit: Unit for surface energy / str, 'eV/A^2' or 'J/m^2', default 'J/m^2'
    Return:
        - The calculated surface energy
     '''
    
    if method == 0:
        se = 1 / (2 * A) * (E_slab - E_bulk / N_bulk * N_slab)
    elif method == 1: # Nanoscale, 2017,9, 13089-13094
        se = 1 / (2 * A) * (E_slab_unrelax - E_bulk / N_bulk * N_slab) + 1 / A * (E_slab - E_slab_unrelax)
        
    if unit == 'eV/A^2':
        return se
    elif unit == 'J/m^2':
        return se * 16.02

def zero_point_energy_from_cm_2_eV(ZPE):
    '''
    Function to convert units cm^-1 to units eV
    
    Parameter:
        - ZPE: Zero point energy in cm^-1 / float
    Return:
        - Zero point energy in eV
    '''
    return ZPE * 0.12398424290373065 / 1000 

def free_energy_correction_from_experiment(H, S, T=298.15):
    '''
    Function to calculate the free energy correction from experimental enthalpy(H) and entropy (S).
    
    Parameters:
        - H: Integrated enthalpy from 0 to T K in K J mol^-1 / float
        - S: entropy at T K in J K^-1 mol^-1 / float
        - T: Temperature in K / float, default 298.15
    Return:
        - H, TS and free energy correction in eV
    '''
    H = H * 0.01036427
    S = S * 0.0000103642723
    G = H - T * S
    return {'G': G, 'H': H, 'S': S}

def calculate_EMT_energy(structure):
    '''
    Function that uses ASE module to calculate potential energy of the structure by EMT

    Parameter:
        - structure: pymatgen.core.Structure
    Return:
        - The EMT potential energy of the structure
    '''
    structure = AseAtomsAdaptor.get_atoms(structure)
    structure.set_calculator(EMT())
    return structure.get_potential_energy()
    
# ====== Part 4: build-in function =============================================================

def screen_print(word1='', word2=None, start='', end='\n', length=36):
    '''
    Functions used to print execution information and split lines on the screen

    Parameters:
        - word1: The first printed sentence / str, default ''
            line model  - The character in the center of the split lines
            state model - The character in the left of the colon
        - word2: The second printed sentencestr / default None, if None 'line' model will be use, else 'state' model
            line model  - Meaningless
            state model - The character in the right of the colon
        - start: The start parameter in the print() fcuntion. One can use '\\r' to overlay the screen print / str, default ''
        - end: The end parameters in the print() function. One can use '' to continue print in the same row / str, default '\\n'
        - length: The total length of the print string / int, default 36, The total length will be length * 2 + 1
            line model  - Half of the total length of the split line
            state model - The position of the colon
    Accomplish:
        - Screen print like, '                Gcor Data : Read Complete - (3957, 8)'
    Example:
        - Screen_Print('Gcor Data','Read Complete - (3957, 8)')
    Cautions:
        - Whether word2 is None or not will decide the print mission type to 'line' or 'state' mode
            It decides whether this output is a split line or an execution message
            line model example  - '-------------xxx-------------'
            state model example - '      xxxxxxx : xxxxxxx      '
    '''
    # calculate print length and decide print model
    length_total = 2 * length + 1
    length_word1 = len(word1)
    if word2 is None:
        print_type = 'line'
    else:
        print_type = 'state'
        length_word2 = len(word2)
    # perform screen pring
    if print_type == 'line':
        if length_word1 > 0:
            length_half_word1 = length_word1 // 2
            print(start, '-' * (length - length_half_word1), word1, '-' * (length - (length_word1 - length_half_word1) + 1), sep='', end=end)
        else:
            print(start, '-' * length_total, sep='', end=end)
    elif print_type == 'state':
        print(start, ' ' * (length - 1 - len(word1)), word1, ' : ', word2, sep='', end=end)

def num_to_subscript(string, neglect_list=['1']):
    '''
    Function that converts not alpha in a string to a subscript format and returns a string that can be used in Markdown

    Parameters:
        - string: String that need to convert / str
        - neglect_list: String that will be ignore and delete from the result / array-like, defaule ['1']
    Return:
        - The new str after processing
    Example:
        - num_to_subscript('Ru3Cu2.2') -> 'Ru$_{3}$Cu$_{2.2}$'
    '''
    str_new = ''
    digit_store = ''
    for i, s in enumerate(string):
        if s.isalpha():
            if len(digit_store) == 0:
                str_new += s
            elif len(digit_store) != 0:
                str_new += '$_{' + digit_store + '}$'
                str_new += s
                digit_store = ''
        elif not s.isalpha():
            if s in neglect_list:
                pass
            else:
                digit_store += s
            if i + 1 == len(string):
                str_new += '$_{' + digit_store + '}$'
    return str_new

def phase_name_convert(expression, convert=True):
    '''
    Function that converts phase name to Markdown type.
    
    Parameters:
        - expression: str name that need to convert / str
        - convert (bool): perform convertion. Default = True
    Return:
        - The new str after processing
    Example:
        - phase_name_convert('ZnO2+2') -> '$ZnO_{2}^{2+}$'
    '''
    if not convert:
        return expression
        
    result = ""
    i = 0
    while i < len(expression):
        if expression[i].isalpha():
            result += expression[i]
            i += 1
        elif expression[i].isdigit():
            start = i
            while i < len(expression) and expression[i].isdigit():
                i += 1
            number = expression[start:i]
            result += f"_{{{number}}}"
        elif expression[i] in "+-":
            sign = expression[i]
            i += 1
            if i < len(expression) and expression[i].isdigit():
                start = i
                while i < len(expression) and expression[i].isdigit():
                    i += 1
                number = expression[start:i]
                result += f"^{{{number}{sign}}}"
            else:
                result += f"^{{{sign}}}"
        else:
            result += expression[i]
            i += 1
    return '$' + result + '$'

def drop_unname(df):
    '''
    Function to drop the 'Unnamed: 0' column in a DataFrame

    Parameter:
        - df: pd.DataFrame
    Return:
        - The DataFrame without 'Unnamed: 0' column
    '''
    return df.drop(columns='Unnamed: 0', inplace=True)
    
def three_point_angle(point1, point2, point3, angle_type='rad'):
    '''
    Function to calculate the angle between three points
    
    Parameter:
        - point1: Point 1 / array-like
        - point2: Point 2, the point at an angle / array-like
        - point3: Point 3 / array-like
        - angle_type: Type of the return value / str, 'rad' or 'deg', default 'rad'
    Return:
        - The angle in corresponding form / float
    '''
    # get point and vector
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)
    vec1 = point1 - point2
    vec2 = point3 - point2
    # calculate vector dot
    dot_product = np.dot(vec1, vec2)
    # calculate vector norm
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # calculate angle in rad
    angle_rad = np.arccos(dot_product / (norm_vec1 * norm_vec2))
    # calculate angle in degree
    angle_deg = np.degrees(angle_rad)
    if angle_type == 'rad':
        return angle_rad
    elif angle_type == 'deg':
        return angle_deg

def check_numbers_close_to_0_and_1(numbers, epsilon=1e-6):
    """
    Check if the list contains two numbers, one infinitely close to 0 and the other infinitely close to 1
    
    Parameters:
        - numbers: List to be tested, list
        - epsilon: Threshold, defined as a range of infinitely close, float, default 1e-6
    Return:
        - Bool，if it contains such two numbers return True, else False
    """
    close_to_0 = any(num < epsilon for num in numbers)
    close_to_1 = any(num > 1 - epsilon for num in numbers)
    return close_to_0 and close_to_1
