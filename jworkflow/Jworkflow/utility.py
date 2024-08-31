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


def data_distribution(data_list, bar_number=10, return_seperate=False):
    '''
    Function to draw bar and boxline graphs

    Parameters:
        - data_list: 1 dimensional data for statistical analysis / array-like
        - bar_number: The number of bar in a bar chart / int, default 10
        - return_seperate: Whether to return the split criterion of the bar graph / bool, default False
    Return:
        - Show the bar and boxline plot
        - Return the list of bar seperation value, if return_seperate is True
    '''
    # init
    data = list(data_list)
    data.sort()
    data_max = max(data) + 0.01
    data_min = min(data) - 0.01
    Fig = plt.figure(figsize=(30, 15))
    # bar graph
    ax1 = Fig.add_axes([0.1, 0.1, 0.5, 0.8])
    ax1.tick_params(length=6, width=3, labelsize=26)
    bar_count = [0] * bar_number
    bar_seperate = []
    bar_seperate_str = []
    for i in range(bar_number):
        bar_seperate_i = data_min + (data_max - data_min) * (i + 1) / bar_number
        bar_seperate.append(bar_seperate_i)
        bar_seperate_str.append(str(round(bar_seperate_i, 2)))
    for i in range(len(data)):
        for j in range(bar_number):
            if data[i] <= bar_seperate[j]:
                bar_count[j] += 1
                break
    ax1.bar(bar_seperate_str, bar_count)
    for i in range(len(bar_count)):
        ax1.text(i, bar_count[i], str(bar_count[i]), fontsize=26, ha='center', va='bottom')
    # boxline graph
    ax2 = Fig.add_axes([0.7, 0.1, 0.2, 0.8])
    ax2.tick_params(length=6, width=3, labelsize=26)
    ax2.scatter([1] * len(data), data)
    Q1 = np.percentile(data, (25))
    Q2 = np.percentile(data, (50))
    Q3 = np.percentile(data, (75))
    IQR = Q3 - Q1
    edge_upper = Q3 + 1.5 * IQR
    edge_lower = Q1 - 1.5 * IQR
    for data_statistic in [edge_lower, Q1, Q2, Q3, edge_upper]:
        ax2.plot([0.5, 1.5], [data_statistic, data_statistic])
        ax2.text(0.75, data_statistic, str(round(data_statistic, 2)), ha='center', va='bottom', fontsize=26)
    ax2.plot([0.5, 0.5], [Q1, Q3], c='k')
    ax2.plot([1.5, 1.5], [Q1, Q3], c='k')
    plt.show()
    # return seperaters
    if return_seperate:
        return bar_seperate


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


def get_file_or_subdirection(path, type='file'):
    '''
    Function that gets all subdirectories or files in a specified directory

    Parameters:
        - path: Directory where the command is executed / str(path)
        - type: Return files or subdirectories / str, 'file' or 'dir', default 'file'
    Return:
        - List of file or direction names
    '''
    for curDir, dirs, files in os.walk(path):
        if curDir == path and type == 'file':
            return files
        elif curDir == path and type == 'dir':
            return dirs


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


def drop_unname(df):
    '''
    Function to drop the 'Unnamed: 0' column in a DataFrame

    Parameter:
        - df: pd.DataFrame
    Return:
        - The DataFrame without 'Unnamed: 0' column
    '''
    return df.drop(columns='Unnamed: 0', inplace=True)


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


def zero_point_energy_from_cm_2_eV(ZPE):
    '''
    Function to convert units cm^-1 to units eV
    
    Parameter:
        - ZPE: Zero point energy in cm^-1 / float
    Return:
        - Zero point energy in eV
    '''
    return ZPE * 0.12398424290373065 / 1000 


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
