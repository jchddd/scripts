from sklearn.metrics import r2_score
from sklearn import linear_model

import pandas as pd
import numpy as np
import os

from Jworkflow.utility import screen_print, write_list_to_file
from Jworkflow.dataset import uniform_adsb, reaction_energy, adsb_to_reaction


class Catalytic_Post_Process:
    '''
    Class to read the table of calculated result data and calculate the reaction energy

    Available Functions:
        - read_excel: Function to read DataFrame of calculated results
        - filter_by_criteria: Function to ccreening structures based on screening criteria to filter out broken and negligible structures
        - calculate_energy_all: Function to calculate energy for all adss in self.df_adss
        - find_stablest_adss: Function to calculate the adsorption energy and calculate reaction energy if it has not done
        - inspecting_energy: Check if there exist some system-adsb pairs have a more lower energy compared to the located system-adsb pair
        - find_barrier: Function to find barrier and PDS
        - export_adss_names: Export specified structure names to a file that can use in the linux system
        - fit_linear_relation: Function to fit linear relation between each adsorbates
    Internal Functions:
        - get_reaction_energy
    '''

    def __init__(self):
        '''
        Attributes:
            - work_path: Path that files will be read in and write out / str(path)
            - column_adss: Column name of ads structure / str
            - column_slab: Column name of slab / str
            - column_energy: Column name of energy / str
            - column_Gcor: Column name of G correction / str
            - task_type: Calculation categories determined by the type of the input data table / int
            - df_adss: DataFrame for ads structure, change with operations / DataFrame
            - df_adss_backup: DataFrame for ads structure, will not change with operations / DataFrame
            - df_slab: DataFrame for pure slab structure calculations / DataFrame
            - df_Gcor: DataFrame for G correction from frequency calculation / DataFrame
            - df_adss_broken: DataFrame that stores broken ads structures that will be discarded / DataFrame
            - df_adss_ignore: DataFrame that stores ads structures need to be examined / DataFrame
            - df_inspect: DataFrame that stores ads structures with lower energy compared to the lowest one and need to be examined / DataFrame
            - df_path_step: DataFrame that stores the energy for each adsb-system pairs on a reaction path / DataFrame
            - df_path_adss: DataFrame that stores the ads structure names for each adsb-system pairs on a reactin path / DataFrame
            - df_lr: DataFrame that stores the linear relation information for each adsbrate pairs / DataFrame
            - print_info: Whether to print running information / bool, default True
        '''
        self.work_path = os.getcwd()
        self.column_adss = ''
        self.column_slab = ''
        self.column_energy = ''
        self.column_Gcor = ''

        self.task_type = None

        self.df_adss = None
        self.df_adss_backup = None
        self.df_slab = None
        self.df_Gcor = None

        self.df_adss_broken = None
        self.df_adss_inspect = None
        self.df_inspect = None

        self.df_path_step = None
        self.df_path_adss = None

        self.df_lr = None

        self.print_info = True

    def read_excel(self, adss_file, slab_file=None, Gcor_file=None, file_path=os.getcwd(), slab_multiply=1,
                   column_adss='ads_sys', column_slab='system', column_energy='energy', column_Gcor='G'):
        '''
        Function to read DataFrame of calculated results

        Parameters:
            - adss_file: Name of xlsx file that store data of adsorption / str
            - slab_file: Name of xlsx file that store data of slab / str
            - Gcor_file: Name of xlsx file that store of free energy correction / str
            - file_path: Path to where xlsx files are sotred / path, default os.getcwd()
            - slab_multiply: The multiple of slab energy used in the calculation / int, default 1
            - column_adss: Column name for adsorption structure / str, default 'ads_sys'
            - column_slab: Column name for slab / str, default 'system'
            - column_energy: Column name for DFT energy / str, default 'energy'
            - column_Gcor: Column name for free energy correction term / str, default 'G'
        Accomplish:
            - Store datas on self.adss.xlsx/slab.xlsx/Gcor.xlsx
        Example:
            - CPP.read_data('adss.xlsx', 'slab.xlsx', 'Gcor.xlsx', r'E:\Documents\data_store')
        Cautions:
            - To keep the program running, the first row of the excel should be numerically indexed
        '''
        # init
        self.work_path = file_path
        self.column_adss = column_adss
        self.column_slab = column_slab
        self.column_energy = column_energy
        self.column_Gcor = column_Gcor
        screen_print('Read Data') if self.print_info else None
        # read three tables to DataFrame
        # read adss table
        self.df_adss = pd.read_excel(os.path.join(file_path, adss_file), index_col=0)
        screen_print('Adss Data', 'Read Complete - ' + str(self.df_adss.shape)) if self.print_info else None
        # read slab table
        if slab_file is not None:
            self.df_slab = pd.read_excel(os.path.join(file_path, slab_file), index_col=0)
            screen_print('Slab Data', 'Read Complete - ' + str(self.df_slab.shape)) if self.print_info else None
            self.df_slab[column_energy] = self.df_slab[column_energy] * slab_multiply
        else:
            screen_print('Slab Data', 'None') if self.print_info else None
        self.df_slab[self.column_slab] = [str(slab) for slab in self.df_slab[self.column_slab]]
        # read Gcor table
        if Gcor_file is not None:
            self.df_Gcor = pd.read_excel(os.path.join(file_path, Gcor_file), index_col=0)
            screen_print('Gcor Data', 'Read Complete - ' + str(self.df_Gcor.shape)) if self.print_info else None
        else:
            screen_print('Gcor Data', 'None') if self.print_info else None
        # decide energy calculation type according to table existence
        if slab_file is None and Gcor_file is None:
            self.task_type = 0
            screen_print('Task type', 'target energy is calculated') if self.print_info else None
        elif slab_file is not None and Gcor_file is not None:
            self.task_type = 1
            screen_print('Task type', 'calculate energy and G correction') if self.print_info else None
        elif slab_file is not None and Gcor_file is None:
            self.task_type = 2
            screen_print('Task type', 'calculate energy only') if self.print_info else None
        elif slab_file is None and Gcor_file is not None:
            self.task_type = 3
            utility.Screen_Print('Task type', 'add G correction only') if self.print_info else None
        # add uniform adsb name
        self.df_adss[column_slab] = [adss.split('_')[0] for adss in list(self.df_adss[column_adss])]
        self.df_adss['adsb'] = [adss.split('_')[1] for adss in list(self.df_adss[column_adss])]
        self.df_adss['uf_adsb'] = [uniform_adsb(adsb) for adsb in self.df_adss['adsb'].values]
        self.df_adss_backup = self.df_adss.copy(deep=True)
        # end
        screen_print('Read Compelete') if self.print_info else None

    def filter_by_criteria(self, criteria_broken="(df['mdista_adsb'] > 0.6) | (df['mupgrade_slab'] > 0.6) | (df['mupgrade_skel'] > 0.9)",
                           criteria_inspect="(df['converg'] == False) | (df['mshift_slab'] > 0.6)"):
        '''
        Function to ccreening structures based on screening criteria to filter out broken and negligible structures

        Paremeters:
            - criteria_broken: Criteria uses to filter broken structures with severe structural reconfiguration and should discard from the study
                / str, default "(df['mdista_adsb'] < 0.6) | (df['mupgrade_slab'] < 0.6) | (df['mupgrade_skel'] < 0.9)"
            - criteria_inspect: Criteria uses to filter structures that are questionable and manual inspection is recommended
                / str, default "(df['converg'] == True) | (df['mshift_slab'] < 0.6)"
        Accomplish:
            - Screen for broken structures from .df_adss_backup, update or generate .(df_adss/df_adss_broken/df_adss_inspect)
        Example:
            - CPP.filter_by_criteria("(df['mshift_skel'] > 0.2) & (df['mupgrade_skel'] > 0.3)")
        Cautions:
            - Screened structures that need to be checked are not directly applied in the subsequent reaction path calculation
            - The filtering criteria should be written strictly according to the above format, use '&' and '|' for 'and' and 'or'
            - The criteria should be constructed based on the column names in the adss.xlsx file
            - This function will update the self.df_adss DataFrame to ensure the broken structures no longer appear in the subsequent calculation
            - Use empty string "" to ignore an filter
        '''
        if self.print_info:
            screen_print('Filter by Criteria')
            screen_print('Data total ', str(self.df_adss.shape[0]))
        # find unbroken mission according to criteria_broken, update self.df_adss_broken and self.df_adss
        if len(criteria_broken) > 0:
            df = self.df_adss_backup
            bools_broken = eval(criteria_broken)
            bools_unbroken = [not b for b in bools_broken]
            self.df_adss_broken = df[bools_broken]
            self.df_adss = df[bools_unbroken]
        # find structures that need inspecttion according to criteria_inspect, update self.df_inspect and self.df_adss
        if len(criteria_inspect) > 0:
            df = self.df_adss
            bools_inspect = eval(criteria_inspect)
            bools_uninspect = [not b for b in bools_inspect]
            self.df_adss_inspect = df[bools_inspect]
            self.df_adss = df[bools_uninspect]
        # reset index
        self.df_adss.reset_index(inplace=True, drop=True)
        if self.df_adss_inspect is not None:
            self.df_adss_inspect.reset_index(inplace=True, drop=True)
        if self.df_adss_broken is not None:
            self.df_adss_broken.reset_index(inplace=True, drop=True)
        # print info
        if self.print_info:
            screen_print('Data broken', str(self.df_adss_broken.shape[0]) if self.df_adss_broken is not None else str('0'))
            screen_print('Data inspect', str(self.df_adss_inspect.shape[0]) if self.df_adss_inspect is not None else str('0'))
            screen_print('Data remain', str(self.df_adss.shape[0]))
            screen_print('Filter Finish')

    def find_stablest_adss(self, adsbs, reactions=['NRR', 'HER'], U=0, PH=0, dict_e_upd={}, dict_G_upd={}):
        '''
        Function to calculate the adsorption energy and calculate reaction energy if it has not done

        Parameters:
            - adsbs: List of adsorbents for which adsorption energy needs to be calculated / list
            - reactions: Type of the reactions / list, default ['NRR', 'HER']
            - U: Applied U in the reaction energy calculation / float, default 0
            - PH: PH correction in the reaction energy calculation / float, default 0
            - dict_e_upd: Additional update dict of energy / dict, default {}
            - dict_G_upd: Additional update dict of G correction / dict, default {}
        Accomplish:
            - Find the lowest energy for each system-adsb pair from .df_adss and calculate energy according to task type
            - Generate self.df_path_adss/df_path_step to store structures and erergies for each system-adsb pair
        Example:
            - CPP.cal_ads_energy(['N2', 'NNH', 'NH', 'NH2','H'], ['NRR','HER'])
        Cautions:
            - If the free energy correction for the structure corresponding to the lowest energy does not exist, its calculation is ignored
            - dict_e_upd, dict_G_upd are designed to add energy data for some general molecules such as gase state nitrogen
            - The energy calculation formula of adsorbent will be found according to the order of reaction in reactions
        '''
        screen_print('Calculate Ads Energy') if self.print_info else None
        # creat empty DataFrames
        systems = list(set(list(self.df_adss[self.column_slab])))
        systems.sort()
        self.df_path_adss = pd.DataFrame(index=systems, columns=adsbs)
        self.df_path_step = self.df_path_adss.copy(deep=True)
        # find the lowest energy adss for each adsb-system pair from df_adss
        df = self.df_adss
        for (sys, adsb), df_group in df.groupby([self.column_slab, 'uf_adsb']):
            Emin = min(df_group[self.column_energy])
            self.df_path_step.at[sys, adsb] = Emin
            Emin_adss = list(df_group.loc[(df_group[self.column_energy] == Emin)][self.column_adss])[0]
            self.df_path_adss.at[sys, adsb] = Emin_adss
        # calculate reaction energy according to task type
        nan_count = 0
        if self.task_type != 0:
            for sys in self.df_path_step.index:
                for adsb in self.df_path_step.columns:
                    if not np.isnan(self.df_path_step.at[sys, adsb]):
                        adss = self.df_path_adss.at[sys, adsb]
                        energy = self.df_path_step.at[sys, adsb]
                        self.df_path_step.at[sys, adsb] = self.get_reaction_energy(adss, energy, dict_e_upd, dict_G_upd, reactions, U, PH)
                    else:
                        nan_count += 1
        # print_info
        if self.print_info:
            screen_print('Adsb_list', adsbs)
            screen_print('Reaction_types', reactions)
            screen_print('System number', str(len(systems)))
            screen_print('Pair number', str(len(systems) * len(adsbs)))
            screen_print('Nan pair count', str(nan_count))
            screen_print('Calculated count', str(len(systems) * len(adsbs) - nan_count))
            screen_print('Calculation Compelete')

    def get_reaction_energy(self, adss, energy, dict_e_upd, dict_G_upd, reactions, U, PH):
        # init
        sys = adss.split('_')[0]
        adsb = uniform_adsb(adss.split('_')[1])
        # confirm reaction
        reaction_may = adsb_to_reaction(adsb)
        for r in reactions:
            if r in reaction_may:
                reaction = r
                break
        # creat energy dict {'*':xxx, '*adsb':yyy} + dict_e_upd
        dict_energy = {'*': list(self.df_slab.loc[(self.df_slab[self.column_slab] == sys)][self.column_energy])[0], '*' + adsb: energy}
        dict_energy.update(dict_e_upd)
        # calculate reaction energy
        # task type 2 calculate energy only
        if self.task_type == 2:
            result = reaction_energy(reaction, adsb, dict_energy, {}, U, PH)
        # check if Gcor exist for G calculation and creat dict_Gcor
        elif self.task_type == 1 or self.task_type == 3:
            assert adss in self.df_Gcor[self.column_adss].values, 'Gcor of ' + adss + ' not found !'
            dict_Gcor = {'*' + adsb: list(self.df_Gcor.loc[(self.df_Gcor[self.column_adss] == adss)][self.column_Gcor])[0]}
            dict_Gcor.update(dict_G_upd)
            # task type 1 calculate energy and G
            if self.task_type == 1:
                result = reaction_energy(reaction, adsb, dict_energy, dict_Gcor, U, PH)
            # task type 3 add Gcor only
            elif self.task_type == 3:
                result = reaction_energy(reaction, adsb, {}, dict_Gcor, U, PH)
        # return
        return result

    def calculate_energy_all(self, reactions=['NRR', 'HER'], U=0, PH=0, dict_e_upd={}, dict_G_upd={}):
        '''
        Function to calculate energy for all adss in self.df_adss

        Parameters:
            - reactions: Type of the reactions / list, default ['NRR', 'HER']
            - U: Applied U in the reaction energy calculation / float, default 0
            - PH: PH correction in the reaction energy calculation / float, default 0
            - dict_e_upd: Additional update dict of energy / dict, default {}
            - dict_G_upd: Additional update dict of G correction / dict, default {}
        Accomplish:
            - Calculate reaction energy and update the column_energy on df_adss
            - This will also set task_type to 0 to avoid additional energy calculation
        Example:
            - CPP.calculate_energy_all(['NRR', 'HER'])
        Cautiou:
            - Adss not in df_Gcor will be drop from df_adss if free energy correction is calculated
        '''
        # init
        reaction_energy = []
        count_adss = self.df_adss.shape[0]
        # calculate energy
        if   self.task_type == 1 or self.task_type == 3:
            for i, adss in enumerate(self.df_adss[self.column_adss].values):
                if adss in self.df_Gcor[self.column_adss].values:
                    energy = self.df_adss.at[i, self.column_energy]
                    reaction_energy.append(self.get_reaction_energy(adss, energy, dict_e_upd, dict_G_upd, reactions, U, PH))
                else:
                    self.df_adss.drop(index=i, inplace=True)
            self.df_adss[self.column_energy] = reaction_energy
            self.df_adss.reset_index(drop=True, inplace=True)
        elif self.task_type == 2:
            for i, adss in enumerate(self.df_adss[self.column_adss].values):
                energy = self.df_adss.at[i, self.column_energy]
                reaction_energy.append(self.get_reaction_energy(adss, energy, dict_e_upd, dict_G_upd, reactions, U, PH))
            self.df_adss[self.column_energy] = reaction_energy
        # change task type
        self.task_type = 0
        if self.print_info:
            screen_print('Calculate Energy All')
            screen_print('Calculation', 'Complete')
            screen_print('Calculated count', str(self.df_adss.shape[0]))
            screen_print('Ignored count',str(count_adss - self.df_adss.shape[0]))
            screen_print('END')

    def inspecting_energy(self):
        '''
        Check if the df_path_inspect exist some system-adsb pairs have a more lower energy compared to the located system-adsb pair

        Accomplish:
            - Find the structures need manual inspection with lower energy, creat and update self.df_inspect
        Example:
            - CPP.inspecting_energy()
        Cautions:
            - This function need to perform filter_by_criteria and find_stablest_adss to generate self.df_path_step at first
            - The system-adsb pairs this function inspected is according to the input of cal_ads_energy
        '''
        # init
        assert self.df_adss_inspect is not None and self.df_path_step is not None, 'Find empty preposition DataFrame !'
        screen_print('Inspection Energy') if self.print_info else None
        tabulation = False
        systems = self.df_path_step.index
        adsbs = self.df_path_step.columns
        self.df_inspect = pd.DataFrame(index=systems, columns=adsbs)
        # loop adss on df_adss_inspect, find its energy and the energy of the lowest one store in self.df_path_step
        for i in self.df_adss_inspect.index:
            system = self.df_adss_inspect.at[i, self.column_slab]
            adsb = self.df_adss_inspect.at[i, 'uf_adsb']
            E_inspect = self.df_adss_inspect.at[i, self.column_energy]
            adss_step = self.df_path_adss.at[system, adsb]
            if type(adss_step) == str:
                E_step = self.df_adss.loc[(self.df_adss[self.column_adss] == adss_step)][self.column_energy].values[0]
            else:
                E_step = 0
            # compare energies, update df_inspect
            if E_inspect < E_step:
                E_lower = E_inspect - E_step
                if not tabulation:
                    screen_print('Inspect adss', '   E_inspect       E_lower') if self.print_info else None
                    tabulation = True
                if type(self.df_inspect.at[system, adsb]) != dict:
                    self.df_inspect.at[system, adsb] = {'adss': [self.df_adss_inspect.at[i, self.column_adss]], 'E': [E_inspect], 'DeltaE': [E_lower]}
                else:
                    self.df_inspect.at[system, adsb]['adss'].append(self.df_adss_inspect.at[i, self.column_adss])
                    self.df_inspect.at[system, adsb]['E'].append(E_inspect)
                    self.df_inspect.at[system, adsb]['DeltaE'].append(E_lower)
                screen_print(self.df_adss_inspect.at[i, self.column_adss], str(E_inspect) + ' ' + str(E_lower)) if self.print_info else None
        # end
        if not tabulation:
            screen_print('Structures need to be examined', 'Not found')
        screen_print('Inspection Done') if self.print_info else None

    def export_adss_names(self, export_type='inspect', file_name='Str_list', file_type='Linux'):
        '''
        Export specified structure names to a file that can use in the linux system

        Parameters:
            - export_type: Which structures to export / str in 'inspect' or 'Emin' default 'inspect'
            - file_name: The name of the file that stored the export structure names / str, default 'Str_list'
            - file_type: The system which the file is used at / str, default 'Linux', or 'Windows'
        Accomplish:
            - Write a file to the work path with structure names on it
        '''
        # write adss names
        adss_names = []
        # write inspect structure
        if export_type == 'inspect':
            for sys in self.df_path_adss.index:
                for adsb in self.df_path_adss.columns:
                    if not self.df_inspect.isnull().at[sys, adsb]:
                        for i in range(len(self.df_inspect.at[sys, adsb]['adss'])):
                            adss_names.append(self.df_inspect.at[sys, adsb]['adss'][i])
        # write stablest structure
        elif export_type == 'Emin':
            for sys in self.df_path_adss.index:
                for adsb in self.df_path_adss.columns:
                    if not self.df_path_adss.isnull().at[sys, adsb]:
                        adss_names.append(self.df_path_adss.at[sys, adsb])
        write_list_to_file(adss_names, os.path.join(self.work_path, file_name), file_type)
        # print info
        if self.print_info:
            screen_print('Export adss')
            screen_print('Save file', os.path.join(self.work_path, file_name))
            screen_print('END')

    def export_df(self, df, file_name):
        '''
        Function to save DataFrame to work path

        Parameters:
            - df: Select the DataFrame to save / pd.DataFrame
            - file_name: Excel name / str, like 'xxx.xlsx'
        Accomplish:
            - Write a excel file to the work path
        '''
        df.to_excel(os.path.join(self.work_path, file_name))
        if self.print_info:
            screen_print('Export DataFrame')
            screen_print('Save excel', os.path.join(self.work_path, file_name))
            screen_print('END')

    def find_barrier(self, reaction_path):
        '''
        Function to find barrier and PDS

        Parameter:
            - reaction_path: List that describe the reaction path in 3 dimentions, section, piece, adsb / (x, y, z) list
        Acomplish:
            - Find barrier and PDS for each system, add 'barrier' and 'PDS' column to df_path_step
        Example:
            - CPP.find_barrier([[['N2', 'NNH']], [['NNH', 'NHNH',...,'NH2'], ['NNH', 'NNH2',...,'NH2']], [['NH2', 'NH3']]])
        '''
        # init
        screen_print('Find_Barrier_PDS') if self.print_info else None
        df = self.df_path_step
        barriers = []
        PDSs = []
        none_barrier_count = 0
        # loop for system
        for sys in df.index:
            sys_barriers = []
            sys_PDSs = []
            for section in reaction_path:
                section_barriers = []
                section_PDSs = []
                for piece in section:
                    piece_barriers = []
                    piece_PDSs = []
                    for adsb_index in range(len(piece) - 1):
                        piece_barriers.append(df.at[sys, piece[adsb_index + 1]] - df.at[sys, piece[adsb_index]])
                        piece_PDSs.append(piece[adsb_index] + '->' + piece[adsb_index + 1])
                    piece_barrier = max(piece_barriers)
                    section_barriers.append(piece_barrier)
                    section_PDSs.append(piece_PDSs[piece_barriers.index(piece_barrier)])
                section_barrier = min(section_barriers)
                sys_barriers.append(section_barrier)
                sys_PDSs.append(section_PDSs[section_barriers.index(section_barrier)])
            sys_barrier = max(sys_barriers)
            if sys_barrier > 0:
                barriers.append(sys_barrier)
                PDSs.append(sys_PDSs[sys_barriers.index(sys_barrier)])
            elif sys_barrier < 0:
                barriers.append(0)
                PDSs.append(None)
                none_barrier_count += 1
            else:
                barriers.append(None)
                PDSs.append(None)
                none_barrier_count += 1
        # collect barriers and make statistical analysis
        df['barrier'] = barriers
        df['PDS'] = PDSs
        PDS_unique = list(set(PDSs))
        PDS_count = [0] * len(PDS_unique)
        for PDS in PDSs:
            PDS_count[PDS_unique.index(PDS)] += 1
        PDS_count_str = ''
        for PDS in PDS_unique:
            if PDS is not None:
                PDS_count_str = PDS_count_str + PDS + ': ' + str(PDS_count[PDS_unique.index(PDS)]) + ', '
            else:
                PDS_count_str = PDS_count_str + 'None: ' + str(PDS_count[PDS_unique.index(PDS)]) + ', ' 
        # print info
        if self.print_info:
            barriers = [barrier if barrier is not None else np.nan for barrier in barriers]
            PDSs = [PDS if PDS is not None else 'None' for PDS in PDSs]
            screen_print('Max barrier', df.index[barriers.index(max(barriers))] + ' - ' + PDSs[barriers.index(max(barriers))] + ' - ' + str(max(barriers)))
            screen_print('Min barrier', df.index[barriers.index(min(barriers))] + ' - ' + PDSs[barriers.index(min(barriers))] + ' - ' + str(min(barriers)))
            screen_print('PDS count', PDS_count_str[:-2])
            screen_print('Zreo barrier count', str(none_barrier_count))
            screen_print('END')
            
    def add_prod(self, reat, prod_name='prod', Gcor=True):
        '''
        Function to add product energy to the df_path_step
        
        Parameters:
            - reat: The reaction / str, default None
            - prod_name: Name of the product column / str, default 'prod'
            - Gcor: Whether or not to add free energy correction / bool, default True
        '''
        corG = {'*': 1} if Gcor else {}
        self.df_path_step[prod_name] = reaction_energy(reat, 'prod', {'*': 1}, corG)
        if self.print_info:
            screen_print('Add Prod')
            screen_print(prod_name, str(reaction_energy(reat, 'prod', {'*': 1}, corG)))
            screen_print('END')

    def fit_linear_relation(self, adsb_list=[], exclude_list=[]):
        '''
        Function to fit linear relation between each adsorbates

        Parameters:
            - adsb_list: Adsorbate list that will be fititted / (n) list, default all adsorbates in self.df_path_setp
            - exclude_list: Slab system list that exclude from the fitting / (n) list, default []
        Acomplish:
            - Fit linear relation for each adsb pair, store the k, b and R2 values to df_lr
        '''
        # init
        screen_print('Fit_Linear_Relation') if self.print_info else None
        R2s = []
        XY_names = []
        sys_list = [s for s in self.df_path_step.index if s not in exclude_list]
        df = self.df_path_step.copy(deep=True)
        df['system'] = df.index
        df = df[df['system'].isin(sys_list)]
        if len(adsb_list) == 0:
            adsb_list = self.df_path_step.columns
            adsb_list = [adsb for adsb in adsb_list if adsb not in ['barrier', 'PDS']]
        else:
            adsb_list = adsb_list
        # fit linear relation
        self.df_lr = pd.DataFrame(columns=adsb_list, index=adsb_list)
        for adsb_x in adsb_list:
            for adsb_y in adsb_list:
                x = np.array(df[adsb_x]).reshape(-1, 1)
                y = np.array(df[adsb_y])
                lm = linear_model.LinearRegression()
                lm.fit(x, y)
                y_predictd = lm.predict(x)
                R2 = r2_score(y, y_predictd)
                k = lm.coef_
                b = lm.intercept_
                self.df_lr.at[adsb_x, adsb_y] = {'x': adsb_x, 'y': adsb_y, 'k': k[0], 'b': b, 'R2': R2}
                if adsb_x != adsb_y:
                    R2s.append(R2)
                    XY_names.append(adsb_x + '_' + adsb_y)
        # print info
        relation_number = len(adsb_list) ** 2 - len(adsb_list)
        R2s_g9 = [r for r in R2s if r >= 0.9]
        if self.print_info:
            screen_print('Relation number', str(relation_number))
            screen_print('Max R\u00b2 score', XY_names[R2s.index(max(R2s))] + ' - ' + str(max(R2s)))
            screen_print('Min R\u00b2 score', XY_names[R2s.index(min(R2s))] + ' - ' + str(min(R2s)))
            screen_print('Percentage of R\u00b2 > 0.9', ' ' + str(len(R2s_g9)) + ' / ' + str(relation_number))
            screen_print('END')
