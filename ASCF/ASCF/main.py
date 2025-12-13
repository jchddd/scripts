from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
import sklearn.decomposition as decomposition
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn import cluster
from collections import Counter
from ase.symbols import atomic_numbers, chemical_symbols
from ase.io import read, write
from ase.db import connect
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure, Element
from dscribe.descriptors import SOAP
import matplotlib.pyplot as plt
import numpy as np
import chemiscope
import fpsample
import os


class ASCF():
    def __init__(self):
        self.centers = None
        self.delete_ele = False
        self.delete_ele_cs = False

    def load_database(self, db):
        '''
        Load ase database 
        
        Parameter:
            - db (path in str): file path to the ase database.
        '''
        self.database = connect(db)

    def delete_elements(self, eles, dele_when_show=False):
        '''
        Delete selected elements when describe_atom_environment

        Parameter:
            - eles (list): List of deleted elements
            - dele_when_show (bool): whether also deleted elements when prepare_chemiscope_input. Default = False
        '''
        self.delete_ele = True
        self.delete_ele_cs = dele_when_show
        self.delete_eles = eles

    def idntify_centers(self, center_type='all', adsb_element=['N', 'H'], atom_height=None, para_infer_site={}, site_height=0.):
        '''
        Function to idntify atomic centers for calculation of descriptor
        
        Parameters:
            - center_type (str): type of center atoms, in 'adsb', 'site', 'layer', 'all'. Default = 'all'
              'adsb' will selected adsorbed atomic centers according to 'adsb_element' and 'atom_height'.
              'site' will use infer_adsorption_site to selected site aomic centers then find site with 'site_height'. 
              'layer' will use selected atoms that satisfy height criterion set by 'atom_height'
              'all' will use all atoms in the structure as centers.
            - adsb_element: (list): list of adsorbate elements. Default = ['N', 'H']
            - atom_height: (list or None): a (2,) list of a height criterion to assist the determination of adsorbed atoms. Default = None
              Passing the criterion and the height corresponding to highest not select atoms, like ['height', 4.6], ['layer', 4]. None for not height criterion.
            - para_infer_site (dict): parameters for infer_adsorption_site except 'slab', and 'adsb'. Default = {}
            - site_height (float): distance between site center and surface. Default = 0
        '''
        self.centers = []
        
        if   center_type == 'adsb':
            for row in self.database.select():
                atoms = row.toatoms()
                height_judge = self._atoms_height_judge(atoms, atom_height)
                symbol_judge = self._atoms_symbol_judge(atoms, adsb_element)
                is_adsb = []
                for i_atom in range(len(atoms)):
                    if height_judge[i_atom] and symbol_judge[i_atom]:
                        is_adsb.append(i_atom)
                self.centers.append(is_adsb)

        elif center_type == 'layer':
            for row in self.database.select():
                atoms = row.toatoms()
                height_judge = self._atoms_height_judge(atoms, atom_height)
                is_above = []
                for i_atom in range(len(atoms)):
                    if height_judge[i_atom]:
                        is_above.append(i_atom)
                self.centers.append(is_above)
                
        elif center_type == 'site':
            for row in self.database.select():
                atoms = row.toatoms()
                para_infer_site['adsb'] = adsb_element
                site, corr, cont = infer_adsorption_site(AseAtomsAdaptor.get_structure(atoms), **para_infer_site)
                site_center_cart = np.mean(atoms.positions[cont[1]], axis=0)
                self.centers.append(site_center_cart[np.newaxis, :] + [[0, 0, site_height]])

        elif center_type == 'adss':
            for row in self.database.select():
                atoms = row.toatoms()
                height_judge = self._atoms_height_judge(atoms, atom_height)
                symbol_judge = self._atoms_symbol_judge(atoms, adsb_element)
                centers = []
                for i_atom in range(len(atoms)):
                    if height_judge[i_atom] and symbol_judge[i_atom]:
                        centers.append(i_atom)
                site, corr, cont = infer_adsorption_site(AseAtomsAdaptor.get_structure(atoms), **para_infer_site)
                for i_atom in cont[1]:
                    if i_atom not in centers:
                        centers.append(i_atom)
                self.centers.append(centers)

        elif center_type == 'all':
            self.centers = None

    def describe_atom_environment(self, env_dsc='soap', para_dscrib={}, n_jobs=6):
        '''
        Function to calculate atomic environment descriptors.
        
        Parameters:
            - env_dsc (str): the descriptor in 'soap'. Default = 'soap'
            - para_dscrib (dict): parameters for the descriptor, check dscribe document. Default = {}
              If descriptors are specific to atoms, be sure to use functions such as averaging to obtain the structure descriptors
            - n_jobs (int): the number of parallel jobs to run.. Default = 6
        '''
        if   env_dsc == 'soap':
            species = self._get_db_symbols(self.delete_ele)
            para_for_soap = {
                    'r_cut'      : 5.0,
                    'n_max'      : 6,
                    'l_max'      : 4,
                    'sigma'      : 1,
                    'rbf'        : 'gto',
                    'weighting'  : {'function':'pow','m':8,'r0':2.82,'c':1,'d':1},
                    'average'    : 'outer',
                    'compression': {'mode': 'mu1nu1', 'species_weighting': None},
                    'species'    : species,
                    'periodic'   : True}
            para_for_soap.update(para_dscrib)
            descriptor = SOAP(**para_for_soap)

        self.atom_envir_describ = descriptor.create(self._get_atoms_list(self.delete_ele), self.centers, n_jobs)

    def perform_dimension_reduction(self, reduct_seq=['PCA'], dimension=[2], para_PCA={}, para_tSNE={}, use_pre_models=False, n_jobs=6):
        '''
        Function to perform dimension reduction of atomic environment descriptor
        
        Parameters:
            - reduct_seq (list): the name of the dimensionality reduction methods used sequentially on features. Default = ['PCA']
              Data with more dimensions can be reduced multiple times to ensure effectiveness. Choose from PCA, KernelPCA, and tSNE.
            - para_PCA (dict): parameters for PCA. check scikit-learn document. Default = {}
            - para_tSNE (dict): parameters for tSNE. check scikit-learn document. Default = {}
            - use_pre_models (bool): use pre-trained models rather than fit new models, only support PCA type models. Default = False
            - n_jobs (int): the number of parallel jobs to run.. Default = 6
        '''
        x = self.atom_envir_describ
        if   not use_pre_models:
            models = []
            for i, reductioin in enumerate(reduct_seq):
                if  'PCA' in reductioin:
                    para = {'n_components': dimension[i]}
                    para.update({'n_jobs': n_jobs}) if reductioin != 'PCA' else None
                    para.update(para_PCA)
                    model = getattr(decomposition, reductioin)(**para)
                elif 'tSNE' in reductioin:
                    para = {'n_components': dimension[i], 'n_jobs': n_jobs}
                    para.update(para_tSNE)
                    model = TSNE(**para)
                x = model.fit_transform(x)
                models.append(model)
            self.models = models
        elif use_pre_models:
            for model in self.models:
                x = model.transform(x)
        self.low_dimensional_describe = x
            
    def prepare_chemiscope_input(self, axis_name='PCA', title='Geometric feature projection', color='', size='', symbol='', palette='cividis',
                                 select_show='all', show_group=None, show_group_by='symbol'):
        '''
        Function to prepare a chemiscope input dict use the low_dimensional_describe generate by perform_dimension_reduction
        
        Parameters:
            - axis_name (str): axis label name. Default = 'PCA'
            - title (str): title of the map. Default = 'Geometric feature projection'
            - color (str): color properity key name in database. Default = ''
            - size (str): size property key name in database. Default = ''
            - symbol (str): symbol property key name in database. Default = ''
            - palette (str): color palette to use ('bwr', 'cividis', 'inferno', 'plasma'). Default = 'cividis'
            - select_show (str): only show one part of samples. 'all', 'sample', 'unsample', 'normal', 'abnormal'. Default = 'all'
            - show_group (str or None): show group ('sample' or 'abnormal') of points. Default = None
            - show_group_by (str): show selected sample point by 'color' or 'symbol'. Default = 'symbol'  
        '''
        ids = list(np.array(self._get_properties_from_db('id')) - 1)
        if   select_show == 'all':
            select_ids = None
        elif select_show == 'sample':
            select_ids = self.sample_index
        elif select_show == 'unsample':
            select_ids = [i for i in ids if i not in self.sample_index ]
        elif select_show == 'abnormal':
            select_ids = self.abnormal_idx
        elif select_show == 'normal':
            select_ids = [i for i in ids if i not in self.abnormal_idx]
            
        frames = self._get_atoms_list(self.delete_ele_cs, select_ids)

        properties = {}
        if len(color) > 0: 
            properties.update({color: self._get_properties_from_db(color, select_ids)})
        if len(size) > 0:
            properties.update({size: self._get_properties_from_db(size, select_ids)})
        if len(symbol) > 0:
            properties.update({symbol: self._get_properties_from_db(symbol, select_ids)})
            
        if select_ids is None:
            select_ids = ids
        properties.update({axis_name: self.low_dimensional_describe[select_ids]})
            
        meta=dict(name=title)
        
        if show_group is not None:
            group_attribute = {'sample': 'sample_index', 'abnormal': 'abnormal_idx'}[show_group]
            group_label = {'sample': 'sampled', 'abnormal': 'abnormal'}[show_group]
            if hasattr(self, group_attribute):
                if   show_group_by == 'symbol':
                    properties = self._show_diff_group(properties, getattr(self, group_attribute), 'showgroup', show_group_by, group_label)
                    symbol = 'showgroup'
                elif show_group_by == 'color':
                    properties = self._show_diff_group(properties, getattr(self, group_attribute), 'showgroup', show_group_by)
                    color  = 'showgroup'
            
        settings=chemiscope.quick_settings(x=axis_name + '[1]', y=axis_name + '[2]', 
                                           z=axis_name + '[3]' if self.low_dimensional_describe.shape[1] == 3 else '', 
                                           color=color, size=size, symbol=symbol,
                                           map_settings={'palette': palette})

        self.chemiscope_input = {'frames': frames, 'properties':properties, 'meta': meta, 'settings': settings}

    def farthest_point_sampling(self, n_samples, other_para={'h': 7, 'start_idx': None}):
        '''
        Function to perform faster point sampling

        Parameters:
            - n_samples (int): number of sample to pick
            - other_pare (dict): other parameters for fpsample.bucket_fps_kdline_sampling. Default = {'h': 7, 'start_idx': None}
        '''
        sample_index = fpsample.bucket_fps_kdline_sampling(self.low_dimensional_describe, n_samples=n_samples, **other_para)
        self.sample_index = sample_index

    def stratified_sampling(self, n_clusters, k_per_cluster=1, cluster_method='KMeans', cluster_para={'random_state': None}):
        '''
        Function to perform stratified sampling

        Parameters:
            - n_clusters (int): number of clusters for cluster 
            - k_per_cluster (int): sample k points from each cluster. Default = 1
            - cluster_method (str): cluster models from sklearn.cluster. Default = 'KMeans'
            - cluster_para (dict): parameters for cluster model. Default = {'random_state': None}
        '''
        cluster_para.update({'n_clusters': n_clusters})
        cluster_model = getattr(cluster, cluster_method)(**cluster_para).fit(self.low_dimensional_describe)
        all_indices = []
        for i in range(n_clusters):
            indices = np.where(cluster_model.labels_ == i)[0]
            if indices.size < k_per_cluster:
                raise IndexError("Not enough unique indices for sampling, please decrease k_per_cluster or n_clusters")
            indices = np.random.choice(indices, size=k_per_cluster, replace=False)
            all_indices.append(indices)
        self.sample_index = np.concatenate(all_indices)
        
    def write_files(self, write_path, files='all', file_format='vasp', file_suffix=None, file_names=None):
        '''
        write sampled structures

        Parameters:
            - write_path (path): path to store outputs
            - files (str): files to write, 'all', 'sample' or 'normal'. Default = 'all'
            - file_format (str): write file format (file_format on ase.os.write). Default = vasp
            - file_suffix (str or None): whether add suffix to file name. Default = None
            - file_names ((len(db), ) list or None): list of file names. If None, use row.struc_name (use row.id if struc_name not exist). Default = None
        '''
        has_struc_name = 'struc_name' in self.database[1].key_value_pairs
        for i, row in enumerate(self.database.select()):
            # determine whether write this structure
            write_file = False
            if   files == 'all':
                write_file = True
            elif files == 'sample':
                if i in self.sample_index: write_file = True
            elif files == 'normal':
                if i not in self.abnormal_idx: write_file = True
            # get file name and write file
            if write_file:
                if   file_names is not None:
                    file_name = file_names[i]
                elif has_struc_name:
                    file_name = row.struc_name
                else:
                    file_name = str(row.id)
                file_name = file_name + file_suffix if file_suffix is not None else file_name
                write(os.path.join(write_path, file_name), row.toatoms(), file_format)

    def k_nearst_dist(self, k=5, metric='euclidean'):
        '''
        Get the mean and std of similarity (distance) upto k nearest
    
        Parameters:
            - k (int)
            - metric (str): metric to use when calculating distance between instances. Default = 'euclidean'
              Select from ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        '''
        print(f' k |    mean    |     std    |')
        dist_simi_matrix = pairwise_distances(self.low_dimensional_describe, metric=metric)
        dist_simi_matrix = np.sort(dist_simi_matrix)
        for ki in range(k):
            mean = np.mean(dist_simi_matrix[:, ki + 1])
            std  = np.std(dist_simi_matrix[:, ki + 1])
            print(f'{ki + 1:2d} | {mean:10.3f} | {std:10.3f} |')

    def show_similarity_vs_targetdiff(self, target, threshold_neibor=3, metric='euclidean', 
                                      para_scatter={'edgecolors':'k', 'linewidths':0.3, 's': 10, 'c': 'skyblue', 'alpha': 0.7}, 
                                      show_line=None, para_line={'color': 'lime'},
                                      suggest_abnormal=False, n_bins=10, method='iqr', threshold=2):
        '''
        Find neighbor pairs and show their similarity (distance) vs target difference
    
        Parameters:
            - target (str): target property name
            - threshold_neibor (int or float): threshold for neighbor determination. Default = 3
            - metric (str): metric to use when calculating distance between instances. Default = 'euclidean'
              Select from ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
            - para_scatter (dict): parameters for plt.scatter, like c, s, marker, alpha, edgecolors, linewidths
              Default, {'edgecolors':'k', 'linewidths':0.3, 's': 10, 'c': 'skyblue', 'alpha': 0.7}
            - show_line (None, int, float, list or tuple): draw a line to assist in the determination of the abnormal criterion.
              if int and float, line is y = show_line, if list or tuple, line is y = show_line[0] * x + show_line[1]
            - para_line (dict): parameters for plt.plot, like linestyle, linewidth, color. Default, {'color': 'lime'}
            - suggest_abnormal (bool): recommend a straight line for determining abnormal samples based on statistical methods. Default = False
            - n_bins (int): number of bins when suggest abnormal. Default = 10
            - method (str): statistical method for abnormal suggestion, 'std', 'iqr' or 'mad'. Default = 'iqr'
            - threshold (float): threshold for abnormal suggestion. e.g. on 'std', abnormal line is mean + threshold * std. Default = 2
        Return:
            - the figure
        Caution:
            - Abnormal points will not be taking into account
        '''
        # calculate similarity and target difference
        targets = np.array(self._get_properties_from_db(target))
        dist_simi_matrix = pairwise_distances(self.low_dimensional_describe, metric=metric)
        num_point = dist_simi_matrix.shape[0]
        similarities = []
        target_diffs = []
        for i in range(num_point):
            for j in range(num_point - i - 1):
                j = j + i + 1
                if dist_simi_matrix[i][j] < threshold_neibor:
                    consider_pair = True
                    if hasattr(self, 'abnormal_idx'):
                        if any([i in self.abnormal_idx, j in self.abnormal_idx]):
                            consider_pair = False
                    if consider_pair:
                        similarities.append(dist_simi_matrix[i][j])
                        target_diffs.append(abs(targets[i] - targets[j]))
        # suggest abnormal threshold
        if suggest_abnormal:
            xc, yu, yc, yl, cu, iu, cl, il = analyze_point_distribution(np.array(similarities), np.array(target_diffs), n_bins, method, threshold)
            print(f'suggset abnormal threshold: coef {cu} intercept {iu}')
        # perform plotting 
        fig = plt.figure()
        plt.scatter(similarities, target_diffs, **para_scatter, label='simil vs tadif')
        # add label
        plt.title('Similarity vs Target Value Difference', fontsize=14)
        plt.xlabel('Similarity', fontsize=12)
        plt.ylabel('Target Value Difference', fontsize=12)
        axes = plt.gca()
        plt.xlim(*axes.get_xlim())
        plt.ylim(*axes.get_ylim())
        if show_line is not None:
            xx = np.array(axes.get_xlim())
            if   isinstance(show_line, int) or isinstance(show_line, float):
                plt.plot(xx, [show_line] * 2, label='show line', **para_line)
            elif isinstance(show_line, list) or isinstance(show_line, tuple):
                plt.plot(xx, show_line[0] * xx + show_line[1], label='show line', **para_line)
        if suggest_abnormal:
            plt.plot(xc, yc, label='center', color='orange', ls='--')
            plt.plot(xc, yu, label='abnormal', color='salmon', ls='--')
            plt.plot(xc, xc * cu + iu, label='suggest abnormal', color='violet')
        plt.legend()
        plt.close()
    
        return fig

    def find_abnormal(self, target, method='kmean-abn', threshold_neibor=5, threshold_abnormal=3, update=False,
                      metric='euclidean', abnormal_method='iqr', random_state=66):
        '''
        Find abnormal samples:
    
        Parameters:
            - target (str): target value property
            - method (str): method to find abnormal samples. Default = 'kmean-std'
              'kmean-abn', 'dbscan-abn': classify points by K-mean (n_cluster=int threshold_neibor) or DBSCAN (eps=float threshold_neibor),
              In each class, regard points that have abs(target value - cluster_target_mean) > float threshold_abnormal * cluster_std as abnormal
              'local_similar': find the nearest k (int threshold_neibor) points or points within a distance of r (float threshold_neibor),
              calculate the z-score for the center points within its neighbors, if z-score > threshold_abnormal, regard as abnormal
              'silvtdf': obtain the distances and target value discrepancies for all sample pairs within a distance threshold (threshold_neibor). 
              Within a certain distance interval, one point from pairs that meet specified invalidity conditions are identified as invalid points, 
              pair target value discrepancie > threshold_abnormal * special value defined by abnormal_method. 
              This method can be operated in conjunction with function show_similarity_vs_targetdiff, which visualizes the relationship between 
              sample pair distances and target value discrepancies.
            - threshold_neibor (int or float): parameter for neighbor determination. Default = 5
            - threshold_abnormal (float or list/tuple): parameter for abnormal determination. Default = 3
            - update (bool): update or reset abnormal_idx. Default = False
            - metric (str): metric to use when calculating distance between instances. Default = 'euclidean'
              Select from ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
            - abnormal_method (str): statistical method for abnormal suggestion, 'std', 'iqr' or 'mad'. Default = 'iqr'
            - random_state (int): random state used in K-mean clustering algorithm. Default = 66
        '''
        # init
        abnormal_idx = []
        targets = np.array(self._get_properties_from_db(target))
        dist_simi_matrix = pairwise_distances(self.low_dimensional_describe, metric=metric)
        
        # finnd abnormal
        if   method == 'kmean-abn':
            kmeans = KMeans(n_clusters=threshold_neibor, random_state=random_state).fit(self.low_dimensional_describe)
            cluster_labels = kmeans.labels_
            for cluster in np.unique(cluster_labels):
                cluster_mask = (cluster_labels == cluster)
                cluster_y = targets[cluster_mask]
                abnormal_mask = detect_abnormal(cluster_y, targets, abnormal_method, threshold_abnormal)
                abnormal_idx.extend(np.where(cluster_mask & abnormal_mask)[0])
                
        elif method == 'dbscan-abn':
            dbscan = DBSCAN(eps=threshold_neibor)
            cluster_labels = dbscan.fit_predict(self.low_dimensional_describe)
            for cluster in np.unique(cluster_labels):
                if cluster != -1:
                    cluster_mask = (cluster_labels == cluster)
                    cluster_y = targets[cluster_mask]
                    abnormal_mask = detect_abnormal(cluster_y, targets, abnormal_method, threshold_abnormal)
                    abnormal_idx.extend(np.where(cluster_mask & abnormal_mask)[0])
                                
        elif method == 'local_similar':
            for i_sample in range(dist_simi_matrix.shape[0]):
                dist_to_i_sample = dist_simi_matrix[i_sample]
                sorted_index = np.argsort(dist_to_i_sample)
                sorted_array = dist_to_i_sample[sorted_index]
                if   type(threshold_neibor) is int:
                    nebrAself_index_dismat = sorted_index[: threshold_neibor]
                elif type(threshold_neibor) is float:
                    nebrAself_index_dismat = sorted_index[np.where(sorted_array < threshold_neibor)[0]]
                if len(nebrAself_index_dismat) >= 3:
                    local_y = targets[nebrAself_index_dismat]
                    abnormal_mask = detect_abnormal(local_y, local_y, abnormal_method, threshold_abnormal)
                    if abnormal_mask[0]:
                        abnormal_idx.append(i_sample)
                    
        elif method == 'silvtdf':
            # get similarty and target difference
            num_point = dist_simi_matrix.shape[0]
            sample_pairs = []
            similarities = []
            target_diffs = []
            for i in range(num_point):
                for j in range(num_point - i - 1):
                    j = j + i + 1
                    if dist_simi_matrix[i][j] < threshold_neibor:
                        sample_pairs.append([i, j])
                        similarities.append(dist_simi_matrix[i][j])
                        target_diffs.append(abs(targets[i] - targets[j]))
            sample_pairs = np.array(sample_pairs)
            similarities = np.array(similarities)
            target_diffs = np.array(target_diffs)
            # find abnormal pair
            if   type(threshold_abnormal) is int or type(threshold_abnormal) is float:
                threshold_abnormal = threshold_abnormal
            elif type(threshold_abnormal) is list or type(threshold_abnormal) is tuple:
                threshold_abnormal = similarities * threshold_abnormal[0] + threshold_abnormal[1]
            mask = (similarities < threshold_neibor) & (target_diffs > threshold_abnormal)
            sample_pairs = sample_pairs[mask]
            similarities = similarities[mask]
            target_diffs = target_diffs[mask]
            # find abnormal point
            for pair, similar, tardiff in zip(sample_pairs, similarities, target_diffs):
                pi, pj = pair
                if pi not in abnormal_idx and pj not in abnormal_idx:
                    num_i = (sample_pairs == pi).sum()
                    num_j = (sample_pairs == pj).sum()
                    if   num_i >  num_j:
                        abnormal_idx.append(pi)
                    elif num_i <  num_j:
                        abnormal_idx.append(pj)
                    elif num_i == num_j:
                        std_dev = []
                        for si in [pi, pj]:
                            dist_to_i_sample = dist_simi_matrix[si]
                            sorted_index = np.argsort(dist_to_i_sample)
                            nebrAself_index_dismat = sorted_index[: 10]
                            local_y = targets[nebrAself_index_dismat]
                            std_dev.append(np.abs(detect_abnormal(local_y, local_y, abnormal_method, threshold_abnormal, True)[0]))
                        zs_i, zs_j = std_dev
                        if   zs_i <= zs_j:
                            abnormal_idx.append(pj)
                        elif zs_i > zs_j:
                            abnormal_idx.append(pi)
    
        # store or update abnormal indexes
        if update and hasattr(self, 'abnormal_idx'):
            self.abnormal_idx.extend(abnormal_idx)
            self.abnormal_idx = list(set(self.abnormal_idx))
        else:
            self.abnormal_idx = abnormal_idx   

    def to_new_db(self, new_db, files='normal'):
        '''
        Make sampled or normal structure to a new ASE database.
    
        Parameter:
            - new_db (filt path): name and path to the new ASE database
            - files (str): 'sample' or 'normal'. Default = 'normal'
        '''
        new_db = connect(new_db)
        if   files == 'normal' and hasattr(self, 'abnormal_idx'):
            for row in self.database.select():
                if row.id - 1 not in self.abnormal_idx:
                    new_db.write(row.toatoms(), key_value_pairs=row.key_value_pairs, data=row.data)
        elif files == 'sample' and hasattr(self, 'sample_index'):
            for row in self.database.select():
                if row.id - 1 in self.sample_index:
                    new_db.write(row.toatoms(), key_value_pairs=row.key_value_pairs, data=row.data)

    def plot_scatter(self, color_by, select_show='all', group_by=None, figsize=(9, 6), 
                     para_scatter1={'cmap': 'cividis', 'edgecolors': 'k', 'linewidths': 0.3}, 
                     para_scatter2={'cmap': 'cividis', 'edgecolors': 'r', 'linewidths': 0.3}, 
                     labels=['Scatter Plot', 'Values', 'X-axis', 'Y-axis']):
        '''
        Plot scatter rather than show in chemscope
    
        Parameters:
            - color_by (str): color value property
            - select_show (str): only show one part of samples. 'all', 'sample', 'unsample', 'normal', 'abnormal'. Default = 'all'
            - group_by (str or None): group attribute, 'abnormal' or 'sample'. Default = None
            - figsize ((2, ) tuple): figure size. Default = (9, 6)
            - para_scatter1 (dict): parameters for plt.scatter, like cmap, s, marker, alpha, edgecolors, linewidths, label.
              Default = {'cmap': 'cividis', 'edgecolors': 'k', 'linewidths': 0.3}
            - para_scatter2 (dict): parameters for plt.scatter. scatter for samples not in group_by idx (abnormal or sampled points).
              Default = {'cmap': 'cividis', 'edgecolors': 'r', 'linewidths': 0.3}
            - labels (list of str): labels for title, cbar, x-axis and y-axis. Default = ['Scatter Plot', 'Values', 'X-axis', 'Y-axis']
        Return:
            - the figure
        '''
        fig     = plt.figure(figsize=figsize)
        ax      = plt.gca()
        
        if group_by is not None and group_by in ['abnormal', 'sample']:
            if   group_by == 'abnormal':
                group_idx = getattr(self, 'abnormal_idx')
            elif group_by == 'sample':
                group_idx = getattr(self, 'sample_index')
            values  = self._get_properties_from_db(color_by)
            scatter1 = np.zeros((0, 2))
            scatter2 = np.zeros((0, 2))
            values1  = []
            values2  = []
            for s_i in range(self.low_dimensional_describe.shape[0]):
                if   s_i not in group_idx:
                    scatter1 = np.vstack([scatter1, self.low_dimensional_describe[s_i]])
                    values1.append(values[s_i])
                elif s_i in group_idx:
                    scatter2 = np.vstack([scatter2, self.low_dimensional_describe[s_i]])
                    values2.append(values[s_i])
                    
            vmin = min(np.min(values1), np.min(values2))
            vmax = max(np.max(values1), np.max(values2))
            plot_scatter1 = ax.scatter(scatter1[:, 0], scatter1[:, 1], c=values1, vmin=vmin, vmax=vmax, **para_scatter1)
            plot_scatter2 = ax.scatter(scatter2[:, 0], scatter2[:, 1], c=values2, vmin=vmin, vmax=vmax, **para_scatter2)
            cbar = plt.colorbar(plot_scatter1)
            
        else:
            ids = list(np.array(self._get_properties_from_db('id')) - 1)
            if   select_show == 'all':
                select_ids = ids
            elif select_show == 'sample':
                select_ids = self.sample_index
            elif select_show == 'unsample':
                select_ids = [i for i in ids if i not in self.sample_index ]
            elif select_show == 'abnormal':
                select_ids = self.abnormal_idx
            elif select_show == 'normal':
                select_ids = [i for i in ids if i not in self.abnormal_idx]
    
            values   = self._get_properties_from_db(color_by, select_ids)
            scatters = np.array([self.low_dimensional_describe[i] for i in select_ids])
            
            plot_scatter = ax.scatter(scatters[:, 0], scatters[:, 1], c=values, **para_scatter1)
            cbar = plt.colorbar(plot_scatter)
            
        # ax.legend()
        cbar.set_label('Values')
        ax.set_title('Scatter Plot with Heatmap Coloring')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        if 'label' in para_scatter1:
            plt.legend()
        plt.close()
        
        return fig

    def _show_diff_group(self, properties, group_index, property_name='group', show_type='color', symbol_label='ingroup'):
        property_show = []
        if   show_type == 'symbol':
            pins = symbol_label
            pout = 'not ' + symbol_label
        elif show_type == 'color':
            pins = 1
            pout = 0
        for si in range(len(self.database)):
            if   si in group_index:
                property_show.append(pins)
            else:
                property_show.append(pout)
        properties.update({property_name: property_show})
        return properties

    @staticmethod
    def _atoms_height_judge(atoms, adsb_height):
        if adsb_height is None:
            method = None
        else:
            method = adsb_height[0]
            refer_height = adsb_height[1]
        if   method is None:
            height_judge = [True] * len(atoms)
            
        elif method == 'height':
            z_height =  atoms.positions[:, 2]
            height_judge = []
            for z in z_height:
                if   z > refer_height:
                    height_judge.append(True)
                else:
                    height_judge.append(False)
                    
        elif method == 'layer':
            z_height = atoms.positions[:, 2]
            refer_z_height = [z_height[0]]
            for z in z_height:
                to_refer = True
                for refer_z in refer_z_height:
                    if abs(z - refer_z) <= 0.36:
                        to_refer = False
                if to_refer:
                    refer_z_height.append(z)
            refer_z_height.sort()
            
            height_judge = []
            for z in z_height:
                for refer_z in refer_z_height:
                    if abs(z - refer_z) <= 0.36:
                        layer = refer_z_height.index(refer_z)
                        break
                if layer > refer_height:
                    height_judge.append(True)
                else:
                    height_judge.append(False)

        return height_judge
        
    @staticmethod
    def _atoms_symbol_judge(atoms, symbols):
        symbol_judge = []
        refer_numbers = [atomic_numbers[symbol] for symbol in symbols]
        for number in atoms.numbers:
            if number in refer_numbers:
                symbol_judge.append(True)
            else:
                symbol_judge.append(False)

        return symbol_judge

    def _get_db_symbols(self, delete_eles=False):
        db_symbols = [chemical_symbols[z] for z in np.unique(np.concatenate([row.toatoms().get_atomic_numbers() for row in self.database.select()]))]
        if delete_eles:
            db_symbols = [symbol for symbol in db_symbols if symbol not in self.delete_eles]
        return db_symbols

    def _get_atoms_list(self, delete_eles=False, select_ids=None):
        atoms_list = [row.toatoms() for row in self.database.select()]
        if delete_eles:
            atoms_list = [atoms[[atom.symbol not in self.delete_eles for atom in atoms]] for atoms in atoms_list]
        if select_ids is not None:
            atoms_list = [atoms_list[i] for i in select_ids]
        return atoms_list

    def _get_properties_from_db(self, property_name, select_ids=None):
        properties = [getattr(row, property_name) for row in self.database.select()]
        if select_ids is not None:
            properties = [properties[i] for i in select_ids]
        return properties


def calculate_line_equation(point1, point2):
    '''
    Calculate the coefficients a and b of the line segment equation y = ax + b passing through two points

    Parameters:
        - point1 ((2, ) array-like)
        - point2 ((2, ) array-like)
    Return:
        - a, b (float)
    '''
    x1, y1 = point1
    x2, y2 = point2
        
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    
    return a, b

    
def detect_abnormal(data, points=None, method='std', threshold=2, return_value=False):
    '''
    Use statistical methods to search for outliers or criteria for outliers from a set of data

    Parameters:
        - data ((n,) array-like): data for find abnormal criteria
        - points ((m, ) array-like or None): data that needs to be determined whether it is an outlier. Default = None
        - method (str): statistical method for abnormal suggestion, 'std', 'iqr' or 'mad'. Default = 'std'
        - threshold (float): threshold for abnormal suggestion. e.g. on 'std', abnormal line is mean + threshold * std. Default = 2
        - return_value (bool): return a numerical criterion (like z-score) or a Boolean variable representing abnormal or not. Default = False
    Return:
        - up_limit, center, low_limit if points is None
        - (points - center) / identifier if points and return_value
        - (points < low_limit) | (points > up_limit) if points and not return_value
    '''
    if   method == 'std':
        mean       = np.mean(data)
        identifier = np.std(data) # std
        center     = mean
        up_limit   = mean + identifier * threshold
        low_limit  = mean - identifier * threshold
    elif method == 'iqr':
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        identifier = q3 - q1 # IQR
        center     = median
        up_limit   = q3 + identifier * threshold
        low_limit  = q1 - identifier * threshold
    elif method == 'mad':
        median     = np.median(data)
        AD         = np.abs(data - median)
        identifier = np.median(AD) # MAD, * 1.4826 ~= std
        center     = median
        up_limit   = median + identifier * threshold
        low_limit  = median - identifier * threshold

    if points is not None:
        if return_value:
            return (points - center) / identifier
        else:
            return (points < low_limit) | (points > up_limit)
    else:
        return up_limit, center, low_limit


def analyze_point_distribution(x, y, n_bins=10, method='iqr', threshold=2):
    """
    Statistically analyze the distribution of y values of points on a two-dimensional plane in different x intervals

    Parameters:
        - x ((n, ) array-like)
        - y ((n, ) array-like)
        - n_bins (int): number of bins (intervals). Default = 10
        - method ('str'): statistical method for abnormal suggestion, 'std', 'iqr' or 'mad'. Default = 'std'
        - threshold (float): threshold for abnormal suggestion. e.g. on 'std', abnormal line is mean + threshold * std. Default = 2
    Return:
        - x_centers, y_uplimit, y_centers, y_lowlimit, lr_coef_up, lr_inter_up, lr_coef_down, lr_inter_down, 
    """
    # init
    x_min, x_max = np.min(x), np.max(x)
    x_interval   = (x_max - x_min) / n_bins
    x_centers    = np.zeros(n_bins)
    y_centers    = np.zeros(n_bins)
    y_uplimit    = np.zeros(n_bins)
    y_lowlimit   = np.zeros(n_bins)
    # calculate center, mean and std
    for i in range(n_bins):
        bin_start     = x_min + i * x_interval
        bin_end       = bin_start + x_interval
        mask          = (x >= bin_start) & (x < bin_end)
        points_in_bin = y[mask]
        if len(points_in_bin) > 0:
            uplm, cent, lolm = detect_abnormal(points_in_bin, None, method, threshold)
        else:
            uplm, cent, lolm = (0, 0, 0)
        x_centers[i]  = (bin_start + bin_end) / 2
        y_uplimit[i]  = uplm
        y_centers[i]  = cent
        y_lowlimit[i] = lolm
    # further process
    mask = (y_uplimit != 0) & (y_centers != 0) & (y_lowlimit != 0)
    x_centers  =  x_centers[mask]
    y_centers  =  y_centers[mask]
    y_uplimit  =  y_uplimit[mask]
    y_lowlimit = y_lowlimit[mask]
    # suggest line
    lr_model_up  = LinearRegression()
    lr_model_up.fit(x_centers.reshape(-1, 1), y_uplimit)
    lr_model_low = LinearRegression()
    lr_model_low.fit(x_centers.reshape(-1, 1), y_lowlimit)
    
    return x_centers, y_uplimit, y_centers, y_lowlimit, lr_model_up.coef_[0], lr_model_up.intercept_, lr_model_low.coef_[0], lr_model_low.intercept_


def infer_adsorption_site(slab, adsb=['N', 'H'], method='cutoff', judge=2.7, max_bond=2.7, defin_dist='dist', pymr=True, print_neibor=False):
    '''
    Function determine the adsorption site of the molecule based on the bond length
    
    Parameters:
        - slab (pymatgen.core.Structure): the slab structure. 
        - adsb (list): species for adsorbate. Default = ['N', 'H']
        - method (str): method for define bond judge distance. Default = 'cutoff'
          A bond whose length is less than the judge distance is considered a bond.
          'cutoff': judge distance will be the judge value itself
          'minplus': judge distance will be the minest bond distance for each adsorbate atom plus judge value
          'scalemin': judge distance will be the minest bond distance for each adsorbate atom multiply by judge
          'scaleradius': judge distance will be sum of atom radius multipy by judge. defin_dist is not work for this method.
        - judge (float): value used to help defend the judge distance. Default = 2.7
        - max_bond (float): the maximum value for bond defined by 'dist'. Default = 2.7
        - defin_dist (str): distance defination. Default = 'dist'
          'dist': bond distace will be the distance between two atoms
          'radius': bond distance will be the distance between two atoms subtract their radius
        - pymr (bool): use the radius from pymatgen, if not use radius from ase. Default = True
        - print_neibor (bool): whether or not to print operational information. Default = False
    Return:
        - sites: Index and element for each adsorption site atom
        - adsb_coordina: Coordina number to substrate atoms of each adsorbate atoms
        - connect: Bond indices for each atom connection. where i atoms are adsorbate atoms and j atoms are adsorption sites.
    '''
    # get distance matrix for adsorbate 
    adsb_indices = np.where(np.logical_or.reduce([np.array(slab.atomic_numbers) == value for value in [Element(e).number for e in adsb]]))[0]
    adsb_dist_matrix = slab.distance_matrix[adsb_indices]
    # mask self and adsb
    adsb_dist_matrix = np.where(adsb_dist_matrix !=0, adsb_dist_matrix, 6.66666)
    adsb_dist_matrix[:, adsb_indices] = 6.66666
    # calculate min and max ditance matrix
    if   method == 'cutoff' and defin_dist == 'dist':
        matrix_minmum = np.full(adsb_dist_matrix.shape, 0)
        matrix_maxmum = np.full(adsb_dist_matrix.shape, judge)
    elif method == 'minplus' and defin_dist == 'dist':
        matrix_minmum = np.repeat(np.min(adsb_dist_matrix, axis=1).reshape(-1, 1), adsb_dist_matrix.shape[1], axis=1)
        matrix_maxmum = matrix_minmum + judge
    elif method == 'scalemin' and defin_dist == 'dist':
        matrix_minmum = np.repeat(np.min(adsb_dist_matrix, axis=1).reshape(-1, 1), adsb_dist_matrix.shape[1], axis=1)
        matrix_maxmum = matrix_minmum * judge
    elif method == 'cutoff' and defin_dist == 'radius':
        array_radius_target = np.array([site.specie.atomic_radius if pymr else covalent_radii[site.specie.number] for site in slab])
        array_radius_source = np.concatenate([np.full((1, len(slab)), slab[ai].specie.atomic_radius if pymr else covalent_radii[slab[ai].specie.number]) for ai in adsb_indices], axis=0)
        array_radius = array_radius_target + array_radius_source
        matrix_minmum = array_radius
        matrix_maxmum = array_radius + judge
    elif method == 'minplus' and defin_dist == 'radius':
        array_radius_target = np.array([site.specie.atomic_radius if pymr else covalent_radii[site.specie.number] for site in slab])
        array_radius_source = np.concatenate([np.full((1, len(slab)), slab[ai].specie.atomic_radius if pymr else covalent_radii[slab[ai].specie.number]) for ai in adsb_indices], axis=0)
        array_radius = array_radius_target + array_radius_source
        matrix_minmum = array_radius + np.repeat(np.min(adsb_dist_matrix - array_radius, axis=1).reshape(-1, 1), adsb_dist_matrix.shape[1], axis=1)
        matrix_maxmum = matrix_minmum + judge
    elif method == 'scalemin' and defin_dist == 'radius':
        array_radius_target = np.array([site.specie.atomic_radius if pymr else covalent_radii[site.specie.number] for site in slab])
        array_radius_source = np.concatenate([np.full((1, len(slab)), slab[ai].specie.atomic_radius if pymr else covalent_radii[slab[ai].specie.number]) for ai in adsb_indices], axis=0)
        array_radius = array_radius_target + array_radius_source
        mindist_matrix = np.repeat(np.min(adsb_dist_matrix - array_radius, axis=1).reshape(-1, 1), adsb_dist_matrix.shape[1], axis=1)
        matrix_minmum = array_radius + mindist_matrix
        matrix_maxmum = array_radius + mindist_matrix * judge
    elif method == 'scaleradius':
        array_radius_target = np.array([site.specie.atomic_radius if pymr else covalent_radii[site.specie.number] for site in slab])
        array_radius_source = np.concatenate([np.full((1, len(slab)), slab[ai].specie.atomic_radius if pymr else covalent_radii[slab[ai].specie.number]) for ai in adsb_indices], axis=0)
        array_radius = array_radius_target + array_radius_source
        matrix_minmum = array_radius
        matrix_maxmum = array_radius * judge
    # get bond indices
    # bond_adsb, bond_metal = np.where((matrix_minmum <= adsb_dist_matrix) & (adsb_dist_matrix < matrix_maxmum) & (adsb_dist_matrix < max_bond))
    bond_adsb, bond_metal = np.where((adsb_dist_matrix <= matrix_maxmum) & (adsb_dist_matrix <= max_bond))
    not_adsb = np.where(~np.in1d(bond_metal, adsb_indices))[0]
    bond_adsb = adsb_indices[bond_adsb[not_adsb]]
    bond_metal = bond_metal[not_adsb]
    # get output
    site = set([str(bmi) + '-' +slab[bmi].specie.symbol for bmi in bond_metal])
    adsb_coordinate = Counter([slab[ai].specie.symbol + '-' + str(np.count_nonzero(bond_adsb == ai)) for ai in adsb_indices])
    connect = [list(bond_adsb), list(bond_metal)]
    # print
    if print_neibor:
        print('sitei | sitej | distance | judge')
        for i, j in zip(*connect):
            ai = np.where(adsb_indices == i)[0][0]
            if   defin_dist == 'dist':
                print('%5s | %5s | %8.5f | %5.3f' % ('-'.join([str(i), slab.species[i].symbol]), '-'.join([str(j), slab.species[j].symbol]), adsb_dist_matrix[ai][j], np.where(matrix_maxmum < max_bond, matrix_maxmum, max_bond)[ai][j]))
            elif defin_dist == 'radius':
                print('%5s | %5s | %8.5f | %5.3f' % ('-'.join([str(i), slab.species[i].symbol]), '-'.join([str(j), slab.species[j].symbol]), (adsb_dist_matrix - array_radius)[ai][j], (np.where(matrix_maxmum < max_bond, matrix_maxmum, max_bond)- array_radius)[ai][j]))
    return site, adsb_coordinate, connect
