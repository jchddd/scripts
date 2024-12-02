import pymatgen.analysis.adsorption as adsana
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

from ase.visualize import view

from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import colors

import numpy as np
import math
import os

from Jworkflow.utility import num_to_subscript


def view_structure_VASP(structure):
    '''
    Function to view structure use ASE module

    Parameter:
        - structure: The structure to view / pymatgen.core.Structure
    '''
    ase_atoms = AseAtomsAdaptor.get_atoms(structure)
    view(ase_atoms)


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
    ax1.bar(bar_seperate_str, bar_count, align='edge', width=-0.8)
    for i in range(len(bar_count)):
        ax1.text(i, bar_count[i], str(bar_count[i]), fontsize=26, ha='right', va='bottom')
    # boxline graph
    ax2 = Fig.add_axes([0.7, 0.1, 0.2, 0.8])
    ax2.tick_params(length=6, width=3, labelsize=26)
    ax2.scatter([0] * len(data), data, zorder=6)
    Q1 = np.percentile(data, (25))
    Q2 = np.percentile(data, (50))
    Q3 = np.percentile(data, (75))
    IQR = Q3 - Q1
    edge_upper = Q3 + 1.5 * IQR
    edge_lower = Q1 - 1.5 * IQR
    for data_statistic in [edge_lower, Q1, Q2, Q3, edge_upper]:
        color = 'r' if data_statistic == Q2 else 'k'
        line_len = 0.3 if data_statistic == edge_lower or data_statistic == edge_upper else 0.5
        ax2.plot([-1 * line_len, line_len], [data_statistic, data_statistic], c=color)
        ax2.text(-0.3, data_statistic, str(round(data_statistic, 2)), c=color, ha='left', va='bottom', fontsize=26)
    ax2.plot([-0.5, -0.5], [Q1, Q3], c='k')
    ax2.plot([0.5, 0.5], [Q1, Q3], c='k')
    ax2.plot([0, 0], [edge_lower, Q1], c='k', ls='--', zorder=3)
    ax2.plot([0, 0], [Q3, edge_upper], c='k', ls='--', zorder=3)
    plt.show()
    # return seperaters
    if return_seperate:
        return bar_seperate


def gradient_graphics(axes, x, y, colormap='viridis', color_matrix=[[0], [1]], edge_para=['k', 1], extend=0.1, alpha=1, zorder=666):
    '''
    Function to plot gradient graphics

    Parameters:
        - axes: Axes to where plot the gradient graphics / matplotlib.axes
        - x: Array of the x data for the patch / (n)D-array
        - y: Array of the y data for the patch / (n)D-array
        - colormap: The Custom or loaded colormap / str, default 'viridis', or (n,2)D-array like [(0,'r'),(1,'w')]
        - color_matrix: color matrix in a square area / (n,m)D-array, default [[0],[1]]
        - edge_para: Edge color and edge line width of the patch / (2)D-array, default ['k',1]
        - extend: extend of the gradient area / float, default 0.1
        - alpha: Transparency / float, default 1
        - zorder: The overlay priority of the graph, zorde of the edge is this zorder + 0.6 / float, default 666

    '''
    # define colormap
    if type(colormap) == str:
        cmap = colormap
    elif type(colormap) == list:
        cmap = colors.LinearSegmentedColormap.from_list('Cmap', colormap, N=256)
    # creat and add patch
    path = Path(np.array([x, y]).transpose())
    patch = patches.PathPatch(path, facecolor='none', edgecolor=edge_para[0], lw=edge_para[1], zorder=zorder + 0.6)
    axes.add_patch(patch)
    # add gradient image
    axes.imshow(color_matrix, interpolation="bicubic", origin='lower', aspect="auto", clip_path=patch, clip_on=True,
                alpha=alpha, cmap=cmap, extent=[min(x) - extend, max(x) + extend, min(y) - extend, max(y) + extend], zorder=zorder)


def plot_slab(ax, slab, scale=1., decay=0.2, max_z_cart=None, repeat=3, radius_dict=None, color_dict=None, w_bottom_extend=0, atom_edge_para=['k', 0.3], draw_unit_cell=True,
              cell_edge_para=['k', '-', 1, 1], axis_extend=1.3, cover_outside='w', lw_spines=1, xy_lim_reset=None
              ):
    '''
    Function to plot slab

    Parameters:
        - ax: Axes to where plot the slab / matplotlib.axes
        - slab: Slab structure / pymatgen.core.Structure
        - scale: Radius scaling for sites / float, default 1
        - decay: How the alpha-value decays along the z-axis / float, default 0.2
        - max_z_cart: Set a z coordinate to perform the decay / float, default None
        - repeat: Number of repeating unit cells to visualize / int, default 3
        - radius_dict: Element radius dict / dict like {'Ru':2}, default, None, use pymatgen default dict
        - color_dict: Element color dict / dict like {'Ru','r'}, default, None, use pymatgen default dict
        - w_bottom_extend: The extended radius of each site on a white background / float, default 0
        - atom_edge_para: Color and line width for each site edge / list, default ['k', 0.3]
        - draw_unit_cell: Flag indicating whether or not to draw cell / bool, default True
        - cell_edge_para: Color, line style, line width and alpht for unit cell edge / list, default ['k', '-', 1, 1]
        - axis_extend: To set the axes limits, is essentiallya fraction of the unit cell limits / float, default 1.3
        - cover_outside: Whether to cover area outside the unit cell / str or None, default 'w', the cover color
        - lw_spines: Axes spines line width / float, default 1
        - xy_lim_reset: Reset x y lim at the end / (4) list for minx, maxx, miny and maxy
    Return:
        - the axes
    '''
    # draw slab information
    orig_cell = slab.lattice.matrix.copy()
    slab = slab.copy()
    slab.make_supercell([repeat, repeat, 1])
    coords = np.array(sorted(slab.cart_coords, key=lambda x: x[2]))
    sites = sorted(slab.sites, key=lambda x: x.coords[2])
    corner = [0, 0, slab.lattice.get_fractional_coords(coords[-1])[-1]]
    corner = slab.lattice.get_cartesian_coords(corner)[:2]
    verts = orig_cell[:2, :2]
    lattsum = verts[0] + verts[1]
    # Draw circles at sites and stack them accordingly
    if not max_z_cart:
        max_z_cart = np.max(coords[:, 2])
    alphas = 1 - decay * (max_z_cart - coords[:, 2])
    alphas = alphas.clip(min=0, max=1)
    for n, coord in enumerate(coords):
        if radius_dict:
            r = radius_dict[sites[n].specie.symbol]
        else:
            r = sites[n].species.elements[0].atomic_radius * scale
        if color_dict:
            color = color_dict[sites[n].specie.symbol]
        else:
            color = adsana.color_dict[sites[n].species.elements[0].symbol]
        ax.add_patch(patches.Circle(coord[:2] - lattsum * (repeat // 2), r + w_bottom_extend, color="w", zorder=2 * n))
        ax.add_patch(patches.Circle(coord[:2] - lattsum * (repeat // 2), r, facecolor=color, alpha=alphas[n], edgecolor=atom_edge_para[0], lw=atom_edge_para[1], zorder=2 * n + 1))
    # Draw unit cell
    verts = np.insert(verts, 1, lattsum, axis=0).tolist()
    verts += [[0.0, 0.0]]
    verts = [[0.0, 0.0]] + verts
    verts = [(np.array(vert) + corner).tolist() for vert in verts]
    if draw_unit_cell:
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor="none", edgecolor=cell_edge_para[0], ls=cell_edge_para[1], lw=cell_edge_para[2], alpha=cell_edge_para[3], zorder=2 * n + 3)
        ax.add_patch(patch)
    # set axis
    ax.set_aspect('equal', adjustable='datalim')
    ax.apply_aspect()
    xmin, xmax, ymin, ymax = ax.axis()
    xvy_axis = (xmax - xmin) / (ymax - ymin)
    xy_length = np.abs(orig_cell[:2, :2]).sum(axis=0)
    xvy_slab = xy_length[0] / xy_length[1]
    xextent = xy_length[0] / 2
    yextent = xy_length[1] / 2
    if xvy_slab > xvy_axis:
        yextent = xextent / xvy_axis
    elif xvy_slab < xvy_axis:
        xextent = yextent * xvy_axis
    center = corner + lattsum / 2.0
    x_lim = [center[0] - xextent * axis_extend, center[0] + xextent * axis_extend]
    y_lim = [center[1] - yextent * axis_extend, center[1] + yextent * axis_extend]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.apply_aspect()
    xmin, xmax, ymin, ymax = ax.axis()
    ax.set_xticks([])
    ax.set_yticks([])
    # cover outside
    if cover_outside:
        xmin += -0.6
        ymin += -0.6
        xmax += 0.6
        ymax += 0.6
        x = [xmin] + [verts[i][0] for i in range(len(verts))] + [xmin, xmin, xmax, xmax, xmin]
        y = [ymin] + [verts[i][1] for i in range(len(verts))] + [ymin, ymax, ymax, ymin, ymin]
        path = Path(np.array([x, y]).transpose())
        patch = patches.PathPatch(path, facecolor=cover_outside, edgecolor='w', lw=0, zorder=2 * n + 2)
        ax.add_patch(patch)
    # reset xy lim
    if xy_lim_reset:
        ax.set_xlim([xy_lim_reset[0], xy_lim_reset[1]])
        ax.set_ylim([xy_lim_reset[2], xy_lim_reset[2]])
        ax.apply_aspect()
    # set border thickness
    ax.spines['bottom'].set_linewidth(lw_spines)
    ax.spines['left'].set_linewidth(lw_spines)
    ax.spines['top'].set_linewidth(lw_spines)
    ax.spines['right'].set_linewidth(lw_spines)
    path = Path(np.array([[xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]]).transpose())
    patch = patches.PathPatch(path, facecolor="none", edgecolor='k', lw=lw_spines, zorder=2 * n + 4)
    return ax


def plot_configration(df, plot_part, sequence=None, structure_path=os.getcwd(), plt_setting=[17, 4, 1, 0.5, 0], titles=None, title_para=[9, -3], **plot_slab_para):
    '''
    Function to plot a series of vasp structures

    Parameters:
        - df: DataFrame store the vasp structure names with raw as system names and columns as adsorbate names / pd.DataFrame
        - plot_part: Which system to plot, corresponding to one of the raw or column name on df / str
        - sequence: In what order to draw the pictures, corresponding to different column names on df / list
        - structure_path: Path to where vasp structure files are stored / str, path
        - plt_setting: Figure size and spacing / (5) list, figure width, number of images per row, ratio of height to width of each images, hspace, wspace
        - titles: Titles for each images / list, same dimention as sqeuence
        - title_para: Parameters for title format / (2) list, fontsize and pad
        - **plot_slab_para: Other parameters on plot_slab_para
    Returen:
        - the figure
    '''
    if plot_part in df.index:
        if not sequence:
            sequence = df.columns
        strus = [df.at[plot_part, adss] for adss in sequence]
    elif plot_part in df.columns:
        if not sequence:
            sequence = df.index
        strus = [df.at[sys, plot_part] for sys in sequence]
    titles = [str(ti) for ti in sequence] if not titles else titles

    length = len(strus)
    rows = math.ceil(length / plt_setting[1])
    width = (plt_setting[0] - plt_setting[-1] * (plt_setting[1] - 1)) / plt_setting[1]
    height = width * plt_setting[2]
    heightT = (height + plt_setting[3]) * rows
    fig = plt.figure(figsize=(plt_setting[0] / 2.54, heightT / 2.54))
    plt.subplots_adjust(left=0, bottom=0, top=1 - plt_setting[3] / heightT, right=1, hspace=plt_setting[3] / height, wspace=plt_setting[-1] / width)
    for i, struc in enumerate(strus):
        if type(struc) == str:
            axes = fig.add_subplot(rows, plt_setting[1], i + 1)
            axes.axis('off')
            slab = Structure.from_file((os.path.join(structure_path, struc + '.vasp')))
            plot_slab(axes, slab, **plot_slab_para)
            if titles[i]:
                axes.set_title(titles[i], fontsize=title_para[0], pad=title_para[1])
    return fig


def energy_diagram(ax, df, sys, path_steps, lp_step=[1, 1.2, '-'], lp_link=[1, 0.6, '--'], xticks=None, xtickpara=[6, -30, 'left'],
                   set_yticks=None, set_ylabel=None, xlim=None, ylim=None, set_legend=None, show_space_line=False, space_line_para=['lightgrey', 0.6, '--'],
                   mark_PDS=None, annotatepara=[0.6,6,6], bar_elp=None, elp_insert=66):
    '''
    Function to plot the energy diagram

    Parameters:
        - ax: The axes / matplotlib.axes
        - df: DataFrame that store the energy data / pandas.DataFrame
        - path_steps: Parameters to control the step plot / (n, 2, x) list
            Creat a new list at first [], add a new list inside to represent steps in the same x axis [[], ...]
            Add two list inside it to represent steps in this x and their links to previous steps, [[[], []], ...]
            The first list in the first x should only contain one list  represents the step and ignore the list that represent the links
            In this case, the list of the previous x can be [[[('NNHv','r')]], [[], []], ...]
            Then, add steps and their colors in the first list of the next x, [[[('NNH2','r'),('NHNH','b')], []], ...]
            Notice that all step names should be in the column names of the df
            Add lists with the same number of steps in the second list, [[[('NNH2','r'),('NHNH','b')], [[], []]], ...]
            Add the index and linking colour of previous steps to which it link to [[[('NNH2','r'),('NHNH','b')], [[(0,'r')], [(0,'b')]]], ...]
            A list in one x must have a list for steps, the list for links can be ignored
        - lp_step: Line parameters of steps / (3) list including step length, line width and line style, default [1, 1.2, '-']
        - lp_link: Line parameters of links / (3) list including link length, line width and line style, default [1, 0.6, '--']
        - xticks: Xticks / (x) list, default None, if None, it will use step names in path_steps would be used as xticks
        - xtickpara: X tick parameters / (3) list including fontsize, ratation and alignment mode, default [6, -30, 'left']
        - set_yticks: Redefine yticks is needed / [(y), fontsize] , new y ticks and its fontsize, default None.
        - set_ylabel: Set Y label / (2) list, ylabel and its fontsize, default None
        - xlim: X axis lims / (2) list include minX, maxX, default None
        - ylim: Y axis lims / (2) list include minY, maxY, default None
        - set_legend: Set legend / (3) list include legend informations, fontsize and index location, default None
            Legend information is (n, 4) list including label name, color, width and style of the line for each legends
        - show_space_line: Show lines that split each steps / bool, default False
        - space_line_para: Color, line width and line style for space line / (3,) list, default ['grey', 0.6, '--'] 
        - mark_PDS: Use arrows to maek PDSs / (n, 3) list including PDS column name, arrow color and text info, default None
            Text info is used to write the value of PDS, it can be ignored and use a (n, 2) list for this parameter
            It should be a (4) list including x and y offset from the center of the arrow, text color and fontsize
        - annotatepara: Properity of the arrow that marks PDS / (3) list including line width, head length and width, default [0.6,6,6]
        - bar_elp: Use three ellipses to draw a barrier / (n, 4 or 3, k) list, 4 or 3 lists are needed for a single barrier, default None
            The first list is for a barrier is the three coterminous steps in the barrier, (k = 3)
            The second controls the shape of ellipses with joint line slop, height and half length for center ellipse (appear part), (k = 3)
            The third controls the line shap including line width, color and line style, (k = 3)
            The final controls the text info with x and y offset from the top of ellipses, color and fontsize, it can be ignored (k = 4)
        - elp_insert: Insert point number use to draw the ellipse / int, default 66
    Cautions: 
        - You can use 'none' color to hide unexpected steps or links
        - The length of each step and links projected on the X-axis is 1 (default) and the diagram starts from 0
        - Encountered an error when drawing ellipses, you can try increasing k and height or decreasing length
    '''
    
    xtick_position = []
    # loop and plot step
    y_ends = [6]
    x_start = 0 - lp_link[0]
    for stepinfo_in_x in path_steps:
        for i, step in enumerate(stepinfo_in_x[0]):
            energy = df.at[sys, step[0]]
            if len(stepinfo_in_x) == 2:
                for link in stepinfo_in_x[1][i]:
                    y_start = y_ends[link[0]]
                    ax.plot([x_start, x_start + lp_link[0]], [y_start, energy], lw=lp_link[1], ls=lp_link[2], c=link[1], zorder=36)
            x_start += lp_link[0]
            ax.plot([x_start, x_start + lp_step[0]], [energy, energy], lw=lp_step[1], ls=lp_step[2], c=step[1], zorder=66)
            x_start -= lp_link[0]
        y_ends = [df.at[sys, istep[0]] for istep in stepinfo_in_x[0]]
        x_start += (lp_link[0] + lp_step[0])
        xtick_position.append(x_start - lp_step[0]/2)
    # set xticks
    ax.set_xticks(xtick_position)
    if not xticks:
        xticks = ['/'.join([num_to_subscript(istep[0]) for istep in stepinfo_in_x[0]]) for stepinfo_in_x in path_steps]
    ax.set_xticklabels(xticks, fontsize=xtickpara[0], rotation=xtickpara[1], ha=xtickpara[2])
    # set yticks
    if set_yticks:
        ax.set_yticks(set_yticks[0])
        ax.set_yticklabels(set_yticks[0], fontsize=set_yticks[1])
    # set ylabel
    if set_ylabel:
        ax.set_ylabel(set_ylabel[0], fontsize=set_ylabel[1])
    # set xylim
    ax.set_xlim([0, x_start])
    if xlim:
        ax.set_xlim([xlim[0], xlim[1]])
    if ylim:
        ax.set_ylim([ylim[0], ylim[1]])
    # set legend
    if set_legend:
        for legend in set_legend[0]:
            ax.plot([-0.00666] * 2, [-0.00666] * 2, label=legend[0], c=legend[1], lw=legend[2], ls=legend[3])
        ax.legend(facecolor='none', edgecolor='none', fontsize=set_legend[1], loc=set_legend[2])
    # show_space_line
    if show_space_line:
        line_position = np.array(xtick_position) + lp_step[0] / 2 + lp_link[0] / 2
        for lp in line_position:
            ax.axvline(x=lp, c=space_line_para[0], lw=space_line_para[1], ls=space_line_para[2], zorder=0)
    # mark PDS
    if mark_PDS:
        for pdsinfo in mark_PDS:
            pds_steps = df.at[sys, pdsinfo[0]].split('->')
            pds_energys = [df.at[sys, pds_steps[0]], df.at[sys, pds_steps[1]]]
            for i, stepinfo_in_x in enumerate(path_steps):
                if pds_steps[0] in [step[0] for step in stepinfo_in_x[0]]:
                    pds_x = [i * lp_link[0] + (i + 1) * lp_step[0], (i + 1) * (lp_step[0] + lp_link[0])]
                    break
            arrowprops = dict(color=pdsinfo[1],width=annotatepara[0],headlength=annotatepara[1],headwidth=annotatepara[2])
            ax.annotate("",xy=(pds_x[1], pds_energys[1]),xytext=(pds_x[0], pds_energys[0]),arrowprops=arrowprops,zorder=99)
            if len(pdsinfo) == 3:
                textinfo = pdsinfo[-1]
                center = [np.average(pds_x), np.average(pds_energys)]
                ax.text(center[0] + textinfo[0], center[1] + textinfo[1], str(round(np.diff(pds_energys)[0], 2)), 
                        c=textinfo[2], fontsize=textinfo[3], ha='center', va='center', zorder=100)
    # plot barrier with ellipse
    if bar_elp:
        for beinfo in bar_elp:
            bar_steps = beinfo[0]
            for i, stepinfo_in_x in enumerate(path_steps):
                if bar_steps[0] in [istep[0] for istep in stepinfo_in_x[0]]:
                    break
            point_start = [i * lp_link[0] + (i + 1) * lp_step[0], df.at[sys, bar_steps[0]]]
            point_barrier = [(i + 1) * lp_link[0] + (i + 3/2) * lp_step[0], df.at[sys, bar_steps[1]]]
            point_end = [(i + 2) * (lp_link[0] + lp_step[0]), df.at[sys, bar_steps[2]]]
            plot_ellipse_barrier(ax,
                                point_start,
                                point_barrier,
                                point_end,
                                beinfo[1][0],
                                beinfo[1][1],
                                beinfo[1][2],
                                beinfo[2],
                                elp_insert,
                                99)
            if len(beinfo) == 4:
                ax.text(point_barrier[0] + beinfo[3][0], point_barrier[1] + beinfo[3][1], 
                        str(round(point_barrier[1] - point_start[1], 2)),
                        c=beinfo[3][2], fontsize=beinfo[3][3], ha='center', va='center', zorder=100)

                
def plot_ellipse_barrier(ax, point_start, point_barrier, point_end, k=16, h=0.3, l=0.6, linepro=[0.6,'k','-'], insert=666, zorder=666):
    '''
    Function that use three ellipses to plot barrier

    Parameters:
        - ax: The axes / matplotlib.axes
        - point_start: The start point / (2) list
        - point_barrier: The center point / (2) list
        - point_end: The end point / (2) list
        - k: joint line slop / float, default 16
        - h: The height of the part of the ellipse that appears in the figure / float, default 0.3
        - l: The half length of the part of the ellipse that appears in the figure / float, default 0.6
        - linepro: Line style including line width, color and style / (3) list ,default [0.6,'k','-']
        - insert: Insert point number use to draw the ellipse / int, default 666
        - zorder: Zorder of the line / float, default 666
    Cautions: 
        - The distribution of points and parameter values need to be reasonable in order to plot normally
        - Encountered an error when drawing ellipses, you can try increasing k and h or decreasing l
    '''    
    # initial
    p1 = point_start
    p2 = point_barrier
    p3 = point_end
    point_joint_left = [p2[0] - l, p2[1] - h]
    point_joint_right = [p2[0] + l, p2[1] - h]
    theta1 = np.linspace(0, 2 * np.pi, insert)
    theta2 = np.linspace(2 * np.pi, 0, insert)
    x = []
    y = []
    # left ellipse
    yd_pjlTp1 = point_joint_left[1] - p1[1]
    xd_pjlTp1 = point_joint_left[0] - p1[0]
    B_ellipse_left = (yd_pjlTp1 - k * xd_pjlTp1) * yd_pjlTp1 / (yd_pjlTp1 - k * xd_pjlTp1 + yd_pjlTp1)
    b_tll = yd_pjlTp1 - k * xd_pjlTp1 - B_ellipse_left
    A_ellipse_left = np.power(-1 * xd_pjlTp1 * b_tll / k, 1 / 2)
    xl = A_ellipse_left * np.cos(theta1) + p1[0]
    yl = B_ellipse_left * np.sin(theta1) + p1[1] + B_ellipse_left
    for i in range(insert):
        if xl[i] >= p1[0] and yl[i] <= point_joint_left[1]:
            x.append(xl[i])
            y.append(yl[i])
    # center ellipse
    yd_p2Tpjl = p2[1] - point_joint_left[1]
    xd_p2Tpjl = p2[0] - point_joint_left[0]
    B_ellipse_center = (yd_p2Tpjl - k * xd_p2Tpjl) * yd_p2Tpjl / (yd_p2Tpjl - k * xd_p2Tpjl + yd_p2Tpjl)
    b_tll = yd_p2Tpjl - k * xd_p2Tpjl - B_ellipse_center
    A_ellipse_center = np.power(-1 * xd_p2Tpjl * b_tll / k, 1 / 2)
    xc = A_ellipse_center * np.cos(theta2) + p2[0]
    yc = B_ellipse_center * np.sin(theta2) + p2[1] - B_ellipse_center
    for i in range(insert):
        if xc[i] >= point_joint_left[0] and xc[i] <= point_joint_right[0] and yc[i] >= point_joint_left[1]:
            x.append(xc[i])
            y.append(yc[i])
    # right ellipse
    yd_pjrTp3 = point_joint_right[1] - p3[1]
    xd_pjrTp3 = p3[0] - point_joint_right[0]
    B_ellipse_right = (yd_pjrTp3 - k * xd_pjrTp3) * yd_pjrTp3 / (yd_pjrTp3 - k * xd_pjrTp3 + yd_pjrTp3)
    b_tlr = yd_pjrTp3 - k * xd_pjrTp3 - B_ellipse_right
    A_ellipse_right = np.power(-1 * xd_pjrTp3 * b_tlr / k, 1 / 2)
    xr = A_ellipse_right * np.cos(theta1) + p3[0]
    yr = B_ellipse_right * np.sin(theta1) + p3[1] + B_ellipse_right
    for i in range(insert):
        if xr[i] <= p3[0] and yr[i] <= point_joint_right[1]:
            x.append(xr[i])
            y.append(yr[i])
    # plot
    ax.plot(x, y, lw=linepro[0], c=linepro[1], ls=linepro[2], zorder=zorder)