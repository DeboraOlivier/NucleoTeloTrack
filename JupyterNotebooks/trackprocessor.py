#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrackProcessor
--------------
This module provides functions used to process tracked objects from CellProfiler.
The main function is process_data() which takes a csv file as an input. 
The csv file contains information of tracked data from a movie. The data will 
be cleaned up, then each tracked object is extracted, removed artifacts, 
corrected the transistion states, aligned by referenced timepoints.
The final output is a table containing aligned, cleaned tracks and their features. 
It is saved into a csv file. The results at intermediate steps are also saved.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import measure
import networkx as nx

def save_plot_matrix(file_pattr,M,nrows_limit,yticklabels,title,cmap="jet",vmin=None,vmax=None,cticklabel=None):
    """Matrix of a given feature is plotted and saved. 
    
    The Image Number is shown in the column, and the tracked objects in row.
    
    Args:
        file_pattr (str): file name pattern.
        M (2D numpy matrix [float]): matrix to be plotted.
        nrows_limit (int): maximal number of row to be plotted. Matrix is divided into smaller plots for clearer visual.
        yticklabels (list [string]): list of tracked object labels.
        title (str): plot title.
        cmap (str): color map.
        vmin (float): value smaller than vmin is displayed as vmin.
        vmax (floag): value larger than vmax is displayed as vmax.
        cticklabel (list [str]): color bar tick labels.
        
    Returns:
        plot is saved into .png file.
    
    """
    nrow = M.shape[0]
    ntimes = int(np.ceil(nrow/nrows_limit))
    ytick_begin = 0

    for itime in range(ntimes):
        ytick_end = ytick_begin + nrows_limit - 1
        if ytick_end > nrow-1:
            ytick_end = nrow-1

        fig = plt.figure(figsize=(9,8))
        ax = fig.add_subplot(111)

        pl = ax.imshow(M[ytick_begin:ytick_end+1],cmap=cmap,vmin=vmin,vmax=vmax);

        ax.set_aspect(3);
        ax.grid("on");
        ax.set_xticks(np.arange(0,M.shape[1],5));
        ax.set_xticklabels(np.arange(0,M.shape[1],5)+1,fontsize=5);
        ax.set_ylim(-0.5,ytick_end-ytick_begin+0.5)
        ax.set_yticks(range(ytick_end-ytick_begin+1));
        ax.set_yticklabels(yticklabels[ytick_begin:ytick_end+1],fontsize=5);
        ax.set_xlabel("Image Number")
        ax.set_ylabel("TrackObject_Label");

        cbax = fig.colorbar(pl,shrink=0.3);
        if cticklabel is not None:
            cbax.set_ticks(range(len(cticklabel)))
            cbax.set_ticklabels(cticklabel);

        ax.set_title(title);

        fig.savefig(file_pattr+"_"+str(itime)+".png",dpi=150)
        plt.close(fig)

        ytick_begin = ytick_end + 1
        
# def save_plot_xy_track(filename,subdata,title,xmax,ymax,cmap="jet",vmin=None,vmax=None,cticklabel=None,norm=None):
    
#     fig = plt.figure(figsize=(8,5))
#     ax = fig.add_subplot(111)
#     for _,g in subdata.groupby('TrackObjects_Label'):
#         x = g['AreaShape_Center_X'].values
#         y = g['AreaShape_Center_Y'].values
#         ax.plot(x,y,c="k",lw=0.5)
#         for t,gg in g.groupby('TrackObjects_StateNumber'):
#             x = gg['AreaShape_Center_X'].values
#             y = gg['AreaShape_Center_Y'].values
#             ax.scatter(x,y,s=15,c=[t for _ in range(len(x))],cmap=cmap,vmin=-0.5,vmax=6.5,edgecolors="black",lw=0.3,alpha=1.)

#     cbax = fig.add_axes([0.85, 0.3, 0.01, 0.4]);
#     cb = mpl.colorbar.ColorbarBase(cbax, cmap=cmap, norm=norm, orientation='vertical');
#     cb.set_ticks(range(7))
#     cb.set_ticklabels(cticklabel);

#     ax.set_xlabel("X");
#     ax.set_ylabel("Y");
#     ax.axis("equal");
#     ax.hlines(0,0,xmax,lw=1,linestyles="dashed");
#     ax.hlines(ymax,0,xmax,lw=1,linestyles="dashed");
#     ax.vlines(0,0,ymax,lw=1,linestyles="dashed");
#     ax.vlines(xmax,0,ymax,lw=1,linestyles="dashed");

#     ax.set_title(title);
    
#     fig.savefig(filename,dpi=150)
#     plt.close(fig)
    
# def save_plot_xy_track_exlu(filename,X,Y,title,xmax,ymax):
#     """Saving track after excluding positions intersecting image border.
#     """
    
#     fig = plt.figure(figsize=(8,5))
#     ax = fig.add_subplot(111)
#     for itrack in range(X.shape[0]):
#         ix = np.argwhere((X[itrack]!=-1)&(Y[itrack]!=-1)).flatten()
#         if len(ix)!=0:
#             ax.plot(X[itrack,ix],Y[itrack,ix],'-o',ms=3,markerfacecolor="None",markeredgewidth=0.5)

#     ax.set_xlabel("X");
#     ax.set_ylabel("Y");
#     ax.axis("equal");
#     ax.hlines(0,0,xmax,lw=1,linestyles="dashed");
#     ax.hlines(ymax,0,xmax,lw=1,linestyles="dashed");
#     ax.vlines(0,0,ymax,lw=1,linestyles="dashed");
#     ax.vlines(xmax,0,ymax,lw=1,linestyles="dashed");

#     ax.set_title(title);
    
#     fig.savefig(filename,dpi=150)
#     plt.close(fig)

def remove_intersect_border_by_circle(M,F,X,Y,R1,R2,xmax,ymax,bymajor=True,percentage=0.8):
    """Remove tracked positions intersecting the image border. Only the positions
    touching the border but not entire track are excluded. The border-touching object 
    is determined by checking the circle whose radius defined by the major axis.
    
    Work for CellProfiler >= 3.x.
    
    Args:
        M (2D numpy matrix [float]): Transistion table.
        F (List of 2D matrices [float]): List of corresponding feature table.
        X, Y, R1, R2, Area (2D matrix [float]): x, y positions, minor and major radii, area of nucleus
        xmax, ymax (float): movie width and height.
        bymajor (boolean): axis used to check the border touching. Set bymajor = False, then the minor axis is used.
        percentage (float):  fraction of radius used to compute circle, smaller value gives smaller circle for detecting border-touching.
        
    Returns:
        Feature matrices, center coordinates of tracks after border-touching removing.
    
    """
    Mr = M.copy()
    Fr = [FF.copy() for FF in F]
    Xr,Yr = X.copy(),Y.copy()
    
    # check which axis is used to check the touching
    if bymajor==True:
        R = np.maximum(R1,R2)*percentage
    else:
        R = np.minimum(R1,R2)*percentage
    
    # define mask used to exclude objects
    Mask = ((X!=-1)&(((X-R)<0)|((X+R)>xmax)|((Y-R)<0)|((Y+R)>ymax)))
    Mr[Mask] = 0
    for ifea in range(len(Fr)):
        Fr[ifea][Mask] = -1 
    Xr[Mask] = -1
    Yr[Mask] = -1
    
    # # major mask
    # R = R2 * percentage
    # Mask2 = ((X!=-1)&(((X-R)<0)|((X+R)>xmax)|((Y-R)<0)|((Y+R)>ymax)))
    # Mask = (Mask2*1. - Mask*1.)>0
    # Mr[Mask] = 0
    # for ifea in range(len(Fr)):
    #     Fr[ifea][Mask] = -1 
    # Xr[Mask] = -1
    # Yr[Mask] = -1
    
    return (Mr, Fr, Xr, Yr)

# def remove_intersect_border_by_circle_old(M,F,X,Y,R1,R2,xmax,ymax,percentage=0.8):
#     """Remove tracked positions intersecting the image border. Only the positions
#     touching the border but not entire track are excluded. The border-touching object 
#     is determined by checking the circle whose radius defined by the major axis.
    
#     Work for CellProfiler >= 3.x.
    
#     Args:
#         M (2D numpy matrix [float]): Transistion table.
#         F (List of 2D matrices [float]): List of corresponding feature table.
#         X, Y, R1, R2 (2D matrix [float]): x, y positions, minor and major radii of nucleus
#         xmax, ymax (float): movie width and height.
#         percentage (float):  fraction of radius used to compute circle, smaller value gives smaller circle for detecting border-touching.
        
#     Returns:
#         Feature matrices, center coordinates of tracks after border-touching removing.
    
#     """
#     Mr = M.copy()
#     Fr = [FF.copy() for FF in F]
#     Xr,Yr = X.copy(),Y.copy()
    
#     R = np.maximum(R1,R2)*percentage
#     Mask = ((X!=-1)&(((X-R)<0)|((X+R)>xmax)|((Y-R)<0)|((Y+R)>ymax)))
    
#     Mr[Mask] = 0
#     for ifea in range(len(Fr)):
#         Fr[ifea][Mask] = -1
    
#     Xr[Mask] = -1
#     Yr[Mask] = -1
    
#     return (Mr, Fr, Xr, Yr)

def remove_intersect_border_by_bbox(M,F,X,Y,Xbmin,Xbmax,Ybmin,Ybmax,xmax,ymax):
    """Removing tracked positions intersecting the image border. Only the positions
    touching the border but not entire track are excluded. The border-touching object 
    is determined by checking the bounding box.
    
    Work for CellProfiler 4.x.
    
    Args:
        M (2D numpy matrix [float]): Transistion table.
        F (List of 2D numpy matrices [float]): List of feature table.
        X, Y (2D numpy matrix [float]): object centers.
        Xbmin,Xbmax,Ybmin,Ybmax (float): bounding box min max coordinates.
        xmax, ymax (float): movie width and height.
        
    Returns:
        Feature matrices, center coordinates of tracks after border-touching removing.
    
    """
    Mr = M.copy()
    Fr = [FF.copy() for FF in F]
    Xr,Yr = X.copy(),Y.copy()

    Mask = ((X!=-1)&((Xbmin<=0)|(Xbmax>=xmax)|(Ybmin<=0)|(Ybmax>=ymax)))
    
    Mr[Mask] = 0
    for ifea in range(len(Fr)):
        Fr[ifea][Mask] = -1
    
    Xr[Mask] = -1
    Yr[Mask] = -1
    
    return (Mr, Fr, Xr, Yr)

def correct_transistion_row(row,G,statelbl):
    """Remove violated transistions from a given row.

    This function is used as a support function for remove_transistions(), which remove violated states
    for all tracks organized in matrix.

    Args:
        row (array of float): list of transistion states.
        G (networkx graph): transistion graph.
        statelbl (list of string): labels of transistion states.

    Return:
        list of transistion states in which the violated ones are marked as "0".

    TODO: exclude "0" group from the checking process.

    """
        
    # Compute super state ID and its length
    # Example: 
    # row = [1,1,1,1,2,2,2,1,1] 
    # super state id = [1,2,1]
    # state len = [4,3,2]
    icol = 0
    state_len = []
    sup_state_id = []
    count = 1
    while (icol < len(row)):
        if((icol)==(len(row)-1)):
            state_len.append(count)
            sup_state_id.append(int(row[icol]))
        elif row[icol] == row[icol+1]:
            count += 1
        else:
            state_len.append(count)
            sup_state_id.append(int(row[icol]))
            count = 1
        icol += 1
        
    # Initialize exclusion flag, set False as default (no states to be excluded)
    is_excluded = [False for _ in sup_state_id]
    
    # Iteratively check each item in super state ID and exclude the violated one by marking the is_excluded flag
    iprev, icrr, inext = -1, 0, 1 # previous, current and next indicators
    while((icrr < (len(sup_state_id)-1)) & (inext < len(sup_state_id))):

        # Compute previous state        
        if iprev == -1:
            prev_state_id = 0
            prev_state_len = 0
        else:
            prev_state_id = sup_state_id[iprev]
            prev_state_len = state_len[iprev]
        
        # Compute current state        
        curr_state_id = sup_state_id[icrr]
        curr_state_len = state_len[icrr]
        
        # Compute next state        
        next_state_id = sup_state_id[inext]
        next_state_len = state_len[inext]
        
        # print("previous current state:",prev_state_id,prev_state_len)
        # print("current state:",curr_state_id,curr_state_len)
        # print("next state:",next_state_id,next_state_len)
        
        # print(curr_state_id)
        # print(next_state_id)
        
        # If there is transistion between curr and next
        if(G.has_edge(statelbl[curr_state_id],statelbl[next_state_id])==True):
            iprev = icrr
            icrr = inext
            inext += 1
        else:
            # If there is no transistion between prev and next
            if(G.has_edge(statelbl[prev_state_id],statelbl[next_state_id])==False):
                is_excluded[inext] = True
                inext += 1
            else:
                # Compute score for curr and next
                # The one with lower score will be excluded
                curr_state_score = prev_state_len + curr_state_len
                next_state_score = prev_state_len + next_state_len
                
                # print(curr_state_score)
                # print(next_state_score)
                
                if curr_state_score < next_state_score:
                    is_excluded[icrr] = True
                    icrr = inext
                    inext += 1
                elif next_state_score < curr_state_score:
                    is_excluded[inext] = True
                    inext += 1
                else:
                    # If the scores are equal
                    # Then compute the number of steps need to go from a transistion to another.
                    # The one with higher number of steps will be excluded
                    curr_path_len = nx.shortest_path_length(G,statelbl[prev_state_id],statelbl[curr_state_id])
                    next_path_len = nx.shortest_path_length(G,statelbl[prev_state_id],statelbl[next_state_id])
                    if curr_path_len <= next_path_len:
                        is_excluded[inext] = True
                        inext += 1
                    else:
                        is_excluded[icrr] = True
                        icrr = inext
                        inext += 1
                
    # print(row)
    
    # Assign excluded states by "0"
    row_corrected = []
    for item in zip(sup_state_id,state_len,is_excluded):
        if item[2] == False:
            row_corrected += [item[0] for _ in range(item[1])]
        else:
            row_corrected += [0 for _ in range(item[1])]
    # print(row_corrected)
                
    # print(sup_state_id)
    # print(state_len)
    # print(is_excluded)
    
    return row_corrected

def remove_transistions(M,F,G,statelbl):
    """Remove transistions violate the transistion rules defining in graph G.
    
    Args:
        M (2D numpy matrix [float]): Transistion table.
        F (List of 2D numpy matrices [float]): List of feature table.
        G (networkx graph [networkx]): graph defines the transistion rules.
        statelbl (list [str]): transistion state names.
        
    Returns:
        Feature matrices of tracks after removing violated transistions.
    
    """

    # Initialize state transistion matrix Mr and its related featured matrices Fr
    Mr = []
    Fr = [[] for _ in F]
    
    # Exclude violated states for every track
    for irow in range(M.shape[0]):
        row = M[irow].copy()
        row_sup = [FF[irow].copy() for FF in F]
        row_corrected = correct_transistion_row(row, G, statelbl)
        for i in range(len(row_corrected)):
            if row_corrected[i] == 0:
               for irowsup in range(len(row_sup)):
                    row_sup[irowsup][i]=-1
        
        Mr.append(row_corrected)
        for ifea in range(len(Fr)):
            Fr[ifea].append(row_sup[ifea])
            
    Mr = np.array(Mr)
    Fr = [np.array(FF) for FF in Fr]
    return (Mr,Fr)

def remove_transistions_old(M,F,G,statelbl):
    """Remove transistions violate the transistion rules defining in graph G.

    This function is outdate. Please use remove_transistions().
    
    Args:
        M (2D numpy matrix [float]): Transistion table.
        F (List of 2D numpy matrices [float]): List of feature table.
        G (networkx graph [networkx]): graph defines the transistion rules.
        statelbl (list [str]): transistion state names.
        
    Returns:
        Feature matrices of tracks after removing violated transistions.
    
    """
    Mr = []
    Fr = [[] for _ in F]
    for irow in range(M.shape[0]):
        row = M[irow].copy()
        row_sup = [FF[irow].copy() for FF in F]
        
        icol = 0
        curr_state = int(row[icol])
        while icol<M.shape[1]-1:
            next_state = int(row[icol+1])
            if G.has_edge(statelbl[curr_state],statelbl[next_state])==False:
                row[icol+1]=0
                for irowsup in range(len(row_sup)):
                    row_sup[irowsup][icol+1]=-1
            else:
                curr_state = next_state
            icol += 1
        
        Mr.append(row)
        for ifea in range(len(Fr)):
            Fr[ifea].append(row_sup[ifea])
    
    Mr = np.array(Mr)
    Fr = [np.array(FF) for FF in Fr]
    return (Mr,Fr)
    
    
def align_time_points(Mr,Fr,state_numbers=[4,5,6],align_modes=["last","first","first"],shifts=[0,1,1]):
    """Align track objects based on referenced states given in state_numbers.
    
    The first state in state_numbers will be checked. If that state does not exist,
    then check the second state from the list and so on.
    
    Args:
        Mr (2D numpy matrix [float]): State transistion matrix
        Fr (List of 2D numpy matrices [float]): Corresponding features matrices.
        state_numbers (list [int]): list of referenced transistions.
        align_modes (list [str]): list of align mode: 
            if "first", then align based on the first time point of the corresponding transisition,
            otherwise, align based on the last time point.
        shifts (list [int]): offset to be added.
        
    Returns:
       Aligned time points, corresponding states and features. 
    
    """
    
    def align_row(row, state_number=4, mode="last",shift=0):
        """Supp. func: align a row (track object) in Mr
        """
        meta_flag = (row==state_number) # state bool flag
        meta_components = measure.label(meta_flag) # distinct subintervals (or components)
        nb_components = len(np.unique(meta_components))-1
        ref = []
        if nb_components > 0:
            prev_ix = 0
            for ic in np.arange(1,nb_components+1):
                if mode=="last":
                    curr_ix = np.argwhere(meta_components==ic).flatten()[-1] # last index in component
                elif mode=="first":
                    curr_ix = np.argwhere(meta_components==ic).flatten()[0] # first index in component
                else:
                    raise Exception("support only 'first' or 'last' for mode")
                ref += np.arange(-(curr_ix-prev_ix),1,1).tolist()
                next_ix = np.argwhere(meta_components==(ic+1)).flatten()
                if len(next_ix)!=0:
                    if mode=="last":
                        ix = next_ix[-1] - curr_ix
                        ref += np.arange(1,ix,1).tolist()
                        prev_ix = next_ix[-1]
                    elif mode=="first":
                        ix = next_ix[0] - curr_ix
                        ref += np.arange(1,ix,1).tolist()
                        prev_ix = next_ix[0]
                    else:
                        raise Exception("support only 'first' or 'last' for mode")
            if len(ref)<len(meta_components):
                ref += np.arange(1,len(meta_components)-len(ref)+1,1).tolist()
        
        if shift != 0:
            ref = [(val + shift) for val in ref]
        
        return ref
    
    ref_time, nucleus_state = [],[]
    feature_cols = [[] for _ in Fr]
    for irow in range(Mr.shape[0]):
        row = Mr[irow].copy()
        
        # aligning...
        ref = []
        istate = 0
        while(len(ref)==0):
            if istate == len(state_numbers):
                break
            
            ref = align_row(row,state_number=state_numbers[istate],
                            mode=align_modes[istate],
                            shift=shifts[istate])
            istate += 1
        
        if len(ref)!=0:
            ref_time += ref
            nucleus_state += row.tolist()
            for ifea in range(len(feature_cols)):
                feature_cols[ifea] += Fr[ifea][irow].tolist()
       
    return (ref_time, nucleus_state, feature_cols)

def process_data(input_file,output_path,features,transition_graph,
                 nrows_limit=30,min_nb_timepoints=5,
                 exclude_borderobjs_conds = {"criterion":"circle","percentage":0.8},
                 align_conds={"state_numbers":[4,5,6],"align_modes":["last","first","first"],"shifts":[0,1,1]}):
    """Process tracked data from CellProfiler.
    
    Args:
        input_file (str): input csv file.
        output_path (str): output directory.
        features (list (str)): list of features to be aligned.
        transition_graph (networkx): graph defining transistion rules.
        nrows_limit (uint): maximal row for each plot. Each row shows a track object from a movie. Movie plot is divided into smaller subplots by nrows_limits.
        min_nb_timepoints (uint): minimal number of timepoints need for each track.
        exclude_borderobjs_conds (dict): conditions are applied to exclude objects touching image border (see remove_intersect_border() for details).
        align_conds (dict): conditions are applied to align time points (see align_time_points() for details)
    
    Returns:
        Table of aligned tracks and corresponding features. Intermediate results are plotted and saved.
    
    """
    
    # load data
    data = pd.read_csv(input_file)
    
    # remove usefuless columns and rows
    try:
        # drop useless columns
        dropdata = data.drop(['Metadata_MovieName.1',
                   'Metadata_MovieName.2',
                   'Metadata_T.1', 
                   'Metadata_C', 
                   'Metadata_ChannelName', 
                   'Metadata_Frame', 
                   'Metadata_Plate', 
                   'Metadata_Site', 
                   'Metadata_Well',
                   'Metadata_ColorFormat', 
                   'AreaShape_EulerNumber', 
                   'Location_CenterMassIntensity_Z_H2B_Smooth', 
                   'Location_CenterMassIntensity_Z_TRF1_Smooth',
                   'Location_MaxIntensity_Z_H2B_Smooth', 
                   'Location_MaxIntensity_Z_TRF1_Smooth',
                   'Parent_Nuclei_TBC'], 
                  axis='columns',
                  inplace=False)
        
        # drop columns containing texts
        dropdata.drop(dropdata.filter(regex='(Z|Kalman|RadialDistribution|Granularity)').columns,
                  axis='columns',
                  inplace=True)
    except:
        dropdata = None
        
    if dropdata is not None:
        data = dropdata
    
    # drop columns with all NaN
    data.dropna(axis='columns',how='all',inplace=True)
    
    G = transition_graph.copy()
    
    # define TrackObjects_Type
    try:
        # get transistion columns from transistion graph
        transistion_columns = ["Children_{}_Count".format(item) for item in list(G.nodes)]
        
        # encode columns to uint number (power of 2)
        # NOTE: work for uint8 (8 labels max) for instant, can be updated for higher labels further.
        # NOTE: the order of columns can change the encoding number => BE CAREFULL
        cycle_bit = np.packbits(data[transistion_columns].values,axis=1,bitorder="little").flatten()
        
        # temporarily replace 0 by a positive number to avoid error when using logarithm
        cycle_bit[cycle_bit==0]=2**len(transistion_columns) 
        
        cycle_number = np.log2(cycle_bit)+1 # make number to be continuous, e.g. 1, 2, 3, ...
        
        # convert replaced number back to 0 (since we modifed it before)
        cycle_number[cycle_number==len(transistion_columns)+1]=0
        
#         print(data[transistion_columns].values)
#         print(cycle_number)
    except:
#         print("columns not found from csv.")
        return # the input file doesn't contain required columns
    
    # define transistion labels
    cycle_labels = ["unknown"] + list(G.nodes)
    cycle_col = [cycle_labels[int(i)] for i in cycle_number]
    
    # define colormap
    mycmap = plt.get_cmap('Set1_r', len(cycle_labels)) # seven color codes for state transistion
    # norm = mpl.colors.Normalize(vmin=-0.5, vmax=len(cycle_labels)-0.5)

    # adding state transistion columns
    data['TrackObjects_StateLabel'] = cycle_col
    data['TrackObjects_StateNumber'] = cycle_number
    
    # add "unknown" state to G
    G.add_node("unknown")
    G.add_edge("unknown","unknown");
    G.add_edges_from([("unknown",item) for item in list(G.nodes)]);
    G.add_edges_from([(item,"unknown") for item in list(G.nodes)]);
    
    # remove TrackObjects_Label with NaN
    data.dropna(subset=["TrackObjects_Label"],inplace=True)
    
    # remove rows with NaN in feature columns
    data.dropna(subset=features,inplace=True)
    
    # check and create folders
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # first level folders
    # Processed_CSV: cleaned dataset
    # Analysed_CSV: aligned dataset
    # Figures: plots at intermediate steps
    # Transistion_Matrix: numpy matrices o track state transistions
    new_dir = ["Processed_CSV","Analysed_CSV","Figures","Transistion_Matrix"]  
    for idir in new_dir:
        subpath = os.path.join(output_path,idir)
        if not os.path.exists(subpath):
            os.makedirs(subpath)
    
    # subfolders of Figures, organized by features
    # Table: matrix plot
    # RefTime: line plot over referenced time
    # Transistion: boxplot over transistion states
    fig_subdir = ["State_Transistion"] + features
    fig_ssubdir = ["Table","RefTime","Transistion"]
    for idir in fig_subdir: 
        subpath = os.path.join(output_path,new_dir[2],idir)
        if not os.path.exists(subpath):
            os.makedirs(subpath)
            
        if idir=="State_Transistion":
            for iidir in fig_ssubdir[:-1]: # don't need Transistion folder
                ssubpath = os.path.join(output_path,new_dir[2],idir,iidir)
                if not os.path.exists(ssubpath):
                    os.makedirs(ssubpath)
        else:
            for iidir in fig_ssubdir:
                ssubpath = os.path.join(output_path,new_dir[2],idir,iidir)
                if not os.path.exists(ssubpath):
                    os.makedirs(ssubpath)
    
    # getting movies
    movies = list(data['Metadata_MovieName'].unique())
    
    # getting data for each movie
    for imovie in range(len(movies)):
        
        # image size
        xmax = data[data["Metadata_MovieName"]==movies[imovie]]["Metadata_SizeX"].unique()[0] - 1
        ymax = data[data["Metadata_MovieName"]==movies[imovie]]["Metadata_SizeY"].unique()[0] - 1
        
        # getting data for the movie
        subdata = data[data['Metadata_MovieName']==movies[imovie]].copy()
        
        # substrating image number offset
        image_number_offset = subdata["ImageNumber"].min()
        subdata["ImageNumber"] = subdata["ImageNumber"] - image_number_offset + 1
        # setting table indices
        subdata.set_index(['ImageNumber','ObjectNumber'],inplace=True)
        # sort image number by ascending order
        subdata.sort_index(level="ImageNumber",inplace=True)
        
        print(image_number_offset)
                
        # save cleaned data
        subfile = os.path.join(output_path,new_dir[0],movies[imovie]+".csv")
        subdata.to_csv(subfile,index=True)
        
        trackinfo = {}
        trackinfo["Movie"] = str(movies[imovie])
        
        # list of track object labels
        trackobjs_label = subdata['TrackObjects_Label'].unique().astype(np.integer)
        
        # save info
        trackinfo["total_tracks"] = list(trackobjs_label)
        
        # list of image numbers
        imgnum = subdata.index.get_level_values('ImageNumber').unique().astype(np.integer)
        
        # initialize state transistion matrix
        M = np.zeros((len(trackobjs_label),max(imgnum)))
        
        # initialize nucleus area features (used to exlude touched border nuclei)
        X = np.ones((len(trackobjs_label),max(imgnum)))*-1
        Y = X.copy()
        R1, R2 = X.copy(), X.copy() # minor, major axis
        Xbmin, Xbmax, Ybmin, Ybmax = X.copy(), X.copy(), X.copy(), X.copy() # bounding box
        
        # initialize others features matrix
        F = []
        for _ in features:
            F.append(np.ones((len(trackobjs_label),max(imgnum)))*-1)
        
        # initialize referenced object number matrix
        O = np.zeros((len(trackobjs_label),max(imgnum)))
       
        # build track matrices
        for itrack in range(len(trackobjs_label)):
            trackobj = trackobjs_label[itrack]
            tmpdf = subdata[subdata['TrackObjects_Label']==trackobj].copy()
            
            obj_imgnum = tmpdf.index.get_level_values("ImageNumber").values
            obj_objnum = tmpdf.index.get_level_values("ObjectNumber").values
            
            if len(obj_imgnum)>=min_nb_timepoints: # only consider track with minimal number of timepoints
            
                O[itrack,obj_imgnum-1] = obj_objnum

                obj_state = tmpdf["TrackObjects_StateNumber"].values
                M[itrack,obj_imgnum-1] = np.floor(obj_state)

                X[itrack,obj_imgnum-1] = tmpdf["AreaShape_Center_X"].values
                Y[itrack,obj_imgnum-1] = tmpdf["AreaShape_Center_Y"].values
                
                if exclude_borderobjs_conds["criterion"]=="circle":
                    R1[itrack,obj_imgnum-1] = tmpdf["AreaShape_MinorAxisLength"].values / 2.
                    R2[itrack,obj_imgnum-1] = tmpdf["AreaShape_MajorAxisLength"].values / 2.
                else:
                    Xbmin[itrack,obj_imgnum-1] = tmpdf["AreaShape_BoundingBoxMinimum_X"].values
                    Xbmax[itrack,obj_imgnum-1] = tmpdf["AreaShape_BoundingBoxMaximum_X"].values
                    Ybmin[itrack,obj_imgnum-1] = tmpdf["AreaShape_BoundingBoxMinimum_Y"].values
                    Ybmax[itrack,obj_imgnum-1] = tmpdf["AreaShape_BoundingBoxMaximum_Y"].values

                tmpdf.reset_index(inplace=True) # transform indices to columns
                for ifea in range(len(features)):
                    F[ifea][itrack,obj_imgnum-1] = tmpdf[features[ifea]].values
        
        # save referenced object number matrix
        odf = pd.DataFrame(O,columns=[str(item+1) for item in range(max(imgnum))])
        odf["TrackObjLabels"] = trackobjs_label
        odf.set_index("TrackObjLabels",inplace=True)
        odf.to_csv(os.path.join(output_path,"ref_obj_number.csv"))
        
        # plot feature table
        for ifea in range(len(fig_subdir)):
            file_pattr = os.path.join(output_path,new_dir[2],fig_subdir[ifea],fig_ssubdir[0],movies[imovie]+"_0raw")
            if ifea == 0:
                save_plot_matrix(file_pattr,M,nrows_limit,
                                 trackobjs_label,"Movie: "+str(movies[imovie]),
                                 cmap=mycmap,vmin=-0.5,vmax=len(cycle_labels)-0.5,cticklabel=cycle_labels)
            else:
#                 print(np.argwhere(np.isnan(F[ifea-1])))
                save_plot_matrix(file_pattr,F[ifea-1],nrows_limit,
                                 trackobjs_label,"Movie: "+str(movies[imovie]),
                                 cmap="jet",vmin=0)
    
        # save transistion matrix
        filename = os.path.join(output_path,new_dir[3],movies[imovie]+"_0raw.npy")
        np.save(filename,M)
        
        # remove border intersected objects
        if exclude_borderobjs_conds["criterion"]=="circle":
            percentage = exclude_borderobjs_conds["percentage"]
            bymajor = exclude_borderobjs_conds["bymajor"]
            Mb, Fb, Xb, Yb = remove_intersect_border_by_circle(M,F,X,Y,R1,R2,xmax,ymax,bymajor,percentage)
        else:
            Mb, Fb, Xb, Yb = remove_intersect_border_by_bbox(M,F,X,Y,Xbmin,Xbmax,Ybmin,Ybmax,xmax,ymax)
            
        # save info
        Mbsum = np.sum(Mb,axis=1)
        trackinfo["tracks_border_filtered"] = list(trackobjs_label[np.argwhere(Mbsum>0).flatten()])
        
        # plot feature table
        for ifea in range(len(fig_subdir)):
            file_pattr = os.path.join(output_path,new_dir[2],fig_subdir[ifea],fig_ssubdir[0],movies[imovie]+"_1border")
            if ifea == 0:
                save_plot_matrix(file_pattr,Mb,nrows_limit,
                                 trackobjs_label,"Movie: "+str(movies[imovie]),
                                 cmap=mycmap,vmin=-0.5,vmax=len(cycle_labels)-0.5,cticklabel=cycle_labels)
            else:
                save_plot_matrix(file_pattr,Fb[ifea-1],nrows_limit,
                                 trackobjs_label,"Movie: "+str(movies[imovie]),
                                 cmap="jet",vmin=0)
        
        # save transistion matrix
        filename = os.path.join(output_path,new_dir[3],movies[imovie]+"_1border.npy")
        np.save(filename,Mb)
        
        # remove violated transistions
        Mr, Fr = remove_transistions(Mb,Fb,G,cycle_labels)
        
        # save info
        Mrsum = np.sum(Mr,axis=1)
        trackinfo["tracks_transistion_filtered"] = list(trackobjs_label[np.argwhere(Mrsum>0).flatten()])
        
        # plot feature table
        for ifea in range(len(fig_subdir)):
            file_pattr = os.path.join(output_path,new_dir[2],fig_subdir[ifea],fig_ssubdir[0],movies[imovie]+"_2rule")
            if ifea == 0:
                save_plot_matrix(file_pattr,Mr,nrows_limit,
                                 trackobjs_label,"Movie: "+str(movies[imovie]),
                                 cmap=mycmap,vmin=-0.5,vmax=len(cycle_labels)-0.5,cticklabel=cycle_labels)
            else:
                save_plot_matrix(file_pattr,Fr[ifea-1],nrows_limit,
                                 trackobjs_label,"Movie: "+str(movies[imovie]),
                                 cmap="jet",vmin=0)
                
        # save transistion matrix
        filename = os.path.join(output_path,new_dir[3],movies[imovie]+"_2rule.npy")
        np.save(filename,Mr)
        
        # align time point according to the last metaphase state or first ana/telophase state
        ref_time, nucleus_state, feature_cols = align_time_points(Mr,Fr,**align_conds)
                
        # building dataframe
        dics = {}
        dics["Nucleus_State_Number"] = nucleus_state
        dics["Reference_Time"] = np.array(ref_time)*6./60. # convert to hours
        for ifea in range(len(features)):
            dics[features[ifea]] = feature_cols[ifea]
        
        dfref = pd.DataFrame(dics)
        
        # remove unvalid rows
        dfref.drop(dfref[dfref["Nucleus_State_Number"]==0].index,inplace=True)
        for fea in features:
            dfref.drop(dfref[dfref[fea]==-1].index,inplace=True)
        
        dfref["Nucleus_State_Name"]=[cycle_labels[int(i)] for i in dfref["Nucleus_State_Number"]]
        
        # save dataframe into file
        subfile = os.path.join(output_path,new_dir[1],movies[imovie]+".csv")
        dfref.to_csv(subfile,index=False)
        
        # save info
        trackinfo["tracks_time_aligned"] = list(dfref["TrackObjects_Label"].unique())
        
        # save to file
        np.save(os.path.join(output_path,"trackinfo.npy"),trackinfo)
        
        # ref time plots
        for ifea in range(len(fig_subdir)):
            filename = os.path.join(output_path,new_dir[2],fig_subdir[ifea],fig_ssubdir[1],movies[imovie]+".png")
            if ifea==0:
                ylabel = "Nucleus_State_Number"
            else:
                ylabel = features[ifea-1]
            
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
            sns.lineplot(x="Reference_Time", y=ylabel, data=dfref);
            ax.set_title("Movie:"+movies[imovie]);
            fig.savefig(filename,dpi=150)
            plt.close(fig)
        
        # boxplot plots
        sns_accent = np.array(sns.color_palette("Accent"))[range(len(cycle_labels)-1)] # color pallete
        for ifea in range(1,len(fig_subdir)):
            filename = os.path.join(output_path,new_dir[2],fig_subdir[ifea],fig_ssubdir[2],movies[imovie]+".png")
            ylabel = features[ifea-1]
            
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(111)
            sns.boxplot(x="Nucleus_State_Name", y=ylabel, data=dfref, order=cycle_labels[1:],palette=sns_accent);
            ax.set_title("Movie:"+movies[imovie]);
            fig.savefig(filename,dpi=150)
            plt.close(fig)
        
#         # save XY track path plot
#         filename = os.path.join(output_path,new_dir[2],subnew_dir[7],movies[imovie]+".png") 
#         save_plot_xy_track(filename,subdata,"Movie: "+movies[imovie],xmax,ymax,
#                            cmap=mycmap,vmin=-0.5,vmax=6.5,cticklabel=cycle_labels,norm=norm)
        
#         # save XY track path plot
#         filename = os.path.join(output_path,new_dir[2],subnew_dir[8],movies[imovie]+".png") 
#         save_plot_xy_track_exlu(filename,Xb,Yb,"Movie: "+movies[imovie],xmax,ymax)
        




































