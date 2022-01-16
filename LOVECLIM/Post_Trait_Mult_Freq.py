# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 14:18:51 2021

@author: Antoine
"""

## Load functions
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
#NEEDED FOR BASEMAP, MAY BE UNNECESARY DEPENDING ON SETUP, COULD COMMENT
#SOL FROM STACKOVERFLOW https://stackoverflow.com/questions/52295117/basemap-import-error-in-pycharm-keyerror-proj-lib
os.environ["PROJ_LIB"] = "C:\\Users\\Antoine\\Anaconda3\\Library\\share";
from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
import math as m
import pandas as pd
import sys

########################################################
####################USEFUL FUNCTIONS####################
########################################################

#
# Function to load NetCDF file in a convenient way
# From F. Massonnet
#
'''
pre:
    fileon: A netCDF file adress
    variables: A variable or list of variables to extract
returns:
    The variable list time series
'''
def ncload(fileon, variables = None):
  
  #MODIFIED FILEIN HERE, DONE IN ORDER TO GO INTO THE DATA DIRECTORY
  filein="data/"+fileon    
  
  try:
    f = Dataset(filein, mode = "r")
  except IOError:
    sys.exit("(ncload) ERROR: File not found:\n >>> " + (filein) + " <<<")

  # List of variables
  if variables is None:
    variables = [vv.encode('ascii') for vv in f.variables]
  else:
    if type(variables) is not list:
      if type(variables) is str: # convert to list if only one variable given
        variables = [variables]
      else:
        sys.exit("(ncload) ERROR: input variables not provided as a list")

  print("(ncload) -- File: " + filein )  
  for vv in variables:
    print("(ncload) -- loading " + str(vv).ljust(20) )
    exec("try:\n"                        +\
         "  global " + str(vv) + "\n"     +\
         "  " + str(vv) + " = np.squeeze(f.variables[\"" + vv + "\"][:])\n" \
         "except KeyError:\n"            +\
         "  sys.exit(\"(ncload) ERROR: variable \" + str(vv) + \" of file \" + filein + \" does not exist in NetCDF file\")"                              )
    
  f.close()

  to_print = ""
  for numbers in range(len(variables)):
    str_1 = "eval(variables["
    if numbers<len(variables)-1:
      str_2 = "]),"
    else: 
      str_2 = "])"
    to_print += (str_1 + str(numbers) + str_2)
  
  return eval(to_print)


# Function to compute average over space (taking into account 
# the weights)
'''
pre:
    data: data grid over which to compute the average
    weights: weight grid for the averages (by the area usually)
    lat: mapping of data grid to lat grid
    lon: mapping of data grid to lon grid
returns:
    time_series: the weighted time series
'''
def mean_weighted_area(data, weights, lat, lon, lat_min, lat_max, lon_min, lon_max):
    """
    Returns a time series from a 3D array [time, lat, lon].
    
    """
    if lat.ndim==1:
        lon, lat = np.meshgrid(lon, lat)
    
    # Mask nans
    data_mask_nan = np.ma.MaskedArray(data, mask=np.logical_or(np.isnan(data),data==-99.99))

    # Create weights
    weights = np.cos(np.pi*lat/180)
    weights = weights
    weights_rp = np.repeat(weights[np.newaxis,...], data.shape[0], axis=0)

    # Create mask coordinates
    mask = ((lat>lat_max) | (lat<lat_min) | (lon<lon_min) | (lon>lon_max))
    mask_rp = np.repeat(mask[np.newaxis,...], data.shape[0], axis=0)

    # mask data and weights
    data_mask = ma.masked_array(data_mask_nan, mask_rp)
    weights_rp = ma.masked_array(weights_rp, mask_rp)

    time_series = np.average(data_mask, axis=(1, 2), weights=weights_rp)

    return time_series

# Function to compute average over area no time component (taking into account 
# the weights)
'''
pre:
    data: data grid over which to compute the average
    weights: weight grid for the averages (by the area usually)
    mask: mask in order to limit the things we take into account
returns:
    A scalar average over time and space
'''
def mean_target_mask(data, mask, weights):
    
    # mask data and weights
    data_masked = ma.masked_array(data, mask)
    weights_masked = ma.masked_array(weights, mask)
    
    return np.average(data_masked,weights=weights_masked)

# Function to compute average over area given mask
    '''
pre:
    data: data grid over which to compute the average
    weights: weight grid for the averages (by the area usually)
    mask: mask in order to limit the things we take into account
returns:
    time_series: the average for each time
'''
def mean_target_mask_time(data, mask, weights):
    
    # mask data and weights
    data_masked = ma.masked_array(data, mask)
    weights_masked = ma.masked_array(weights, mask)
    
    return np.average(data_masked,weights=weights_masked, axis=(1,2))
    

#Calculates the correlation coefficient between time series
'''
pre:
    d: first grid time series
    d_t: second grid time series
returns:
    corr_tab: grid of correlations
'''
def corr_cells(d, d_t):
    s_d=np.shape(d)
    s_d_t=np.shape(d_t)
    if s_d!=s_d_t:
        sys.exit("both tables dont have the same shape"+ str(s_d)+ " vs "+ str(s_d_t))
    corr_tab=np.zeros(s_d[1:])
    for i in range(s_d[1]):
        for j in range(s_d[2]):
            corr_matrix=np.corrcoef(d[:,i,j],d_t[:,i,j])#calculates correlation matrix between time series
            corr_tab[i,j]=corr_matrix[0,1]#takes only the correlations of interest
    return corr_tab

#Calculates the RMSE in each cell between time series
'''
pre:
    d: first grid time series
    d_t: second grid time series
returns:
    RMSE_tab: grid of RMSE
'''
def RMSE_cells(d, d_t):
    s_d=np.shape(d)
    s_d_t=np.shape(d_t)
    if s_d!=s_d_t:
        sys.exit("both tables dont have the same shape"+ str(s_d)+ " vs "+ str(s_d_t))
    RMSE_tab=np.sqrt(np.mean((d-d_t)**2, axis=0))
    return RMSE_tab

#Calculates the CE in each cell between time series
'''
pre:
    d: first grid time series
    d_t: second grid time series
returns:
    CE_tab: grid of CE
'''
def CE_cells(d, d_t):
    s_d=np.shape(d)
    s_d_t=np.shape(d_t)
    if s_d!=s_d_t:
        sys.exit("both tables dont have the same shape"+ str(s_d)+ " vs "+ str(s_d_t))
    CE_tab=1-np.sum((d-d_t)**2, axis=0)/np.sum((d_t-np.mean(d_t,axis=0))**2, axis=0)
    return CE_tab
    


#Function which takes monthly averages from 0 to 2000 AC to yearly from 1000 to 1200AC
def return_to_average_200(arr):
    to_average=arr[12000:14400,:,:]
    reshaped=np.reshape(to_average,(200,12,np.shape(arr)[1],np.shape(arr)[2]))
    return np.mean(reshaped,axis=1)

#makes a grid from a full set of lons, useful for ts grid
def make_lon_grid(lons, lats):
    nb_lon=len(lons)
    nb_lat=len(lats)
    
    step_len=lons[3]-lons[2]
    start_lon=lons[0]-step_len/2
    end_lon=lons[nb_lon-1]+3*step_len/2
    
    mid_section_points=np.arange(start_lon,end_lon,step_len)
    
    return np.tile(mid_section_points, (nb_lat+1,1))

#makes a grid from a full set of lats, useful for ts grid
def make_lat_grid(lats, lons):
    nb_lon=len(lons)
    nb_lat=len(lats)
    
    #Max 90 and min -90
    max_lat=90
    min_lat=-90
    
    #Calculate the midpoints
    mid_lats=(lats[1:]+lats[:(nb_lat-1)])/2
    mid_lats=np.append([min_lat],mid_lats)
    mid_lats=np.append(mid_lats,[max_lat])
    
    return np.tile(mid_lats,(nb_lon+1,1)).T

#Function which takes monthly averages to yearly averages
def make_yearly(ts):
    return np.mean(np.reshape(ts,(int(np.shape(ts)[0]/12),12,np.shape(ts)[1],np.shape(ts)[2])), axis=1)#get to annual from monthly

###############GET TABLE COORDINATES AND GRIDS#####################

#Check if there is a data file
if not os.path.isdir('data'):
        exit('NO data file found, please refer to the readme for information on how to acquire the data file')
        
#Ocean sst grid load
file_ocean_grid_ref='base_sst.nc' 
sst_lon_grid, sst_lat_grid, sst_lon_coords, sst_lat_coords, sst_area = ncload(file_ocean_grid_ref, ['tlonp','tlatp','tlon','tlat','area'])

#Surface temperature grid load and treatment
file_surface_grid_ref='base_ts.nc'
ts_lon_list, ts_lat_list = ncload(file_surface_grid_ref, ['lon','lat'])
ts_lon_coords=np.tile(ts_lon_list,(len(ts_lat_list),1)) #tesselate
ts_lat_coords=np.tile(ts_lat_list,(len(ts_lon_list),1)).T #tesselate
ts_lon_grid=make_lon_grid(ts_lon_list, ts_lat_list)
ts_lat_grid=make_lat_grid(ts_lat_list, ts_lon_list)
ts_area=np.cos(np.pi*ts_lat_coords/180)

#Ocean proxy indices load and treatment
prox_rms_sst=ncload('sst_pseudo_modif.nc', 'rms')
p_sst=np.array(np.where(prox_rms_sst!=-99.99))#prox_coords
p_sst_lons=sst_lon_coords[p_sst[0],p_sst[1]].data
p_sst_lons=np.mod(p_sst_lons,360)
p_sst_lats=sst_lat_coords[p_sst[0],p_sst[1]].data

#Land proxies indices load and treatment
prox_rms_ts=ncload('ts_pseudo_modif.nc', 'rms')
p_ts=np.array(np.where(prox_rms_ts!=-99.99))#prox_coords
p_ts_lons=ts_lon_coords[p_ts[0],p_ts[1]].data
p_ts_lats=ts_lat_coords[p_ts[0],p_ts[1]].data

#Reference files for sst on which pseudoproxies were made load and treatment
file_ref_sst = 'reference_sst.nc'
sst_ref_unprocessed = ncload(file_ref_sst, 'sst')
sst_ref=return_to_average_200(sst_ref_unprocessed)
sst_ref_10=np.mean(np.reshape(sst_ref,tuple(np.append([20,10],list(np.shape(sst_ref)[1:])))),axis=1)#over 10 yrs average

#Reference files for ts on which pseudoproxies were made load and treatment
file_ref_ts = 'reference_ts.nc'
ts_ref_unprocessed_kelvin = ncload(file_ref_ts, 'ts')
ts_ref_kelvin = return_to_average_200(ts_ref_unprocessed_kelvin)
ts_ref = ts_ref_kelvin-np.ones(np.shape(ts_ref_kelvin))*273.15
ts_ref_10=np.mean(np.reshape(ts_ref,tuple(np.append([20,10],list(np.shape(ts_ref)[1:])))),axis=1)#over 10 yrs average

#Pseudoproxy files for sst and ts
file_pseudo_sst= 'pseudodata_sst_modif.nc'
sst_pseudo_unprocessed=ncload(file_pseudo_sst, 'sst')
sst_pseudo=make_yearly(sst_pseudo_unprocessed)
sst_pseudo_10=np.mean(np.reshape(sst_pseudo,tuple(np.append([20,10],list(np.shape(sst_pseudo)[1:])))),axis=1)#over 10 yrs average

file_pseudo_ts = 'pseudodata_ts_modif.nc'
ts_pseudo_unprocessed_kelvin = ncload(file_pseudo_ts, 'ts')
ts_pseudo_kelvin = make_yearly(ts_pseudo_unprocessed_kelvin)
ts_pseudo = ts_pseudo_kelvin-np.ones(np.shape(ts_pseudo_kelvin))*273.15
ts_pseudo_10=np.mean(np.reshape(ts_pseudo,tuple(np.append([20,10],list(np.shape(ts_pseudo)[1:])))),axis=1)#over 10 yrs average
###############################################################
#############MAKE MASKS FOR CORRELATIONS AVERAGING#############
###############################################################

#Getting masks for sst points
sst_mask=np.ones(np.shape(sst_lon_coords), dtype=bool)
sst_mask[p_sst[0],p_sst[1]]=False

#Getting masks for ts points
ts_mask=np.ones(np.shape(ts_lon_coords), dtype=bool)
ts_mask[p_ts[0],p_ts[1]]=False
    
#Getting masks for sst points in ts grid
sst_mask_tstab=np.ones(np.shape(ts_lon_coords), dtype=bool)
for i in range(len(p_sst[0])):
    lon=np.amax(np.where(p_sst_lons[i]>ts_lon_list))+1#Plus one to go from grid to coords table index
    lat=np.amax(np.where(p_sst_lats[i]>ts_lat_list))+1#Plus one to go from grid to coords table index
    sst_mask_tstab[lat, lon]=False

#Getting masks for ts and sst points in ts grid    
ts_sst_mask=(sst_mask_tstab & ts_mask)

#Getting zones for sst correlation calculations
#Zone around the proxies in sst grid
sst_max_lon_index=np.amax(p_sst[0])+1
sst_min_lon_index=np.amin(p_sst[0])
sst_max_lats_index=np.amax(p_sst[1])+1
sst_min_lats_index=np.amin(p_sst[1])
sst_zone_mask = np.ones(np.shape(sst_lon_coords),dtype=bool)
sst_zone_mask[sst_min_lon_index:sst_max_lon_index,sst_min_lats_index:sst_max_lats_index]=False

#In the northern hemisphere
sst_nh_mask=(sst_lat_coords<0)

#Getting zones for ts correlation calculations
#Zone of ts proxies, from ts grid
ts_max_lon_index=np.amax(p_ts[1])+1
ts_min_lon_index=np.amin(p_ts[1])
ts_max_lats_index=np.amax(p_ts[0])+1
ts_min_lats_index=np.amin(p_ts[0])
ts_zone_mask = np.ones(np.shape(ts_lon_coords),dtype=bool)
ts_zone_mask[ts_min_lats_index:ts_max_lats_index,ts_min_lon_index:ts_max_lon_index]=False

#Zone of sst proxies, from ts grid
#Shift everything in order to have a proper functionning
shift=70#Manual input, in order to have the proper zone selection
shift_cells=m.ceil(70*len(ts_lon_grid[0])/360)#Get the number of cells by which our perception is shifted
p_sst_lons_shifted=np.mod(p_sst_lons+np.ones(np.shape(p_sst_lons))*70,360)#Shift the proxy longitude accordingly

sst_max_lon_index_tstab=ts_lon_list[np.amin(np.where(np.amax(p_sst_lons_shifted)<ts_lon_grid[0]))-1]#Minus one to go from grid to coords table index
sst_min_lon_index_tstab=ts_lon_list[np.amax(np.where(np.amin(p_sst_lons_shifted)>ts_lon_grid[0]))]
sst_max_lats_index_tstab=ts_lat_list[np.amin(np.where(np.amax(p_sst_lats)<ts_lat_grid[:,0]))-1]#Minus one to go from grid to coords table index
sst_min_lats_index_tstab=ts_lat_list[np.amax(np.where(np.amin(p_sst_lats)>ts_lat_grid[:,0]))]
sst_zone_lats_tstab_shifted=((ts_lat_coords>sst_max_lats_index_tstab) | (ts_lat_coords<sst_min_lats_index_tstab))
sst_zone_lons_tstab_shifted=((ts_lon_coords<sst_min_lon_index_tstab) | (ts_lon_coords>sst_max_lon_index_tstab))
sst_zone_mask_tstab_shifted=(sst_zone_lats_tstab_shifted | sst_zone_lons_tstab_shifted)
sst_zone_mask_tstab=np.zeros(np.shape(sst_zone_mask_tstab_shifted),dtype=bool)
sst_zone_mask_tstab[:,(len(ts_lon_grid[0])-shift_cells):]=sst_zone_mask_tstab_shifted[:,:shift_cells-1]
sst_zone_mask_tstab[:,:(len(ts_lon_grid[0])-shift_cells)]=sst_zone_mask_tstab_shifted[:,shift_cells-1:]


#Combined zones for sst and ts in ts_grid
ts_sst_zone_mask=(sst_zone_mask_tstab & ts_zone_mask)

#Zone over northern hemisphere
ts_nh_mask=(ts_lat_coords<0)

#Order in which the masks are being applied for code execution
mask_order=["sst_mask","sst_zone_mask","sst_nh_mask","ts_mask","sst_mask_tstab"
            ,"ts_sst_mask","ts_zone_mask","sst_zone_mask_tstab","ts_sst_zone_mask",
            "ts_nh_mask"]
specific_masks_pseudo=["ts_mask","sst_mask"]

#Grids on which each mask applies
var_mask_order=["sst","sst","sst","ts","ts","ts","ts","ts","ts","ts"]

#MAKE RESULT DIRECTORIES if not there
result_dirs=['maps','time_series','bar_plots']
for directory in result_dirs:
    if not os.path.isdir(directory):
        os.mkdir(directory)


#for free reconstruction (takes an average from files from a list)
def make_free_sst(listing):
    sst_buffer=np.zeros(tuple(np.append(len(listing),list(np.shape(sst_ref)))))
    for i in range(len(listing)):
        unprocessed=ncload(listing[i], 'sst')
        sst_buffer[i]=return_to_average_200(unprocessed)
    return np.mean(sst_buffer,axis=0)

#for free reconstruction (takes an average from files from a list)
def make_free_ts(listing):
    ts_buffer=np.zeros(tuple(np.append(len(listing),list(np.shape(ts_ref)))))
    for i in range(len(listing)):
        unprocessed_kelvin=ncload(listing[i], 'ts')
        processed_kelvin=return_to_average_200(unprocessed_kelvin)
        ts_buffer[i]=processed_kelvin-np.ones(np.shape(processed_kelvin))*273.15
    return np.mean(ts_buffer,axis=0)

#cleanup of previous latex file
if os.path.isfile("latex.txt"):
    os.remove("latex.txt")

#Function to make a latex list of all the figures, was useful for the report
def make_latex_fig(title, label, save_file_name):
    lines = ["\\begin{figure}[H]", "\centering", "\includegraphics[width=10cm]{fig/"+save_file_name+"}",
             "\caption{"+title+"}","\label{fig:"+label+"}","\end{figure}"]
    with open('latex.txt', 'a') as f:
        f.writelines('\n'.join(lines))
    
#free_table(path to all files to average in free method)
nbfiles_free=10
free_table_sst=np.array([""]*nbfiles_free,dtype=object)
free_table_ts=np.array([""]*nbfiles_free,dtype=object)
for i in range(nbfiles_free):
    free_table_sst[i]='free_'+str(i)+'_sst.nc'
    free_table_ts[i]='free_'+str(i)+'_ts.nc'

'''
Computes the performance of a certain method over all masks sst and ts grids
pre:
    method: The method used
    yrs: The timescale of metric measurement
    sst_comp: the sst values extracted for this method reconstruction
    ts_comp: the ts values extracted for this reconstruction
    metric: The measurement metric
    plot_stuff: Whether or not we should plot the maps etc
returns:
    vec_tab: a list of the average metric each entry matches with the mask in the mask_order table
'''
def compute(method,yrs,sst_comp,ts_comp,metric= "correlation", plot_stuff=True):
    
    #Allocate for metric
    vec_tab=np.zeros(len(mask_order))
    
    #Load the reference run
    if yrs==10:
        ts_ref_temp=ts_ref_10
        sst_ref_temp=sst_ref_10
    else:
        ts_ref_temp=ts_ref
        sst_ref_temp=sst_ref
    
    #Choose the metric and calculate
    if metric=="correlation":
        vec_sst=corr_cells(sst_comp,sst_ref_temp)
        vec_ts=corr_cells(ts_comp,ts_ref_temp)
        
    if metric=="CE":
        vec_sst=CE_cells(sst_comp,sst_ref_temp)
        vec_ts=CE_cells(ts_comp,ts_ref_temp)
    
    if plot_stuff: #Plot maps and save them
        make_map(metric,vec_ts,"ts",method,yrs)
        make_map(metric,vec_sst,"sst",method,yrs)
    
    #Make mask averages of the metric
    for i in range(len(mask_order)):
        mask=mask_order[i]
        var=var_mask_order[i]
        exec("vec_tab[i]=mean_target_mask(vec_"+var+","+mask+","+var+"_area)")
            
    
    return vec_tab

mask=0
weights=0

'''
Computes the performance of a certain method over all masks sst and ts grids
pre:
    ref: The reference run
    comp: The reconstruction run
    method: The method used
    var: the variable it concerns (sst or ts)
    yrs: The time series timescale
    mask: mask over which the time series is averaged
returns: Nothing
    saves the time series plot
'''
def plot_time_series(ref,comp,method,var,yrs,mask):
    plt.figure()
    if method=="base_i":
        method="base_interpolation"
    if "free" in method:
        method="free"
    plt.plot(list(range(1000,1200,yrs)),ref,color='red',label='reference')
    plt.plot(list(range(1000,1200,yrs)),comp,color='blue',label='reconstruction')
    plt.ylabel("Temperature")
    plt.xlabel("Year")
    if yrs==1:
        plt.title("Yearly Time Series using "+method)
    else:
        plt.title("Decadal Time Series using "+method)
    plt.legend()
    plt.savefig("time_series/time_series_"+method+"_"+var+"_"+str(yrs)+"_"+mask+".png")
    
'''
Same as compute, except it is an correlation of averages and not average of correlation
This function is only used for correlation, CE is still there as a leftover
'''
def compute_zonally(method,yrs,sst_comp,ts_comp,metric= "correlation", plot_stuff=True):
    global mask, weights
    
    vec_tab=np.zeros(len(mask_order))
    
    if yrs==10:
        ts_ref_temp=ts_ref_10
        sst_ref_temp=sst_ref_10
    else:
        ts_ref_temp=ts_ref
        sst_ref_temp=sst_ref
    
    for i in range(len(mask_order)):
        
        var=var_mask_order[i]
        
        exec("mask="+mask_order[i],globals())
        
        mask=np.tile(mask,(np.shape(ts_ref_temp)[0],1,1))
        
        ref=0
        comp=0
        
        if var=="sst":
            weights=np.tile(sst_area,(np.shape(ts_ref_temp)[0],1,1))
            ref= mean_target_mask_time(sst_ref_temp,mask,weights)
            comp= mean_target_mask_time(sst_comp,mask,weights)
            if metric=="correlation":
                vec_tab[i]=np.corrcoef(ref,comp)[0,1]
            if metric=="CE":
                vec_tab[i]=np.sqrt(np.mean((ref-comp)**2))
        if var=="ts":
            weights=np.tile(ts_area,(np.shape(ts_ref_temp)[0],1,1))
            ref= mean_target_mask_time(ts_ref_temp,mask,weights)
            comp= mean_target_mask_time(ts_comp,mask,weights)
            if metric=="correlation":
                vec_tab[i]=np.corrcoef(ref,comp)[0,1]
            if metric=="CE":
                vec_tab[i]=np.sqrt(np.mean((ref-comp)**2))
        
        if plot_stuff:
            plot_time_series(ref,comp,method,var,yrs,mask_order[i])
                
    return vec_tab
    
yrs=[1,10]

'''
Function runnign all the diagnostics on a certain method
'''   
def diagnose(method, plot_stuff=True):
    
    global yrs
    if method=="free":
        corr_tab_tot=np.zeros((2,len(mask_order),nbfiles_free))
        CE_tab_tot=np.zeros((2,len(mask_order),nbfiles_free))
        corr_tab_zonally_tot=np.zeros((2,len(mask_order),nbfiles_free))
        CE_tab_zonally_tot=np.zeros((2,len(mask_order),nbfiles_free))
        for i in range(nbfiles_free):
            if i==0: #Plots stuff, only plots for the first run that way we dont get 10 times too many plots that mean nothing
                corr_tab_tot[:,:,i],CE_tab_tot[:,:,i],corr_tab_zonally_tot[:,:,i],CE_tab_zonally_tot[:,:,i]=diagnose("free_"+str(i), plot_stuff=True)
            else:
                corr_tab_tot[:,:,i],CE_tab_tot[:,:,i],corr_tab_zonally_tot[:,:,i],CE_tab_zonally_tot[:,:,i]=diagnose("free_"+str(i), plot_stuff=False)
        corr_tab=np.mean(corr_tab_tot,axis=2)
        CE_tab=np.mean(CE_tab_tot,axis=2)
        corr_tab_zonally=np.mean(corr_tab_zonally_tot,axis=2)
        CE_tab_zonally=np.mean(CE_tab_zonally_tot,axis=2)
            
    else:
        file_sst=str(method+"_sst.nc")
        file_ts=str(method+"_ts.nc")
        sst_comp=ncload(file_sst, 'sst')
        ts_comp=ncload(file_ts, 'ts')
        if "free" not in method:
            ts_comp=make_yearly(ts_comp)
            if method=="backtrack":#Backtrack sst given monthly the rest isnt
                sst_comp=make_yearly(sst_comp)
        else:
            sst_comp=return_to_average_200(sst_comp)
            ts_comp=return_to_average_200(ts_comp)
            ts_comp=ts_comp-np.ones(np.shape(ts_comp))*273.15
    
        corr_tab=np.zeros((2,len(mask_order)))
        CE_tab=np.zeros((2,len(mask_order)))
        corr_tab_zonally=np.zeros((2,len(mask_order)))
        CE_tab_zonally=np.zeros((2,len(mask_order)))
        total_yrs=len(sst_comp)
        for i in range(len(yrs)):
            sst_comp_temp=np.mean(np.reshape(sst_comp,tuple(np.append([int(total_yrs/yrs[i]),yrs[i]],list(np.shape(sst_comp)[1:])))),axis=1)
            ts_comp_temp=np.mean(np.reshape(ts_comp,tuple(np.append([int(total_yrs/yrs[i]),yrs[i]],list(np.shape(ts_comp)[1:])))),axis=1)
            
            corr_tab[i,:]=compute(method,yrs[i],sst_comp_temp,ts_comp_temp,metric="correlation",plot_stuff=plot_stuff)
            CE_tab[i,:]=compute(method,yrs[i],sst_comp_temp,ts_comp_temp,metric="CE",plot_stuff=plot_stuff)
            corr_tab_zonally[i,:]=compute_zonally(method,yrs[i],sst_comp_temp,ts_comp_temp,metric="correlation",plot_stuff=plot_stuff)
            CE_tab_zonally[i,:]=compute_zonally(method,yrs[i],sst_comp_temp,ts_comp_temp,metric="CE",plot_stuff=plot_stuff)
        
    return corr_tab,CE_tab,corr_tab_zonally,CE_tab_zonally

'''
Function runnign all the diagnostics on the pseudoproxies with regard to their reference run
'''   
def diagnose_pseudoproxies():
    
    global mask, weigths
    
    corr_tab=np.zeros((2,2))
    CE_tab=np.zeros((2,2))
    corr_tab_zonally=np.zeros((2,2))
    CE_tab_zonally=np.zeros((2,2))
    
    for i in range(len(yrs)):
        if i==0:
            reff_ts=ts_ref
            pseudoo_ts=ts_pseudo
            reff_sst=sst_ref
            pseudoo_sst=sst_pseudo
        else:
            reff_ts=ts_ref_10
            pseudoo_ts=ts_pseudo_10
            reff_sst=sst_ref_10
            pseudoo_sst=sst_pseudo_10
            
        for j in range(len(specific_masks_pseudo)):
            exec("mask="+specific_masks_pseudo[j],globals())
            if "sst" in specific_masks_pseudo[j]:
                r=reff_sst
                pseud=pseudoo_sst
                area=sst_area
            else:
                r=reff_ts
                pseud=pseudoo_ts
                area=ts_area
            
            corr_tab[i,j]=mean_target_mask(corr_cells(r,pseud),mask,area)
            CE_tab[i,j]=mean_target_mask(np.sqrt(np.mean((r-pseud)**2,axis=0)),mask,area)
            mask=np.tile(mask,(np.shape(r)[0],1,1))
            weights=np.tile(area,(np.shape(r)[0],1,1))
            corr_tab_zonally[i,j]=np.corrcoef(mean_target_mask_time(r,mask,weights),mean_target_mask_time(pseud,mask,weights))[0,1]
            CE_tab_zonally[i,j]=np.sqrt(np.mean((mean_target_mask_time(r,mask,weights)-mean_target_mask_time(pseud,mask,weights))**2))
    
    return corr_tab,CE_tab,corr_tab_zonally,CE_tab_zonally
        
def make_map(metric,vec, var, method, yrs):
    
    fig = plt.figure(figsize=(8, 8))
    m = Basemap(projection='npstere',boundinglat=15,lon_0=0,resolution='l', round=True)
    
    if var=="sst":
        m.fillcontinents(color=(0.75,0.75,0.75),lake_color='#000545',zorder=1)                  
    if var=="ts":
        m.drawcoastlines(zorder=2)
        
#    if method=="base" and var=="ts" and yrs==1:
#        print("The minimum "+metric+" is: ",str(np.amin(vec)))
        
    
    m.drawparallels(np.arange(-90,90,30), color='w',dashes=[1, 0],labels=[1,1,1,1],zorder=4)
    m.drawmeridians(np.arange(-180.,180.,60.),color='w',dashes=[1, 0],labels=[1,1,1,1],zorder=4)
    m.drawmapboundary(fill_color='#000545')
    
    # Draw data
    exec("m.pcolormesh("+var+"_lon_grid, "+var+"_lat_grid, vec, latlon=True, cmap='seismic',zorder=1)")
    
    if metric=="correlation":
        plt.clim(-1,1)
        # Draw colorbar
        cb = plt.colorbar(label='Correlation',fraction=0.046, pad=0.04);
        cb.ax.plot([0, 1], [0.15,0.15], color=(0.0,0.0,0.0), linewidth=3)
        cb.ax.plot([0, 1], [0.15,0.15], color=(0.8,0.8,0.8), linewidth=3)
    if metric=="CE":
        plt.clim(0,1)
        plt.clim(vmin=0)
        cb = plt.colorbar(label='CE',fraction=0.046, pad=0.04);
        cb.ax.plot([0, 1], [0.15,0.15], color=(0.0,0.0,0.0), linewidth=3)
        cb.ax.plot([0, 1], [0.15,0.15], color=(0.8,0.8,0.8), linewidth=3)
    
    # Add assimilpoints
    lons, lats = m(p_sst_lons, p_sst_lats)
    m.scatter(lons, lats, color='lime', marker='o', zorder=3)
    
    # Add assimilpoints
    lons_ts, lats_ts = m(p_ts_lons, p_ts_lats)
    m.scatter(lons_ts, lats_ts, color='orange', marker='o',zorder=3)
    
    # sinon les bords sont coupés dans la figure sauvée
    plt.tight_layout() 
    
    # save figure 
    label=str("map_"+var+"_"+method+"_"+str(yrs)+"_"+metric)
    save_file_name=str(label+".png")
    fig.savefig('maps/'+save_file_name)
    
    # make title
    yearly="yearly"
    if yrs==10:
        yearly="decadal"
    
    metrica="\\rho"
    if metric=="CE":
        metrica=metric
        
    varia="TS"
    if varia=="sst":
        varia="SST"
        
    title=str("Illustration of "+yearly+" "+metrica+" on "+varia)
    
    make_latex_fig(title,label,save_file_name)

#Makes a map that shows the zones in a certain grids gicen a list of zones and a list of variables
'''
pre:
    zone_vals: a list of masked integer grids where the grid areas of the zone contain 1
    var_list: a list of proxy variables over which each grid spreads
    name: name of the saved file of this map
    var_map: the grid map on which we focus (sst grid map or ts grid map)
    show_proxies: whether or not to show proxies
'''   
def make_zone_map(zone_vals, var_list, name, var_map, show_proxies=True):
    
    fig = plt.figure(figsize=(8, 8))
    m = Basemap(projection='npstere',boundinglat=15,lon_0=0,resolution='l', round=True)
    
    m.fillcontinents(color=(0.75,0.75,0.75),lake_color='#000545',zorder=1)#make continents                  
        
    m.drawparallels(np.arange(-90,90,30), color='w',dashes=[1, 0],labels=[1,1,1,1],zorder=4)#pretty lines
    m.drawmeridians(np.arange(-180.,180.,60.),color='w',dashes=[1, 0],labels=[1,1,1,1],zorder=4)
    m.drawmapboundary(fill_color='#000545')
    
    if show_proxies:#Show proxies, with more transparent colormap
        # Draw data
        for i in range(len(zone_vals)):#Choose color of plot depending on proxy areas it covers, will color to that specific color since vmax is set to 1
            var=var_list[i]
            if var=='sst':
                colormap='Reds'
            if var=='ts':
                colormap='Blues'
            if var=='tssst':
                colormap='Purples'
            exec("m.pcolormesh("+var_map+"_lon_grid, "+var_map+"_lat_grid,zone_vals[i], latlon=True, cmap=colormap,vmin=-1, vmax=1,zorder=2, alpha=0.6)")
        # Add assimilpoints
        lons, lats = m(p_sst_lons, p_sst_lats)
        m.scatter(lons, lats, color='lime', marker='o', zorder=3)
        
        # Add assimilpoints
        lons_ts, lats_ts = m(p_ts_lons, p_ts_lats)
        m.scatter(lons_ts, lats_ts, color='orange', marker='o',zorder=3)
        
    else:#Dont show proxies with less transparent colormap
        # Draw data
        for i in range(len(zone_vals)):
            var=var_list[i]
            if var=='sst':
                colormap='Reds'
            if var=='ts':
                colormap='Blues'
            exec("m.pcolormesh("+var_map+"_lon_grid, "+var_map+"_lat_grid,zone_vals[i], latlon=True, cmap=colormap,vmin=-1, vmax=1,zorder=2, alpha=0.8)")
    
    # sinon les bords sont coupés dans la figure sauvée
    plt.tight_layout() 
    
    # save figure 
    fig.savefig('maps/'+name)

#Global variables set for analysis
method_order=["free","base_i","base","backtrack","cumulative"]
columnas=["metric","f","f_10","b_i","b_i_10","b","b_10","bk","bk_10","c","c_10"]
corr_table=np.zeros((len(method_order),2,len(mask_order)))
CE_table=np.zeros((len(method_order),2,len(mask_order)))
corr_table_zonally=np.zeros((len(method_order),2,len(mask_order)))
CE_table_zonally=np.zeros((len(method_order),2,len(mask_order)))
prox_corr_table=np.zeros((2,2))
prox_CE_table=np.zeros((2,2))
prox_corr_table_zonally=np.zeros((2,2))
prox_CE_table_zonally=np.zeros((2,2))

#Function which runs both the diagnostics on method performance and proxy correlations
def run():
    global method_order, corr_table,CE_table, corr_table_zonally, CE_table_zonally,prox_corr_table,prox_CE_table,prox_corr_table_zonally,prox_CE_table_zonally
    for i in range(len(method_order)):
        method=method_order[i]
        corr_table[i,:,:],CE_table[i,:,:],corr_table_zonally[i,:,:],CE_table_zonally[i,:,:]=diagnose(method)
    
    prox_corr_table,prox_CE_table,prox_corr_table_zonally,prox_CE_table_zonally=diagnose_pseudoproxies()
    make_bar_charts()

#Prints a recap table of the performance of each method on each mask when given a certain metric
#if zonally=true, we get the correlations on average temperatures and not average correlations on temperatures       
def recap_table(metric="correlation", zonally=False):
    if zonally:
        if metric=="correlation":
            reshape_corr_table=np.reshape(np.append(mask_order,np.round(corr_table_zonally,decimals=2)),(len(method_order)*2+1,len(mask_order)))
            df=pd.DataFrame(reshape_corr_table.T, columns=columnas)
        if metric=="CE":
            reshape_CE_table=np.reshape(np.append(mask_order,np.round(CE_table_zonally,decimals=2)),(len(method_order)*2+1,len(mask_order)))
            df=pd.DataFrame(reshape_CE_table.T, columns=columnas)
    else:
        if metric=="correlation":
            reshape_corr_table=np.reshape(np.append(mask_order,np.round(corr_table,decimals=2)),(len(method_order)*2+1,len(mask_order)))
            df=pd.DataFrame(reshape_corr_table.T, columns=columnas)
        if metric=="CE":
            reshape_CE_table=np.reshape(np.append(mask_order,np.round(CE_table,decimals=2)),(len(method_order)*2+1,len(mask_order)))
            df=pd.DataFrame(reshape_CE_table.T, columns=columnas)
    return df

#Function to generate all bar charts
def make_bar_charts():
    global corr_table, yrs
    #Grouping the masks to be placed on a similar bar chart together
    grouped_masks={"sst_point":np.array(["sst_mask"]),"ts_point":np.array(["ts_mask","sst_mask_tstab","ts_sst_mask"]),
                   "sst_zone":np.array(["sst_zone_mask"]), "ts_zone":np.array(["ts_zone_mask","sst_zone_mask_tstab","ts_sst_zone_mask"]),
                   "sst_nh":np.array(["sst_nh_mask"]), "ts_nh":np.array(["ts_nh_mask"])}
    grouped_masks_order={"sst_point":np.array([0]),"ts_point":np.array([3,4,5]),
                   "sst_zone":np.array([1]), "ts_zone":np.array([6,7,8]),
                   "sst_nh":np.array([2]), "ts_nh":np.array([9])}
    plot_order=["sst_point","ts_point","sst_zone","ts_zone","sst_nh","ts_nh"]
    
    for i in range(len(yrs)):
        temp_full_tab=corr_table[:,i,:]
        temp_full_tab_z=corr_table_zonally[:,i,:]
        yrs_val=yrs[i]
        for plot in plot_order:
            if plot=="sst_point" and yrs_val==10:#Allows to add the proxies (sst only have 10 year proxies)
                make_histogram_correlations(plot,grouped_masks[plot],grouped_masks_order[plot],yrs_val,temp_full_tab, optional_series=prox_corr_table[i,1])
                make_histogram_correlations(plot,grouped_masks[plot],grouped_masks_order[plot],yrs_val,temp_full_tab_z, zonally=' zonally', optional_series=prox_corr_table_zonally[i,1])
            else:
                if plot=="ts_point":#ts proxies can have 1 year and 10 year proxies
                    make_histogram_correlations(plot,grouped_masks[plot],grouped_masks_order[plot],yrs_val,temp_full_tab, optional_series=prox_corr_table[i,0])
                    make_histogram_correlations(plot,grouped_masks[plot],grouped_masks_order[plot],yrs_val,temp_full_tab_z, zonally=' zonally', optional_series=prox_corr_table_zonally[i,0])
                else:
                    make_histogram_correlations(plot,grouped_masks[plot],grouped_masks_order[plot],yrs_val,temp_full_tab)
                    make_histogram_correlations(plot,grouped_masks[plot],grouped_masks_order[plot],yrs_val,temp_full_tab_z, zonally=' zonally')
'''
pre:
    plot: a string of the format var_type (with var=sst and ts) and type(point,zone,nh)
    masks: the masks over which the correlations are evaluated (one color in the bar plot for each mask)
    mask_order: the locations of the masks of the masks list prior in the mask_order table
    yrs_val: the time series timescale over which the correlations are calculated
    temp_full_tab: The data table of the metric performance over the timescale yrs_val each cell is a pair [method, mask]
    zonally: string to be added to name of the file to indicate it was zonally (where we get the correlations on average temperatures and not average correlations on temperatures)
    optional_series: used if we want to add a non-conventional series to the table, is used when we can add a proxy comparison
'''            
def make_histogram_correlations(plot,masks,masks_order,yrs_val,temp_full_tab,zonally='', optional_series=None):
    def give_color(name):
        if 'ts and sst' in name:
            return 'purple'
        if 'sst' in name:
            return 'red'
        if 'ts' in name:
            return 'blue'
        else:
            exit("WRONG NAME FORMAT")
    nb_masks=len(masks)
    thicccness=0.8/nb_masks
    plt.figure()
    
    if optional_series==None:
        X = method_order
        X_axis = np.arange(len(X))
        X_axis_t = np.copy(X_axis)
    else:#There is an optional proxy series to add
        X = np.append(["proxy"],method_order)
        modify_mask=masks[0].replace('_mask',' ').replace('_tstab',' (in ts table)').replace('_zone', ' zone').replace('_nh',' nh').replace('_',' and ')
        plt.bar(0, optional_series, thicccness, color=give_color(modify_mask))
        X_axis = np.arange(len(X))
        X_axis_t = np.copy(X_axis[1:])
    
    for i in range(nb_masks):
        modify_mask=masks[i].replace('_mask',' ').replace('_tstab',' (in ts table)').replace('_zone', ' zone').replace('_nh',' nh').replace('_',' and ')
        displacement=-0.4+thicccness*(i+1/2)
        plt.bar(X_axis_t +displacement, temp_full_tab[:,masks_order[i]], thicccness, label = modify_mask, color=give_color(modify_mask))
      
    plt.xticks(X_axis, X)
    plt.xlabel("Methods")
    plt.ylabel("Correlation")
    if zonally=='':
        intitle="Means of correlations"
    else:
        intitle="Correlations of mean temperature"
    if yrs_val>1:
        plt.title(intitle+" "+str(yrs_val)+" years")
    else:
        plt.title(intitle+" "+str(yrs_val)+" year")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(str("bar_plots/bar_plot_"+plot+"_"+str(yrs_val)+"_yr"+zonally.replace(' ','_')))
    
    
run()


######PLOTTING ALL ZONE MAPS#########
sst_zone_plot=ma.masked_array(np.array(sst_zone_mask,dtype=int), mask=(sst_zone_mask | (sst_ref.data[0]==-99.99)))
sst_zone_plot_tstab=ma.masked_array(np.array(sst_zone_mask_tstab,dtype=int), mask=sst_zone_mask_tstab)
ts_zone_plot=ma.masked_array(np.array(ts_zone_mask,dtype=int), mask=ts_zone_mask)
sst_prox_plot=ma.masked_array(np.array(sst_mask,dtype=int), mask=sst_mask)
sst_prox_plot_tstab=ma.masked_array(np.array(sst_mask_tstab,dtype=int), mask=sst_mask_tstab)
ts_prox_plot=ma.masked_array(np.array(ts_mask,dtype=int), mask=ts_mask)
sst_nh_plot=ma.masked_array(np.array(sst_nh_mask,dtype=int), mask=(sst_nh_mask.data | (sst_ref.data[0]==-99.99)))
ts_nh_plot=ma.masked_array(np.array(ts_nh_mask,dtype=int), mask=ts_nh_mask)
to_see_proxies=ma.masked_array(np.array(ts_nh_mask,dtype=int), mask=True)
ts_sst_zone_plot=ma.masked_array(np.array(ts_sst_zone_mask,dtype=int), mask=ts_sst_zone_mask)


make_zone_map([sst_zone_plot], ['sst'], "sst_zone_plot.png",'sst')
make_zone_map([sst_zone_plot_tstab,ts_zone_plot], ['sst','ts'], "ts_zones_plot.png",'ts')
make_zone_map([sst_prox_plot], ['sst'], "sst_prox_plot.png",'sst', show_proxies=False)
make_zone_map([sst_prox_plot_tstab,ts_prox_plot], ['sst','ts'], "ts_prox_plot.png",'ts', show_proxies=False)
make_zone_map([sst_nh_plot], ['sst'], "sst_nh_plot.png",'sst')
make_zone_map([ts_nh_plot], ['ts'], "ts_nh_plot.png",'ts')
make_zone_map([to_see_proxies], ['ts'], "map_proxies.png",'ts')
make_zone_map([ts_sst_zone_plot], ['tssst'], "map_zone_ts_sst.png",'ts')