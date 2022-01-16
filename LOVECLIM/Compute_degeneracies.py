# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:44:46 2022

@author: Antoine
"""

import numpy as np
import pandas as pd
import os

methods=["base","base_i","cumulative"]
os_path='C:\\Users\\Antoine\\Documents\\School\\EPL\\Master\\Mémoire\\LOVECLIM\\data'

def nb_part_percent(resamp_coeff, percent):
    threshold=len(resamp_coeff)*percent
    increment=0
    decroissant=np.flip(np.sort(resamp_coeff))
    for i in range(len(resamp_coeff)):
        increment+=decroissant[i]
        if increment > threshold:
            return i+1

def test_for_all_files(method, percent):
    already_treated=False
    if os.path.isfile(os.path.join(os_path,str("fcosts\\"+method+"\\already_treated.txt"))):
        already_treated=True
    else:
        with open(os.path.join(os_path,str("fcosts\\"+method+"\\already_treated.txt")), 'a') as f:
            f.write("YO I'M TREATED")
            f.close()
    if method=="cumulative":
        nb_files=20
        part_threshold=np.zeros(nb_files)
        ref=np.arange(1000,1200,10)
        for i in range(nb_files):
            yr=ref[i]
            file="00"+str(yr)+"_001_00"+str(yr+9)+"_360.dat"
            file_path=os.path.join(os_path,str("fcosts\\"+method+"\\"+file))
            if not already_treated:
                with open(file_path, 'r') as fin:
                    data = fin.read().splitlines(True)
                with open(file_path, 'w') as fout:
                    fout.writelines(data[2:])
            df=pd.read_table(file_path,delim_whitespace=True, header=None)
            part_threshold[i]=nb_part_percent(df.loc[:,5], percent)
    else:
        nb_files=200
        part_threshold=np.zeros(nb_files)
        ref=np.arange(1000,1200,1)
        for i in range(nb_files):
            yr=ref[i]
            file="00"+str(yr)+"_001.dat"
            file_path=os.path.join(os_path,str("fcosts\\"+method+"\\"+file))
            if not already_treated:
                with open(file_path, 'r') as fin:
                    data = fin.read().splitlines(True)
                with open(file_path, 'w') as fout:
                    fout.writelines(data[2:])
            df=pd.read_table(file_path,delim_whitespace=True, header=None)
            part_threshold[i]=nb_part_percent(df.loc[:,5], percent)
    return part_threshold
            
            
            
                        