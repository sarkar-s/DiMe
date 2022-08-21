#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os,sys
import copy
from scipy.signal import savgol_filter
import math

import pandas as pd
import seaborn as sns
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
def compute_pdf(cdf,ps,bin_ends):
    wts = np.zeros(shape=len(bin_ends))

    j = 0

    for i in range(0,len(bin_ends)):
        if i==len(bin_ends):
            wts[i] = cdf[-1]
        else:
            k = np.searchsorted(ps,bin_ends[i])
            k = min(k,cdf.shape[0]-1)
            wts[i] = cdf[k]

    ww = savgol_filter(wts,5,1,1)

    for i in range(0,ww.shape[0]):
        if ww[i]<1e-6:
            ww[i] = 0.0

    total_w = np.sum(ww)

    if total_w>0.0:
        ww *= 1.0/np.sum(ww)
    # else:
    #     ww = np.zeros(shape=len(bin_ends))
    #     ww[0] = 1.0

    final_wts = np.zeros(shape=(1,len(bin_ends)))
    final_wts[0,:] = ww

    return final_wts
"""

def compute_pdf(cdf,ps,bin_locs):
    wts = np.zeros(shape=len(bin_locs))
    j = 0

    for i in range(0,len(bin_locs)):
        while j<cdf.shape[0] and ps[j]<bin_locs[i]:
            j += 1

        if j<cdf.shape[0] and j>0:
            step = (bin_locs[i]-ps[j-1])
            wts[i] = cdf[j-1] + step*(cdf[j]-cdf[j-1])/(ps[j]-ps[j-1])
        elif j>=cdf.shape[0]:
            wts[i] = cdf[-1]

    wts[-1] = 1.0

    ww = savgol_filter(wts,7,2,1,mode='nearest')
    ww = np.abs(ww)

    final_wts = np.zeros(shape=(1,wts.shape[0]))
    final_wts[0,:] = ww/np.sum(ww)

    return final_wts

def get_mean_response(a,hills):
    g_mean = np.zeros(shape=a.shape)

    for i in range(0,a.shape[0]):
        step_factor = a[i]**hills[3]/(a[i]**hills[3] + hills[2]**hills[3])
        g_mean[i] = (hills[1]-hills[0])*step_factor + hills[0]

    return g_mean


def get_response_function(iptgs,g_m,percentile_values):
    cdf = np.linspace(0.01,0.99,99)
    response = np.zeros(shape=(iptgs.shape[0],4))
    response[:,0] = iptgs

    for i in range(0,iptgs.shape[0]):
        E = 0.0

        for k in range(0,percentile_values.shape[1]-1):
            dx = percentile_values[i,k+1] - percentile_values[i,k]

            E += (1-0.5*(cdf[k]+cdf[k+1]))*dx

        response[i,1] = g_m[i]
        response[i,2] = percentile_values[i,-4] - response[i,1]
        response[i,3] = response[i,1] - percentile_values[i,4]

    return response

# Create labels

c = np.linspace(1,99,99,dtype=int)
quantile_names = [str(ic) for ic in c]

gs = np.linspace(1,11,11,dtype=int)
g_names = ['g'+str(i) for i in gs]

#column_names = ['c']+g_names+['g']+quantile_names
column_names = ['c']+['g']#+quantile_names

model_folder = '/Users/sns9/Research/IMS_project/CytometryData/PredictDispersion/2-layers-16/'
model = {}

for q in range(1,100):
    model[q] = keras.models.load_model(model_folder+'model_'+str(q)+'.h5')

data_directory = '/Users/sns9/Research/IMS_project/BarSeq_data/measurands/'
os.chdir(data_directory)

hill_coeffs = pd.read_csv('cleaned_hill_coeffs.csv',header=None).to_numpy()
#print(hill_coeffs.shape)

data_directory = '/Users/sns9/Research/IMS_project/BarSeq_data/august-expressions/'
os.chdir(data_directory)

iptgs = np.zeros(shape=12)

for i in range(0,12):
    if i>0:
        iptgs[i] = 2**i

#cdf = np.linspace(0.0,1.0,101)
cdf = np.linspace(0.01,0.99,99)

for i in range(0,hill_coeffs.shape[0]):
    g_m = get_mean_response(iptgs,hill_coeffs[i,:])

    total_data = np.zeros(shape=(g_m.shape[0],2))
    total_data[:,0] = iptgs
    total_data[:,1] = g_m

    test_data = pd.DataFrame(total_data,index=[k for k in range(iptgs.shape[0])],columns=column_names)
    new_data = copy.deepcopy(test_data)

    for q in range(1,100):
        #model = keras.models.load_model(model_folder+'model_'+str(q)+'.h5')

        test_predictions = model[q].predict(test_data).flatten()

        new_data[str(q)] = test_predictions

        if q>1:
            test_data[str(q)] = test_data[str(q-1)] + test_predictions
        else:
            test_data[str(q)] = test_predictions

        #print(q,' completed.')
        #sys.stdout.flush()

    if i%100==0:
        print(i,' completed.')

    #for gn in g_names:
    #    new_data.pop(gn)

    for iq in range(1,len(quantile_names)):
        new_data[quantile_names[iq]] += new_data[quantile_names[iq-1]]

    data = new_data.to_numpy()
    percentile_values = data[:,2:]
    np.savetxt('percentile'+str(i)+'.csv',percentile_values,delimiter=',')

    """
    max_p, min_p = np.max(percentile_values[:,:-1]), np.min(percentile_values[:,1:])

    bin_locs = np.linspace(min_p,max_p,50)
    bin_ends = [0.5*(bin_locs[i]+bin_locs[i+1]) for i in range(0,bin_locs.shape[0]-1)]
    bin_ends.append(np.max(percentile_values))

    #expression = np.zeros(shape=(percentile_values.shape[0]+1,bin_locs.shape[0]))
    expression = np.zeros(shape=(1,bin_locs.shape[0]))

    expression[0,:] = bin_locs
    """

    # max_p, min_p = np.max(percentile_values[:,:-2]), max(np.min(percentile_values[:,2:]),1.0)
    # bin_ends = list(math.exp(1)**np.linspace(math.log(min_p),math.log(max_p),26))
    # bin_locs = [0.5*(bin_ends[i]+bin_ends[i+1]) for i in range(0,len(bin_ends)-1)]
    #
    # expression = np.zeros(shape=(1,len(bin_locs)))

    response = get_response_function(iptgs,g_m,percentile_values)

    np.savetxt('response'+str(i)+'.csv',response,header='i,g,+,-',comments='',delimiter=',')

    # expression[0,:] = bin_locs
    #
    # for k in range(0,percentile_values.shape[0]):
    #     p_row = percentile_values[k,2:-2]
    #     wts = compute_pdf(cdf[2:-2],p_row,bin_locs)
    #
    #     expression = np.concatenate((expression,wts),axis=0)
    #
    #     #expression[k+1,:] = wts
    #
    # np.savetxt('expressions_'+str(i)+'.csv',expression,delimiter=',')
