import glob #filenames and pathnames utility
import os   #operating sytem utility
import numpy as np
import pandas as pd
import pickle
import random as rand
import sys
import math
from scipy import special

def get_initial_shape_parameters(x_mean,x_var):
    scale = x_var/x_mean
    shape = x_mean**2/x_var

    return shape, scale

def solve_for_shape(initial_shape,s_factor):
    d_shape = initial_shape
    shape = initial_shape

    while abs(d_shape/shape)>1e-9:
        f_shape = math.log(shape) - special.digamma(shape) - s_factor

        digamma_p = (special.digamma(shape+0.001)-special.digamma(shape-0.001))/(2*0.001)

        f_p_shape = 1.0/shape - digamma_p

        d_shape = -f_shape/f_p_shape

        shape += d_shape
        #print 'new shape: ', shape
        #sys.stdout.flush()

    return shape

#data_directory = '/Users/sns9/Research/IMS_project/FeedbackExpDec18/WTA'

#data_directory = '/Volumes/Shared_Data/GSF-IMS/E-Coli/pLMSF-lacI/2020-02-18_IPTG-Cytom-12-plasmids/plate_1'

data_directory = '/Users/swarnavo/Research/IMS_project/pLMSF-lacI/2020-03-17_IPTG-Cytom-12-plasmids/plate_2'

output_directory = '/Users/swarnavo/Research/IMS_project/pLMSF-lacI/All_CDF_survey/'

current_dir = os.getcwd()
os.chdir(data_directory)

plates = ['B','C','D','E','F','G']
reps = ['140','141','142','143','144','145']

cdf_wts = np.linspace(0,100,101)

#os.chdir('../../')


# Plate and duplicate label
for p,r in zip(plates,reps):
    print(p,r)
    plate_label = [p]#['F']
    rep_label = 'pVER-IPTG-'+r#12'
    tag = 'pVER-IPTG-'+r#12'
    filter_string = 'pVER-IPTG-'+r+'-'#12-'
    conc_separator = '-'
    plate_separator = '_'
    data_fraction = 1

    def extract_data(data_set,index_set):
        extracted_data = []

        for i in index_set:
            try:
                extracted_data.append(data_set.values[i])
            except IndexError:
                print(i)
                sys.exit()

        return np.array(extracted_data)

    def compute_mean_variance(data):
        min_offset = min(data)
        shifted = data - min_offset

        mean_response = np.mean(shifted)
        var_response = np.var(shifted)

        percentiles = np.percentile(shifted,[5.0,95.0])

        return mean_response, var_response, percentiles

    coli_files = glob.glob('*.frame_pkl')

    skips = []

    for k in coli_files:
        if 'summary' in k:
            skips.append(k)

    for k in skips:
        coli_files.remove(k)

    filenames = [file.rsplit('.',1)[0] for file in coli_files]

    coli_frame = [ pickle.load(open(file, 'rb')) for file in coli_files ]

    # for file in filenames:
    #     if 'summary' in file:
    #         summary_idx = filenames.index(file)
    #
    # del filenames[summary_idx]
    # del coli_frame[summary_idx]

    #gated_data = [frame.loc[frame['is_cell']] for frame in coli_frame]


    singlet_data = [frame.loc[frame['is_singlet']] for frame in coli_frame]
    all_data = [frame for frame in coli_frame]

    fl_channel = 'BL1-A-MEF'
    glob_min = 1000000
    glob_max = 0

    all_mins = []

    data_covered = []

    location_string = {}
    wt_string = {}

    means = {}
    vars = {}
    percents = {}

    index_set = None
    data_size = 0

    conclist = []
    datas = {}

    for i, singlet, file in zip(range(len(all_data)), singlet_data, coli_files):
        index_set = None
        for j in range(1):
            label, plate_no = filenames[i].split(plate_separator)
            this_plate = plate_no[0]

            #print(label,plate_no)

            if (plate_label[0] in plate_no) and rep_label in label: # or plate_label[1] in plate_no
                #conc_v = float(label.lstrip(filter_string))#conc_separator)[1])
                conc_v = float(label.replace(filter_string,''))
                if conc_v!=0.0:
                    expo = math.log(conc_v)/math.log(2.0)
                    if abs(expo-int(expo))<1e-16:
                        conc_value = str(conc_v)
                    else:
                        conc_value = str(conc_v*1000)
                else:
                    conc_value = str(conc_v)

                    #print(conc_value,label,conc_v)

                if conc_value not in data_covered:
                    data_covered.append(conc_value)
                    conclist.append(float(conc_value))

                    data_array = singlet[fl_channel].to_numpy()

                    datas[conc_value] = data_array

                    this_min = min(data_array)

                    #print(len(singlet[fl_channel]))


                    glob_min = min(this_min,glob_min)
                    all_mins.append(this_min)

    conclist.sort()

    os.chdir(output_directory)

    of = open(r+'_CDF_survey.csv','w')

    for c in conclist:
        cs = str(c)
        this_data = np.copy(datas[cs])
        this_data -= glob_min*np.ones(shape=this_data.shape)
        this_data += np.ones(shape=this_data.shape)

        outstring = cs+','+str(np.mean(this_data))

        print(outstring,' completed')
        sys.stdout.flush()

        for i in range(0,cdf_wts.shape[0]):
            outstring += ',' + str(np.percentile(this_data,cdf_wts[i]))

        print(outstring,file=of)

    os.chdir(data_directory)

of.close()

# rf = open('local_CDF_survey.csv','r')
# new_lines = rf.readlines()
# rf.close()
#
# os.chdir('../../')
#
# wf = open('global_CDF_survey.csv','a')
#
# for l in new_lines:
#     print(l,file=wf)
#
# wf.close()
