
#clean code in progress

import os
from glob import glob
import pathlib
import numpy as np
import json
import pandas as pd
from dict_base_option import DictOptions
import random

def filter_patient(directory):
    #filter the patient based on some rationale given by parser settings
    # option.filter_data = 1 will run this function
    # option.discriminator will give the file containing the data information to use for filtering
    # option.criteria_name will give the header of the column to use
    # optioncriteria_value will set the value for the filter. Filter can include values=criteria_value, or in case of age will include criteria_value_min<value<criteria_value_max
    discr_f_path = os.path.join(directory,info.discriminator)
    discr_df = pd.read_excel(discr_f_path) #create a dataframe with the information
    #print(discr_df)
    criteria_n = info.criteria_name
    criteria_v = info.criteria_value
    if criteria_n == "PatientSex":
        discr_list = discr_df.index[discr_df[str(criteria_n)]==str(criteria_v[0])].tolist()
    elif criteria_n == "PatientAge":
        if len(criteria_v)==1: #in case only one value
            if int(criteria_v) > 50: # if the age values given <50 then the younger patients will be included
                discr_list = discr_df.index[discr_df[criteria_n]<int(criteria_v)].tolist()
                print("Included patient with age < {criteria_v:i}")
            else:
                discr_list = discr_df.index[discr_df[criteria_n]>int(criteria_v)].tolist()
                print("Included patient with age > {criteria_v:i}")
        elif len(criteria_v)==2: 
            if criteria_v[0]>criteria_v[1]:
                criteria_v = criteria_v.reverse()
            discr_list1 = discr_df.index[discr_df[criteria_n]>int(criteria_v[0])].tolist()
            discr_list2 = discr_df.index[discr_df[criteria_n]<int(criteria_v[1])].tolist()
            discr_list = np.intersect1d(discr_list1,discr_list2)
    elif criteria_n == "BAV":
        discr_list = discr_df.index[discr_df[criteria_n]==1].tolist()
    ids = discr_df["PatientID"][discr_list]
    return ids

def append_pat(directory,s_to_file,patients):
    #dir = directory where data are
    #s_to_file = string to use to find the files
    directory = pathlib.Path(directory)
    for p in directory.glob(s_to_file):
        patient = os.path.dirname(p)
        if patient not in patients: 
            patients.append(patient)
    #return np.random.shuffle(patients)
    return patients

def create_pat_list(directory):
    patients=[]
    if info.filter_data == 1:
        ids = filter_patient(directory)
        for pat in ids:
            s_to_file = '**/*'+pat+'*/**/*_velx.nii.gz'
            patients = append_pat(directory,s_to_file,patients)
    else:
        s_to_file = '**/*_velx.nii.gz'
        patients = append_pat(directory,s_to_file,patients)
    #patients = np.random.shuffle(patients)
    return patients

def split_patients(patients):
    N = len(patients)
    pTrain = info.train_split
    pVal = info.val_split
    if pTrain > 1 or pVal > 1 or (pTrain+pVal)>1:
        raise Exception("Sum, train and validation need to be < 1")
    #pTest = 1 - (pTrain+pVal)
    np.random.shuffle(patients)
    train_pat = patients[:int(N*pTrain)]
    val_pat = patients[int(N*pTrain):int(N*(pTrain+pVal))]
    test_pat = patients[int(N*(pTrain+pVal)):]
    return train_pat, val_pat, test_pat

def dict_in(p):
    pd={
        'magnitude': str(p).replace('velx','magnitude'),
        'mask': str(p).replace('velx', 'mask'),
        'velx': str(p),
        'vely': str(p).replace('velx','vely'),
        'velz': str(p).replace('velx','velz')
        }
    return pd

def build_cases_dict(pat):
    #pat : list of patients to create a dictionary that contains all the planes available for that patient
    cases_dict = []
    #np.random.shuffle(pat)
    for i in pat:
        for p in pathlib.Path(i).glob('**/*_velx.nii.gz'):
            d = dict_in(p)
            cases_dict.append(d)
    return cases_dict

def save_dict_as_json(out_folder,file,dcase):
    filename = os.path.join(out_folder,file+'_dict.json')
    with open(filename,'w') as fp:
        json.dump(dcase,fp)
    return

def check_isdir(folder):
    id_extra = 0
    folder_o = folder
    while os.path.exists(folder):
        id_extra += 1
        folder = folder_o + str(id_extra)
    return folder

def main():
    base_directory = info.data_path #directory of the dataset from a specific site
    training_pat = []
    validation_pat = []
    test_pat = []
    tot_cases = 0
    for i in info.datasets:
        c_dir = os.path.join(base_directory,i)
        p_list = create_pat_list(c_dir)
        #print(p_list)
        tot_cases += len(p_list)
        tr,val,te = split_patients(p_list)
        training_pat += tr
        validation_pat += val
        test_pat += te
    out_folder = os.path.join(info.out_path, info.modelID)
    out_folder = check_isdir(out_folder)
    os.makedirs(pathlib.Path(out_folder))
    tr_d = build_cases_dict(training_pat)
    val_d = build_cases_dict(validation_pat)
    test_d = build_cases_dict(test_pat)
    random.shuffle(tr_d)
    random.shuffle(val_d)
    random.shuffle(test_d)
    for d,fname in zip([tr_d,val_d,test_d],["training","validation","test"]):
        save_dict_as_json(out_folder,fname,d)
    print('dictionaries saved in',out_folder)
    options.print_options(out_folder)
    return 

options = DictOptions()
info = options.gather_options()
base_directory = info.data_path #directory of the dataset from a specific site
sites = info.datasets #list of sites included

main()
