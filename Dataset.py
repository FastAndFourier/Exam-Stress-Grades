import numpy as np
import pandas as pd
import json

import warnings
import os
from time import time

from cvxEDA.src.cvxEDA import *
from pywt import wavedec
from scipy.signal import butter, filtfilt



N_FEATURE_ACC = 15#19
N_FEATURE_EDA = 10
N_FEATURE_HR = 5

PATH_REGISTRY = "/home/isir/TECH-TOYS/code_git/exam_stress_dataset/data_registry.json"
PATH_DATASET = "/home/isir/TECH-TOYS/code_git/exam_stress_dataset/feature_dataset/"

ACC_SET = ['std','median','zcr','min','max','en','H','mean 1st diff','std 1st diff','en [1,5]Hz','mean sqrt db4 coef (4 levels)']
EDA_SET = ['median phasic','AUC phasic','std phasic','max phasic','median tonic','AUC tonic','std tonic','max tonic','ns_edr_amp','amp_sum']
HR_SET = ['mean','std','1st q','3rd q','median']



def normalize(x,z_score=False,nmin=0,nmax=1):

    if len(x.shape)==1:
        x = x[:,np.newaxis]

    if z_score:
        x = (x - np.mean(x,axis=0))/(np.std(x,axis=0)+1e-10)
    else:
        x = nmin + nmax*(x - np.min(x,axis=0))/(np.ptp(x,axis=0)+1e-10)

    return x

class Dataset():

    def __init__(self,modalities,load,size_win=30):

        self.modalities = modalities
        self.size_win = size_win
        self.feature_set = []

        if not load:
            if 'eda' in self.modalities:
                self.feature_set.append(EDA_SET)
            if 'acc' in self.modalities:
                self.feature_set.append(ACC_SET)
            if 'hr' in self.modalities:
                self.feature_set.append(HR_SET)

        
        f,i = self.build_feature_dataset(load)

        self.feature = f
        self.index = i



    def compute_entropy(self,x):

        x_minmax = normalize(x)

        p = np.histogram(x_minmax,bins=np.linspace(0,1,50),density=True)[0] 
                
        p = p/np.size(p)

        return -np.sum(p*np.log(p+1e-10))


    def extract_window(self,x,type_):

        sr = x[1,0]
        x = x[2:]
        
        L = len(x)//(self.size_win*sr)

        x = normalize(x[:int(L*self.size_win*sr)],z_score=True)


        return self.compute_features(np.stack(np.array_split(x,L)),type_,sr)





    

    def compute_features(self,y,type_,sr):

        if type_ == 'acc':
            features = np.zeros((y.shape[0],N_FEATURE_ACC*y.shape[-1]))

        elif type_ == "eda":
            features = np.zeros((y.shape[0],N_FEATURE_EDA*y.shape[-1]))

        elif type_ == "hr":

            features = np.zeros((y.shape[0],N_FEATURE_HR*y.shape[-1]))


        for i,x in enumerate(y[1:]):


            for ch in range(x.shape[1]):


                x_ch = x[:,ch]

                if type_ == 'acc':



                    x_zcr = x_ch - np.mean(x_ch)
                    zcr = 0.5*np.mean([abs(np.sign(x_zcr[k+1]) - np.sign(x_zcr[k])) for k in range(len(x_zcr)-1)])

                    S = np.fft.fft(x_ch)
                    freq = np.fft.fftfreq(len(x_ch))
                    freq = freq[:len(freq)//2]*sr
                    x_fft = normalize(abs(S)[:len(S)//2])

                    en_lf = np.sum(x_fft[freq<=1]**2)
                    en_hf = np.sum(x_fft[np.logical_and(freq>1,freq<=5)]**2)


                    first_diff = [x_ch[k+1] - x_ch[k] for k in range(len(x)-1)]


                    wavelet_coef = wavedec(x_ch,'db4',level=4)
                    mean_dcoef = [np.mean(wcoef**2) for wcoef in wavelet_coef]


                    features[i,ch*N_FEATURE_ACC:(ch+1)*N_FEATURE_ACC] = \
                        [np.std(x_ch),np.median(x_ch),zcr,
                        np.min(x_ch),np.max(x_ch),np.sqrt(np.mean(x_ch**2)),
                        self.compute_entropy(x_ch),np.mean(first_diff),
                        en_hf,np.std(first_diff)] + mean_dcoef


                elif type_=='eda':
    
                    [phasic,p,tonic,_,_,_,_] = cvxEDA(x_ch,1./sr,options={'show_progress':False}) 

                    ns_edr_freq = len(p[p>(np.mean(p)+np.std(p))])
                    amp_sum = np.mean(p[p>(np.mean(p)+np.std(p))])

                    # f,eda_psd = welch(x_ch,sr,nperseg=len(x_ch))
                    # f_lim = f[f>=0.045]
                    # f_lim = f_lim[f_lim<=0.25]
                    # eda_symp = eda_psd[f==f_lim]

                    features[i,ch*N_FEATURE_EDA:(ch+1)*N_FEATURE_EDA] = [np.median(phasic),np.median(tonic),
                                                                np.sum(phasic)/len(phasic),np.sum(tonic)/len(tonic),
                                                                np.std(phasic),np.std(tonic),
                                                                np.max(phasic),np.max(tonic),
                                                                ns_edr_freq,amp_sum]


                elif type_ == "hr":

                    features[i,ch*N_FEATURE_HR:(ch+1)*N_FEATURE_HR] = [np.mean(x_ch),np.std(x_ch),
                                                                    np.median(x_ch),np.quantile(x_ch,.25),
                                                                    np.quantile(x_ch,.75)]

                else:

                    warnings.warn("Unknown sensor type")

                    

        return np.array(features,dtype=np.float32)





    def build_feature_dataset(self,load):
        

        if load:

            registry = print_registry()
            id_ = input("ID ? ")

            while id_ not in registry.keys():
                print("Please select a valid ID")
                id_ = input("ID ? ")

            return np.load(PATH_DATASET+'feature_'+id_+'.npy',allow_pickle=True), \
                np.load(PATH_DATASET+'index_'+id_+'.npy',allow_pickle=True)

        session = ['Midterm 1','Midterm 2','Final']
        id_ = ['S'+str(i) for i in range(1,11)]


        index = []

        z = []
        
        start = time()

        for i in id_:

            for s in session:

                print("Session",s,'Student',i)


                f = []

                if 'acc' in self.modalities:

                    path = 'dataset/Data/'+i+'/'+s+'/ACC.csv'
                    data_acc = pd.read_csv(path,header=None).to_numpy()


                    filt = butter(10,3,'low',fs=32)

                    for k in range(3):
                        data_acc[2:,k] = filtfilt(filt[0],filt[1],data_acc[2:,k],padlen=10)

        
                    data_acc[2:,0] = np.sqrt(np.sum(data_acc[2:]**2,axis=1))
                    data_acc = data_acc[:,0][:,np.newaxis]
                    f.append(self.extract_window(data_acc,'acc'))
                    

                if 'eda' in self.modalities:

                    path = 'dataset/Data/'+i+'/'+s+'/EDA.csv'
                    data_eda = pd.read_csv(path,header=None).to_numpy()
                    f.append(self.extract_window(data_eda,'eda'))
                    

                if 'hr' in self.modalities:

                    path = 'dataset/Data/'+i+'/'+s+'/HR.csv'
                    data_hr = pd.read_csv(path,header=None).to_numpy()
                    f.append(self.extract_window(data_hr,'hr'))

                min_len = min([v.shape[0] for v in f])

                features_ = np.concatenate([v[:min_len] for v in f],axis=1)                

                z += [features_]
                
                index.append(z[-1].shape[0])

            print()
        z = np.concatenate(z)
        print("Processing duration:",int(time()-start),"seconds")

        temp = [index[0]]
        for k in range(1,len(index)):
            temp.append(temp[k-1]+index[k])

        index = np.array(temp).reshape((len(id_),len(session)))



        if os.path.isfile(PATH_REGISTRY):
            with open(PATH_REGISTRY) as f:
                registry = json.load(f)
        else:
            registry = {}

        
        #input_size = input_size if not session_dataset else size_session

        
        id = np.random.randint(0,high=10**5)
        registry[id] = {'modalities':self.modalities,'size window':self.size_win,
                        'feature set':self.feature_set}


        with open(PATH_REGISTRY,"w") as out:
            json.dump(registry,out)

        np.save(PATH_DATASET+"feature_"+str(id),z)
        np.save(PATH_DATASET+"index_"+str(id),index)



        return z,index

def print_registry():

    with open(PATH_REGISTRY) as f:
        registry = json.load(f)


    print("")
    print("="*os.get_terminal_size().columns)
    print("-"*(os.get_terminal_size().columns//2 - 10)+\
        "Dataset summary"+"-"*(os.get_terminal_size().columns//2 - 4))
    print("="*os.get_terminal_size().columns)
    print("")

    columns = list(registry[list(registry.keys())[-1]].keys())
    size_c = np.zeros(1+len(columns),dtype=int)

    print("ID"+" "*8+"| ",end="")
    size_c[0] = len("ID"+" "*8+"| ")
    for i,k in enumerate(columns[:-1]):

        print(k+" "*10+"| ",end="")
        size_c[i+1] = len(str(k)+" "*10+"| ")



    print("")
    print("-"*os.get_terminal_size().columns)


    for v in registry:
        print(v,end=" "*(size_c[0]-len(str(v))))
        for i,v in enumerate(list(registry[v].values())[:-1]):
            print(str(v),end=" "*(size_c[i+1]-len(str(v))))
        print()


    return registry


if __name__ == "__main__":

    Dataset(['acc','hr','eda'],load=False,size_win=60)