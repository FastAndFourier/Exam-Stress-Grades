from tkinter import N
from tkinter.ttk import LabeledScale
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from tensorflow.keras.utils import to_categorical
import warnings
from nltk.util import ngrams
from scipy.signal import resample
from cvxEDA.src.cvxEDA import *

N_FEATURE = 14
SIZE_W = 10

colors = ["#00b3b3","#ff66cc","#00b300","#0000b3","#bb99ff",
          "#cccc00","#e60000","#993333","#808000","#003300",
          "#66ffff","#3333ff","#cc0066","#006600","#006699",
          "#6699ff","#ecc6d9","#ffd24d","#ff751a","#800080"]

def compute_entropy(x):

    x_minmax = normalize(x)

    p = np.histogram(x_minmax,bins=np.linspace(0,1,50),density=True)[0] 
             
    p = p/np.size(p)

    return -np.sum(p*np.log(p+1e-10))

def compute_features(y):


    features = np.zeros((y.shape[0],N_FEATURE*y.shape[-1]))

    for i,x in enumerate(y):

        

        for ch in range(x.shape[1]):


            x_ch = x[:,ch]

            x_znormed = normalize(x_ch,z_score=True)

            S = np.fft.fft(x_znormed)
            x_fft = abs(x)[:len(S)//2]
            f0 = np.argmax(S)
            first_diff = [x_ch[k+1] - x_ch[k] for k in range(len(x)-1)]


            features[i,ch*N_FEATURE:(ch+1)*N_FEATURE] = [np.mean(x_ch),np.std(x_ch),np.median(x_ch),
                                                        np.min(x_ch),np.max(x_ch),np.percentile(x_ch,25),
                                                        np.percentile(x_ch,75),np.sqrt(np.mean(x_ch**2)),
                                                        compute_entropy(x_ch),compute_entropy(x_fft),
                                                        int(np.sqrt(np.mean(x_fft**2))),
                                                        100/f0 if f0!=0 else 0,np.mean(first_diff),
                                                        np.std(first_diff)]


    return np.array(features,dtype=np.float32)

def normalize(x,z_score=False,nmin=0,nmax=1):

    if len(x.shape)==1:
        x = x[:,np.newaxis]

    if z_score:
        x = (x - np.mean(x,axis=0))/(np.std(x,axis=0)+1e-10)
    else:
        x = nmin + nmax*(x - np.min(x,axis=0))/(np.ptp(x,axis=0)+1e-10)

    return x




def extract_window(data):

    sr = data[1,0]
    data = data[2:]
    

    L = len(data)//(SIZE_W*sr)

    data = data[:int(L*SIZE_W*sr)]


    return compute_features(np.stack(np.array_split(data,L)))



def build_feature_dataset(load):
    

    if os.path.isfile("feature_all_student.npy") and load:
        return np.load('feature_all_student.npy',allow_pickle=True), \
               np.load('index_all_student.npy',allow_pickle=True)

    session = ['Midterm 1','Midterm 2','Final']
    id_ = ['S'+str(i) for i in range(1,11)]


    index = []

    z = []
    
    start = time()

    for i in id_:

        for s in session:

            print("Session",s,'Student',i)

            path = 'dataset/Data/'+i+'/'+s+'/ACC.csv'
            data_acc = pd.read_csv(path,header=None).to_numpy()
            data_acc[2:,0] = np.sqrt(np.sum(data_acc[2:]**2,axis=1))
            data_acc = data_acc[:,0]


            path = 'dataset/Data/'+i+'/'+s+'/EDA.csv'
            data_eda = pd.read_csv(path,header=None).to_numpy()

            [phasic,_,tonic,_,_,_,_] = cvxEDA(data_eda[2:].T[0],1./data_eda[1,0]) 


            data_eda_comp = np.concatenate((phasic.reshape(-1,1),tonic.reshape(-1,1)),axis=1)
            data_eda_comp = np.concatenate((np.repeat(data_eda[:2],2,axis=1),data_eda_comp),axis=0)
    
            z += [np.concatenate((extract_window(data_acc[:,np.newaxis]), 
                                  extract_window(data_eda_comp)),axis=1)]
            index.append(z[-1].shape[0])

        print()
    z = np.concatenate(z)
    print("Processing duration:",int(time()-start),"seconds")

    temp = [index[0]]
    for k in range(1,len(index)):
        temp.append(temp[k-1]+index[k])

    index = np.array(temp).reshape((len(id_),len(session)))

    np.save("feature_all_student",z)
    np.save("index_all_student",index)

    return z,index

class FeatureAnalysis:

    def __init__(self,C=10,proj=False,tfidf=True,load_feature=False):

        f,index = build_feature_dataset(load=load_feature)

        self.features = f
        self.index = index

        self.num_words = C
        self.proj_BoW = proj
        self.dictionnary = None

        self.do_tfidf = tfidf

        self.idf = None


    def get_nested_index(self):
        return np.array([self.index[0]] + [self.index[k] - self.index[k-1][-1] for k in range(1,self.index.shape[0])])




    def get_frame_embedding(self,fname="tsne_z-embedded_full"):

        if self.proj_BoW:
            if os.path.isfile(fname+".npy"):
                z_embedded = np.load(fname+".npy",allow_pickle=True)
            else:
                z_embedded = TSNE(n_components=2,perplexity=50,init='pca',learning_rate='auto').fit_transform(self.features)
                z_embedded = normalize(z_embedded,z_score=True) 
                np.save(fname,z_embedded)
        else:
            z_embedded = self.features 

        return z_embedded


    def word_bagging(self,plot=True,tfidf=True):

        z_embedded = normalize(self.get_frame_embedding(),z_score=True)
        self.dictionnary = KMeans(n_clusters=self.num_words,random_state=42).fit(z_embedded)
        

        if self.proj_BoW and plot:
            plt.figure()
            for k in range(self.num_words):
                plt.scatter(z_embedded[self.dictionnary.labels_==k][:,0],z_embedded[self.dictionnary.labels_==k][:,1],
                            label=str(k),c=colors[k])
                plt.title(str(self.num_words)+" clusters")
                plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))

        else:
            warnings.warn("Warning: Bag of Words was created in the feature space. If you want to visualize words in 2D, perform clustering in the 2D space with proj=True at initialization")


        if tfidf:

            label_id = np.array_split(self.dictionnary.labels_,self.index.ravel())[:-1]
            term_per_doc = np.array([[len(l[l==k]) for k in range(self.num_words)] for l in label_id])
            self.idf = np.log(self.index.size/np.array([1 + len(v[v!=0]) for v in term_per_doc.T]))
        else:
            self.idf = np.ones(self.num_words)/self.num_words


        return self.dictionnary.labels_


    def get_session_embedding(self):

        label_id = np.array_split(self.dictionnary.labels_,self.index.ravel())[:-1]
        term_per_doc = np.array([[len(l[l==k]) for k in range(self.num_words)] for l in label_id])
        tf = np.array([v/np.sum(v) for v in term_per_doc])

        tfidf = tf*np.tile(self.idf[np.newaxis,:],(tf.shape[0],1))

        if not self.do_tfidf:
            tfidf = np.ones(tfidf.shape)

        labels_tfidf = np.zeros((self.index.size,self.num_words))
        
        last_idx=0
        for k,idx in enumerate(self.index.ravel()):

            temp = to_categorical(self.dictionnary.labels_[last_idx:idx],self.num_words)
            labels_tfidf[k,:] = np.sum(temp,axis=0)*tfidf[k]
            last_idx = idx



        return labels_tfidf

        





if __name__ == "__main__":




    f = open("dataset/StudentGrades.txt")
    grades = []
    while(True):
        line = f.readline()
        if not line:
            break
        if 'S' in line and 'GRADES' not in line:
            grades.append(line.split('\n')[0].split(" ")[-1])

    grades = np.array(grades).reshape((3,10)).T

    for i in range(10):
        grades[i,2] = str(int(grades[i,2])//2)

    # plt.figure()
    # for k in range(10):
    #     plt.plot(list(map(int,grades[k])),label=str(k+1))
    # plt.legend()


    f_extract = FeatureAnalysis(tfidf=True,C=20,load_feature=True)
    # print(f_extract.index)
    f_extract.word_bagging()

    # # plt.figure()
    # # for k in range(f_extract.num_words):
    # #     plt.subplot(4,5,k+1)
    # #     plt.plot(f_extract.dictionnary.cluster_centers_[k])
    # #     plt.title(str(k+1))
    # #     plt.xticks([])
    # #     plt.yticks([])


    
    labels = f_extract.get_session_embedding()

    # labels_id = labels#np.array_split(labels,10)
    # distance = [[np.linalg.norm(p1 - p2) for p1 in labels_id] for p2 in labels_id]

    # plt.matshow(distance)
    
    
    
    #z_session = PCA(n_components=2).fit_transform(normalize(labels))
    z_session = TSNE(n_components=2,perplexity=7,random_state=0,init='pca',learning_rate='auto').fit_transform(labels)
    z_session = normalize(z_session,z_score=True)


    C = 4
    profile = KMeans(n_clusters=C,random_state=42).fit(z_session)


    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(C):
        #v = z_session[i*3:(i+1)*3]
        v = z_session[profile.labels_==i]
        
        #for k in range(3):
        #ax.scatter(v[:,0],v[:,1],list(map(int,grades[i,:])),label=str(i+1)) #,list(map(int,grades[i,:]))
        
        grades_profile = np.array(list(map(int,grades.ravel())))[profile.labels_==i]
        ax.scatter(v[:,0],v[:,1],grades_profile,label=str(i)) 



        # for k in range(3):
        #     plt.annotate(grades[i,k],(v[k][0]+0.01,v[k][1]+0.01))
        #     plt.annotate("["+str(k+1)+"]",(v[k][0]+0.01,v[k][1]-0.1))


    # plt.title("Session embedding for each student (grade + [session])")
    plt.legend(loc='center left',bbox_to_anchor=(1.2, 0.5))



    plt.show()




    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    # z,index = build_feature_dataset()
    
    # # if os.path.isfile("tsne_z-embedded_full.npy"):
    # #     z_embedded = np.load("tsne_z-embedded_full.npy",allow_pickle=True)
    # # else:
    # #     z_embedded = TSNE(n_components=2,perplexity=50,init='pca',learning_rate='auto').fit_transform(z)
    # #     z_embedded = normalize(z_embedded,z_score=True) 
    # #     np.save("tsne_z-embedded_full",z_embedded)

    # z_embedded = z
    # z_train = z_embedded[np.random.choice(np.arange(z.shape[0]),10000)]    

    # C = 20

    # model = KMeans(n_clusters=C,random_state=42).fit(z_train)

    # z_embedded = TSNE(n_components=2,perplexity=50,init='pca',learning_rate='auto').fit_transform(z_train)
    # z_train = z_embedded  




    # plt.figure()
    # for k in range(C):
    #     plt.scatter(z_train[model.labels_==k][:,0],z_train[model.labels_==k][:,1],
    #                 label=str(k),c=colors[k])
    # plt.title(str(C)+" clusters")
    # plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))

    # y_id = np.split(z_embedded,index[:,-1])

    # label_id = np.array_split(model.predict(z_embedded),index[:,-1])[:-1]

    # temp = [index[0]] + [index[k] - index[k-1][-1] for k in range(1,index.shape[0])]


    # label = []

    # for k in range(len(label_id)):
    #     label_session = np.split(label_id[k],temp[k])[:-1]
    #     label_session = np.array([np.sum(to_categorical(v,C),axis=0) for v in label_session])
    #     label.append(label_session)


    # label_train = np.concatenate(label)
    
    
    # plt.figure()
    # for n,p in enumerate([5,10,20,30]):
    #     z_session = TSNE(n_components=2,perplexity=p,n_iter=int(5e3),learning_rate='auto',random_state=42).fit_transform(label_train)
    #     z_session = normalize(z_session,z_score=True)
        
    #     plt.subplot(2,2,n+1)
    #     for i in range(10):
    #         v = z_session[i*3:(i+1)*3]
    #         plt.scatter(v[:,0],v[:,1],label=str(i+1))
            
    #         for k in range(3):
    #             plt.annotate(grades[i,k],(v[k][0]+0.01,v[k][1]+0.01))

    #     plt.title("Perplexity = "+str(p))
    #     plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))



    # plt.figure()
    # for k in range(index.shape[0]):
    #     y = y_id[k]
    #     plt.scatter(y[:,0],y[:,1],label='S'+str(k+1))
    #     plt.legend()

    


    # t = np.linspace(0,duration//60,len(data))
    # t100 = np.linspace(0,duration//60,len(data[::100]))
    # plt.plot(t,np.sum(np.sqrt(data**2),axis=1))
    # plt.plot(t100,np.sum(np.sqrt(data**2)[::100],axis=1))
    # #plt.plot(t,data,label=['x','y','z'])
    # #plt.legend()
    # plt.show()

    