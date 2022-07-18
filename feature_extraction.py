from tkinter import N
from tkinter.ttk import LabeledScale
from urllib.parse import non_hierarchical
import pandas as pd
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
import os
from tensorflow.keras.utils import to_categorical
import warnings
from nltk.util import ngrams
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import jensenshannon
from cvxEDA.src.cvxEDA import *
from pywt import wavedec

N_FEATURE_ACC = 15#19
N_FEATURE_EDA = 10
SIZE_W = 30

colors = ["#00b3b3","#ff66cc","#00b300","#0000b3","#bb99ff",
          "#cccc00","#e60000","#993333","#808000","#003300",
          "#66ffff","#3333ff","#cc0066","#006600","#006699",
          "#6699ff","#ecc6d9","#ffd24d","#ff751a","#800080"]

def compute_entropy(x):

    x_minmax = normalize(x)

    p = np.histogram(x_minmax,bins=np.linspace(0,1,50),density=True)[0] 
             
    p = p/np.size(p)

    return -np.sum(p*np.log(p+1e-10))

def compute_features(y,type_,sr):

    if type_ == 'acc':
        features = np.zeros((y.shape[0],N_FEATURE_ACC*y.shape[-1]))

        # filt = butter(10,5,'low',fs=sr)

        # w, h = freqs(filt[0],filt[1])

        # plt.semilogx(w, abs(h))
        # plt.title('Butterworth filter frequency response')

        # plt.xlabel('Frequency [radians / second]')

        # plt.ylabel('Amplitude [dB]')

        # plt.margins(0, 0.1)

        # plt.grid(which='both', axis='both')
        # plt.show()


    elif type_ == "eda":
        features = np.zeros((y.shape[0],N_FEATURE_EDA*y.shape[-1]))


    for i,x in enumerate(y[1:]):

        # plt.plot(x)
        # plt.show()

        

        for ch in range(x.shape[1]):


            x_ch = x[:,ch]


            

            if type_ == 'acc':

                #x_filt = filtfilt(filt[0],filt[1],x=x_ch,padlen=10)
                x_znormed = normalize(x_ch,z_score=True)

                x_zcr = x_znormed - np.mean(x_znormed)
                zcr = 0.5*np.mean([abs(np.sign(x_zcr[k+1]) - np.sign(x_zcr[k])) for k in range(len(x_zcr)-1)])

                # plt.plot(x_ch)
                # plt.title(str(zcr))
                # plt.show()



                S = np.fft.fft(x_ch)
                freq = np.fft.fftfreq(len(x_ch))
                freq = freq[:len(freq)//2]*sr
                x_fft = normalize(abs(S)[:len(S)//2])
                x_fft = x_fft/sum(x_fft)
                


                en_lf = np.sum(x_fft[freq<=1]**2)
                en_hf = np.sum(x_fft[np.logical_and(freq>1,freq<=5)]**2)
                # plt.subplot(211)
                # plt.plot(freq[freq<=2],x_fft[freq<=2])
                # plt.ylim((0,1))
                # plt.subplot(212)
                # plt.plot(freq[np.logical_and(freq>1,freq<=5)],x_fft[np.logical_and(freq>1,freq<=5)])
                # plt.ylim((0,1))
                # plt.show()

                # S_filt = np.fft.fft(x_filt)
                # freq_filt = np.fft.fftfreq(len(x_filt))
        


                # f0 = np.argmax(x_fft)


                # plt.subplot(121)
                # plt.plot(x_ch)
                # # plt.plot(x_filt)
                # # plt.title(str(i))


                # plt.subplot(122)
                # plt.plot(freq*sr,abs(S))

                # plt.show()
                
                # # plt.plot(freq_filt[:len(freq_filt)//2]*sr,abs(S_filt)[:len(freq_filt)//2])
                # plt.show()

                first_diff = [x_ch[k+1] - x_ch[k] for k in range(len(x)-1)]


                wavelet_coef = wavedec(x_ch,'db4',level=4)
                mean_dcoef = [np.mean(wcoef**2) for wcoef in wavelet_coef[1:]]


                features[i,ch*N_FEATURE_ACC:(ch+1)*N_FEATURE_ACC] = \
                    [np.std(x_ch),np.median(x_ch),#zcr,
                    np.min(x_ch),np.max(x_ch),np.sqrt(np.mean(x_ch**2)),
                    compute_entropy(x_ch),np.mean(first_diff),
                    en_lf,en_hf,np.std(first_diff),np.mean(wavelet_coef[0])] + mean_dcoef

                    # [np.mean(x_ch),np.std(x_ch),np.median(x_ch),zcr,
                    # np.min(x_ch),np.max(x_ch),np.percentile(x_ch,25),
                    # np.percentile(x_ch,75),np.sqrt(np.mean(x_ch**2)),
                    # compute_entropy(x_ch),compute_entropy(x_fft),
                    # int(np.sqrt(np.mean(x_fft**2))),
                    # 100/f0 if f0!=0 else 0,np.mean(first_diff),
                    # np.std(first_diff)] + mean_dcoef
                    


                    


            elif type_=='eda':

                x_znormed = normalize(x_ch,z_score=True)

   
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



            else:

                warnings.warn("Unknown sensor type")

                

    return np.array(features,dtype=np.float32)

def normalize(x,z_score=False,nmin=0,nmax=1):

    if len(x.shape)==1:
        x = x[:,np.newaxis]

    if z_score:
        x = (x - np.mean(x,axis=0))/(np.std(x,axis=0)+1e-10)
    else:
        x = nmin + nmax*(x - np.min(x,axis=0))/(np.ptp(x,axis=0)+1e-10)

    return x




def extract_window(data,type_):

    sr = data[1,0]
    data = data[2:]
    


    # data = resample(data,int(len(data)//downsample))
    # sr = int(sr//downsample)


    L = len(data)//(SIZE_W*sr)

    data = data[:int(L*SIZE_W*sr)]


    return compute_features(np.stack(np.array_split(data,L)),type_,sr)



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

            # plt.plot(data_acc[2+(SIZE_W*32)*6:2+(SIZE_W*32)*7])
            # plt.show()

            filt = butter(10,3,'low',fs=32)

            for k in range(3):
                data_acc[2:,k] = filtfilt(filt[0],filt[1],data_acc[2:,k],padlen=10)


            # plt.plot(data_acc[2+(SIZE_W*32)*6:2+(SIZE_W*32)*7])
            # plt.show()

   
            data_acc[2:,0] = np.sqrt(np.sum(normalize(data_acc[2:],z_score=True)**2,axis=1))
            data_acc = data_acc[:,0][:,np.newaxis]

            # plt.plot(data_acc[2+(SIZE_W*32)*6:2+(SIZE_W*32)*7])
            # plt.show()


            


            path = 'dataset/Data/'+i+'/'+s+'/EDA.csv'
            data_eda = pd.read_csv(path,header=None).to_numpy()

            # [phasic,_,tonic,_,_,_,_] = cvxEDA(data_eda[2:].T[0],1./data_eda[1,0]) 


            # data_eda_comp = np.concatenate((phasic.reshape(-1,1),tonic.reshape(-1,1)),axis=1)
            # data_eda_comp = np.concatenate((np.repeat(data_eda[:2],2,axis=1),data_eda_comp),axis=0)

            # z += [extract_window(data_acc[:,np.newaxis])]
            z += [np.concatenate((extract_window(data_acc,'acc'), 
                                  extract_window(data_eda,'eda')),axis=1)]
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

    def __init__(self,C=10,ngram=1,proj=False,tfidf=True,load_feature=False):

        f,index = build_feature_dataset(load=load_feature)

        self.features = f
        self.index = index

        self.num_words = C
        self.num_pattern = C
        self.proj_BoW = proj
        self.dictionnary = None

        self.behav_pattern = None

        self.ngram = ngram

        self.do_tfidf = tfidf

        self.idf = None


    def get_nested_index(self):
        return np.array([self.index[0]] + [self.index[k] - self.index[k-1][-1] for k in range(1,self.index.shape[0])])


    def ngrams2lin(self,label):

        index_lin = []
        for e in label:
            index = 0
            for i,l in enumerate(e):
                index += l*(self.num_words**(self.ngram-i-1))
     
            index_lin.append(index)

        return np.array(index_lin)

    def lin2ngrams(self,label):

        index_gram = []
        for e in label:
            index = []
            for i in range(self.ngram-1):
                index.append(int(e//self.num_words**(self.ngram-i-1)))
            index.append((e//self.num_words)%self.num_words)
     
            index_gram.append(tuple(index))

        return index_gram





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



    def compute_doc_term_matrix(self):
        label_id = np.array_split(self.behav_pattern,self.index.ravel())[:-1]
        term_per_doc = np.array([[len(l[l==k]) for k in range(self.num_words**self.ngram)] for l in label_id])

        non_zero = np.argwhere(np.sum(term_per_doc!=0,axis=0)!=0).ravel()

        distrib = np.sum(term_per_doc,axis=0)
        distrib = distrib/np.sum(distrib)
        top_pattern = np.argsort(distrib)[:-1]
        print(top_pattern)

        new_pattern = np.intersect1d(non_zero,top_pattern)
        
        

        compressed_term_matrix = np.squeeze(term_per_doc[:,new_pattern])


        return term_per_doc, compressed_term_matrix, new_pattern


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


        

        if self.ngram > 1:
            self.behav_pattern = self.ngrams2lin(list(ngrams(self.dictionnary.labels_,self.ngram)))
            _, _, new_pattern = self.compute_doc_term_matrix()
            self.num_pattern = len(new_pattern)
        else:
            self.behav_pattern = self.dictionnary.labels_



        if tfidf:
            _, c_term_matrix, _ = self.compute_doc_term_matrix()
            self.idf = 1 + np.log(1 + self.index.size/np.array([len(v[v!=0]) for v in c_term_matrix.T]))
        else:
            self.idf = np.ones(self.pattern)/self.num_pattern



        return self.behav_pattern


    def get_session_embedding(self,n_topic=64):

        _, c_term_matrix, _ = self.compute_doc_term_matrix()


        tf = np.array([v/np.sum(v) for v in c_term_matrix])

        tfidf = tf*np.tile(self.idf[np.newaxis,:],(tf.shape[0],1))

        # plt.subplot(121)
        # plt.pcolormesh(tfidf)
        # plt.subplot(122)
        # plt.pcolormesh(c_term_matrix)
        # plt.show()


        lda_input = c_term_matrix*tfidf

        topic_model = LatentDirichletAllocation(n_components=n_topic,random_state=42,max_iter=50)
        embedded = topic_model.fit_transform(lda_input)
        #print(embedded)
        # distance_embedded = [[jensenshannon(t1,t2) for t2 in embedded] for t1 in embedded]
        # plt.pcolormesh(distance_embedded)
        # plt.xticks(np.arange(0,31,3))
        # plt.yticks(np.arange(0,31,3))
        # plt.grid()
        # plt.colorbar()
        # plt.show()

        return embedded





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

    # inertia = []

    # for K in range(5,100,5):
    #     f_extract = FeatureAnalysis(tfidf=True,C=K,load_feature=True,ngram=2)
    #     f_extract.word_bagging()
    #     inertia.append(f_extract.dictionnary.inertia_)

    # plt.plot(list(range(5,100,5)),inertia)
    # plt.show()

    f_extract = FeatureAnalysis(tfidf=True,C=10,load_feature=True,ngram=2)
    C = f_extract.num_words
    
    f_extract.word_bagging()


    plt.hist(f_extract.behav_pattern,bins=np.arange(C**f_extract.ngram),density=True)
    plt.xlabel("Behavioral bi-grams")
    plt.ylabel("Frequency")
    plt.title("Histogram of behavioral patterns")


    

    plt.show()


    plt.figure()

    bigrams = f_extract.lin2ngrams(f_extract.behav_pattern)

    #idx = [(i,j) for i in range(C) for j in range(C)]# for k in range(C)]
    # histo_grams = [[sum([v==(i,j) for v in bigrams]) for j in range(C)] for i in range(C)]
    # identical = 0

    histo_grams = []
    identical = 0
    for i in range(C):
        row = []
        for j in range(C):
            if i==j:
                identical += len(f_extract.behav_pattern[f_extract.behav_pattern==C*i+j])
            row.append(len(f_extract.behav_pattern[f_extract.behav_pattern==C*i+j]))
        histo_grams.append(row)



    
    
    non_identical = 100*(1 - identical/len(f_extract.behav_pattern))
    # print(identical,len(f_extract.behav_pattern))
    print(f"\nRate of non-redondant pattern : {non_identical:.2f} %")
    plt.pcolormesh(np.array(histo_grams)/len(bigrams))
    plt.xticks(np.arange(1,C+1),labels=[i+1 for i in range(C)])
    plt.yticks(np.arange(1,C+1),labels=[i+1 for i in range(C)])
    plt.grid()
    plt.colorbar()
    plt.show()




    # term_per_doc = f_extract.compute_doc_term_matrix()
    # term_per_doc_new = term_per_doc[:,np.sum(term_per_doc!=0,axis=0)!=0]

    # print(f"\nCompression ratio : {term_per_doc.shape[1]/term_per_doc_new.shape[1]:.2f}")

    # new_idf = f_extract.idf[np.sum(term_per_doc!=0,axis=0)!=0]

    # tf = np.array([v/np.sum(v) for v in term_per_doc_new])
    # tfidf = tf*np.tile(new_idf[np.newaxis,:],(tf.shape[0],1))



    # plt.matshow(tfidf)
    # plt.matshow(term_per_doc_new*tfidf)
    # plt.colorbar()
    # plt.show()
    # document_term_matrix = np.zeros()

    
    #print(embedded.shape)
    embedded = f_extract.get_session_embedding(16)
    # plt.plot(f_extract.dictionnary.labels_[:index_0])

    # labels_id = labels#np.array_split(labels,10)
    # distance = [[np.linalg.norm(p1 - p2) for p1 in labels_id] for p2 in labels_id]

    # plt.matshow(distance)
    
    
    
    #z_session = LatentDirichletAllocation(n_components=2,random_state=42).fit_transform(normalize(labels))
    z_session = TSNE(n_components=2,perplexity=3,random_state=0,init='pca',learning_rate='auto').fit_transform(normalize(embedded,z_score=True))
    z_session = normalize(z_session,z_score=True)



    C = 4
    profile = KMeans(n_clusters=C,random_state=42).fit(z_session)


    

    fig = plt.figure()
    ax = Axes3D(fig)#fig.add_subplot(projection='3d')

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
    #plt.legend(loc='center left',bbox_to_anchor=(1.2, 0.5))
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("grade")
    fig.legend()

    # def rotate(angle):
    #     ax.view_init(azim=angle)

    # angle = 3
    # ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    # ani.save('bigram_C10.gif', writer=animation.PillowWriter(fps=20))
        


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

    