import numpy as np
import matplotlib.pyplot as plt

from Dataset import Dataset

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-load',help="Load or create dataset",type=int,default=1,choices=[0,1])
parser.add_argument('-mod',help="Modalities to use if load=False",type=str,nargs='*',choices=['eda','acc','hr'])
parser.add_argument('-wsize',help="Window size to use if load=False",type=int,default=30)


parser.add_argument('-C',help="Number of cluster for the bag of words",type=int,default=60)
parser.add_argument('-lemm',help="Perform lemmatization",type=int,default=1,choices=[0,1])
parser.add_argument('-ngram',help="n_grams' length",type=int,default=2)

parser.add_argument('-T',help="Number of topic for the LDA",type=int,default=64)
parser.add_argument('-K',help="Number of final clusters (profile)",type=int,default=4)
parser.add_argument('-tfidf',help="Use tfidf as LDA's input, tf otherwise",type=int,default=1,choices=[0,1])

from nltk.util import ngrams
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import jensenshannon, cosine
from scipy.stats import pointbiserialr, pearsonr, chisquare

def normalize(x,z_score=False,nmin=0,nmax=1):

    if len(x.shape)==1:
        x = x[:,np.newaxis]

    if z_score:
        x = (x - np.mean(x,axis=0))/(np.std(x,axis=0)+1e-10)
    else:
        x = nmin + nmax*(x - np.min(x,axis=0))/(np.ptp(x,axis=0)+1e-10)

    return x


def ngrams2lin(label,C,ngram):

    index_lin = []
    for e in label:
        index = 0
        for i,l in enumerate(e):
            index += l*(C**(ngram-i-1))
    
        index_lin.append(index)

    return np.array(index_lin)

def lin2ngrams(label,C,ngram):

    index_gram = []
    for e in label:
        index = []
        for i in range(ngram-1):
            index.append(int(e//C**(ngram-i-1)))
        index.append(e%C)
    
        index_gram.append(tuple(index))

    return index_gram


def get_grades():
    f = open("StudentGrades.txt")
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

    return grades

class ProfileMining:

    def __init__(self,dataset:Dataset,args:argparse.ArgumentParser()):


        """
            C = number of cluster for the bag of words
            n_gram = n_grams' length

            T = number of topic for the LDA
            K = number of final clusters (profile)
        """


        self.features = dataset.feature
        self.index = dataset.index

        self.C = args.C
        self.ngram = args.ngram
        self.word_model = KMeans(n_clusters=self.C,random_state=42)
        #SpectralClustering(n_clusters=self.C,affinity='precomputed',random_state=42)
        #
        self.dictionary = None

        self.topic_model = None
        self.use_tfidf = args.tfidf
        self.T = args.T

        self.profile_model = None
        self.K = args.K 


    def bag_of_words(self,lemm):

        z_embedded = normalize(self.features,z_score=True)
       
        self.word_model.fit(z_embedded)
        
   

        if self.ngram > 1: 
            if lemm:
                new_label = self.lemmatization()
                wl = [new_label[l] for l in self.word_model.labels_]          
                words = ngrams2lin(list(ngrams(wl,self.ngram)),self.C,self.ngram)  
            else:
                words = ngrams2lin(list(ngrams(self.word_model.labels_,self.ngram)),self.C,self.ngram)
            self.word_model.labels_ = words 
            self.dictionary = np.unique(words)

        else:
            if lemm:
                self.dictionary = np.unique(new_label)
            else:
                self.dictionary = np.arange(self.C)




    def lemmatization(self):

        w = self.word_model.cluster_centers_
        new_label = []
        dist = []

        for i in range(self.C):
            d = [np.linalg.norm(w[i]-w[j]) for j in range(self.C)]
            dist.append(np.mean(d))
            new_label.append(np.argsort(d)[1])


        limit = np.percentile(dist,25)
        new_label = np.array(new_label)
        new_label[dist>limit] = np.argwhere(dist>limit).ravel()
        
        for i in range(self.C):
            
            for j in range(self.C):

                if new_label[j]!= j and new_label[new_label[j]]!= new_label[j]:
                    new_label[j] = new_label[new_label[j]]


        return new_label

        

        # closest_w = [np.argsort(d)[1] for d in distance_center]
        # closest_d = [distance_center[i,c] for i,c in enumerate(closest_w)]

        # np.percentile(closest_d,25)

        # plt.pcolormesh(distance_center)
        # plt.gca().set_aspect('equal')
        # plt.colorbar()
        # plt.show()


    def compute_tf(self):

        if self.dictionary is None:
            raise Exception("Bag of word hasn't initialized yet!")

        label_id = np.array_split(self.word_model.labels_,self.index.ravel())[:-1]
        tf = np.array([[len(l[l==k]) for k in self.dictionary] for l in label_id])


        return tf

        


    def remove_low_occ(self,occ=1):
        
        tf = self.compute_tf()
        freq = np.array([len(w[w!=0]) for w in tf.T])

        low_occ_index = np.argwhere(freq<=occ).ravel()

        self.dictionary = np.delete(self.dictionary,low_occ_index)

        return self.compute_tf()



    def remove_top_words(self,n_top=1):

        tf = self.compute_tf()
        max_freq = np.argsort((np.sum(tf,axis=0)))[-n_top:]
        self.dictionary = np.delete(self.dictionary,max_freq)

        return self.compute_tf()
    

    def remove_redundant_pattern(self):

        redundant_idx = []

        for i,l in enumerate(self.dictionary):
            ngram = lin2ngrams([l],self.C,self.ngram)[0]
            if ngram[0]==ngram[1]:
                redundant_idx.append(i)
        
        self.dictionary = np.delete(self.dictionary,redundant_idx)

    
        return self.compute_tf()

    def compute_tfidf(self):

        tf = self.compute_tf()
        idf = 0.5 + np.log(0.5 + self.index.size/np.array([len(v[v!=0]) for v in tf.T]))

        return tf*np.tile(idf[np.newaxis,:],(tf.shape[0],1)) 




    def topic_modelling(self):

    
    
        if self.use_tfidf:
            lda_input = self.compute_tfidf()
        else: 
            lda_input = self.compute_tf()

        print("Number of words in the dictionnary :",lda_input.shape[1])

        self.topic_model = LatentDirichletAllocation(
                        n_components=self.T,random_state=42,
                        max_iter=10,learning_method="online",
                        learning_offset=50.0)

        
        # lda_input = np.apply_along_axis(lambda x: (x-np.min(x))/(np.ptp(x)+1e10),axis=1,arr=lda_input)
        
        # plt.pcolormesh(np.repeat(lda_input,5,axis=0))
        # plt.gca().set_aspect('equal')
        # plt.show()
        profile = self.topic_model.fit_transform(lda_input)


        return profile


    def profile_clustering(self,topics):


        f = lambda x: np.exp(x)/np.sum(np.exp(x))
        #topics = self.topic_model.labels_
        #affinity_matrix = np.array([[np.exp(-jensenshannon(f(p1),f(p2))) for p1 in topics] for p2 in topics])

        affinity_matrix = np.array([[np.exp(-jensenshannon(p1,p2)) for p1 in topics] for p2 in topics])

        self.profile_model = SpectralClustering(n_clusters=self.K,affinity='precomputed',random_state=42)
        labels = self.profile_model.fit_predict(affinity_matrix)

        return labels
    
if __name__ == "__main__":


    args = parser.parse_args()
    

    """ Training the model """

    d = Dataset(modalities=args.mod,load=args.load,size_win=args.wsize)
    model = ProfileMining(dataset=d,args=args)

    model.bag_of_words(lemm=bool(args.lemm))


    session = np.array_split(model.word_model.labels_,model.index.ravel())[:-1]
    dist_p = [[jensenshannon(np.histogram(s1,bins=128)[0],np.histogram(s2,bins=128)[0]) for s1 in session] for s2 in session]


    #model.remove_redundant_pattern()
    #model.remove_low_occ(occ=1)
    #model.remove_top_words(n_top=1)


    if model.ngram==2:
        hist_gram = np.zeros((model.C,model.C))
        labels_ = model.word_model.labels_
        # for i in range(model.C):
        #     for j in range(model.C):
        #         hist_gram[i,j] = len(labels_[labels_==(i*model.C+j)])
        for l in model.dictionary:
            index = lin2ngrams([l],model.C,model.ngram)[0]
            hist_gram[index] = len(labels_[labels_==l])
            


        plt.pcolormesh(hist_gram, edgecolor='k')
        plt.gca().set_aspect('equal')
        plt.show()


    topics = model.topic_modelling()
    profile = model.profile_clustering(topics)

    


    """ Tests and visualization """
    
    grades = get_grades()

    print(grades.ravel()[6],grades.ravel()[9])

    f = lambda x: np.exp(x)/np.sum(np.exp(x))
    f_sum = lambda x : x/np.sum(x)



    normed_topics = np.apply_along_axis(f_sum,axis=1,arr=model.topic_model.components_)#model.topic_model.components_/np.sum(model.topic_model.components_,axis=1)[:,np.newaxis]
    H_topics = np.apply_along_axis(lambda x: -np.sum(x*np.log2(x)),1,normed_topics)

    #ormed_profile = [np.sort(t)[-5:] for t in topics]#np.apply_along_axis(f,axis=1,arr=topics)#topics/np.sum(topics,axis=1)[:,np.newaxis]
    normed_profile = np.apply_along_axis(f_sum,axis=1,arr=topics)
    H_profile = np.apply_along_axis(lambda x: -np.sum(x*np.log2(x)),1,normed_profile)



    ## LDA components' H
    uniform_d = np.ones(len(model.dictionary))/len(model.dictionary)
    max_H = -np.sum(uniform_d*np.log2(uniform_d))


    plt.figure()
    plt.bar(np.arange(model.T),H_topics,align='edge')
    plt.plot(np.arange(-1,model.T+1),[max_H]*(model.T+2),'r--')
    plt.annotate(r'$H_{max}$ = '+f'{max_H:.3f}',
                (model.T-5,np.max(H_topics)+0.5),fontsize=10,color='red',weight="bold")
    plt.title("LDA component's entropy")
    plt.xlabel("Topic")
    plt.ylabel(r'$H$')

    ## Sessions topics' H

    plt.figure()

    stride = max(H_profile)/40
    for k in range(30):
        plt.annotate(str(profile[k]),(k+0.2,H_profile[k]+stride),color='green',weight='bold')

    plt.bar(np.arange(30),H_profile,align='edge')
    plt.annotate(r'$H_{max}$ = '+f'{-np.sum(np.ones(30)/30*np.log2(np.ones(30)/30)):.3f}',
                (27.5,np.max(H_profile)),fontsize=15,color='red',weight="bold")
    plt.title("Sessions topics' entropy")
    plt.xlabel("Session")
    plt.ylabel(r'$H$')
    for l in np.arange(0,30,3):
        plt.axvline(l-0.1,ls='--',color='grey',lw=1)





    ## H(grades)

    grades = np.array(list(map(int,grades.ravel())))
    ordered = np.argsort(grades)
 

    lim_ = np.median(grades)

    grades_bin = grades.copy()
    grades_bin[grades>lim_] = 1
    grades_bin[grades<=lim_] = 0

    print(pointbiserialr(grades_bin,H_profile),pearsonr(grades[ordered],H_profile[ordered]))


    # H_profile = (H_profile - np.mean(H_profile))/np.std(H_profile)
    
    lin_model = LinearRegression().fit(grades[:,np.newaxis],H_profile)
    
    print(min(H_profile))
    plt.figure()
    # for k in range(model.K):
    #     plt.scatter(grades[ordered][profile==k],H_profile[ordered][profile==k],s=20,label=str(k))
    # plt.legend()

    plt.scatter(grades[ordered],H_profile[ordered],s=20)
    x = np.arange(min(grades),max(grades),0.01)
    y = lin_model.predict(x.reshape(-1,1))
    
    plt.plot(x,y,'r')

    plt.xlabel("Grade")
    plt.ylabel("Topics' entropy")
    
    plt.grid()



    plt.show()