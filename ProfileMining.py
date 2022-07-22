import numpy as np
import matplotlib.pyplot as plt

from Dataset import Dataset

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-load',help="Load or create dataset",type=int,default=1)
parser.add_argument('-mod',help="Modalities to use if load=False",type=str,nargs='*',choices=['eda','acc','hr'])
parser.add_argument('-wsize',help="Window size to use if load=False",type=int,default=30)


parser.add_argument('-C',help="Number of cluster for the bag of words",type=int,default=60)
parser.add_argument('-ngram',help="n_grams' length",type=int,default=2)
parser.add_argument('-T',help="Number of topic for the LDA",type=int,default=64)
parser.add_argument('-K',help="Number of final clusters (profile)",type=int,default=4)
parser.add_argument('-tfidf',help="Use tfidf as LDA's input, tf otherwise",type=int,default=1)

from nltk.util import ngrams
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import jensenshannon

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
        index.append((e//C)%C)
    
        index_gram.append(tuple(index))

    return index_gram


def get_grades():
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
        self.dictionary = None

        self.topic_model = None
        self.use_tfidf = args.tfidf
        self.T = args.T

        self.profile_model = None
        self.K = args.K 


    def bag_of_words(self):

        z_embedded = normalize(self.features,z_score=True)
        self.word_model.fit(z_embedded)
        
        if self.ngram > 1:
            words = ngrams2lin(list(ngrams(self.word_model.labels_,self.ngram)),self.C,self.ngram)  
            self.word_model.labels_ = words   
            self.dictionary = np.unique(words)

        else:
            self.dictionary = np.arange(self.C)


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
                        max_iter=5,learning_method="online",
                        learning_offset=50.0)

        profile = self.topic_model.fit_transform(lda_input)


        return profile


    def profile_clustering(self,topics):

        #topics = self.topic_model.labels_
        affinity_matrix = np.array([[np.exp(-jensenshannon(p1,p2)) for p1 in topics] for p2 in topics])
        
        self.profile_model = SpectralClustering(n_clusters=self.K,affinity='precomputed',random_state=42)
        labels = self.profile_model.fit_predict(affinity_matrix)

        return labels
    
if __name__ == "__main__":


    args = parser.parse_args()
    

    """ Training the model """

    d = Dataset(modalities=args.mod,load=args.load,size_win=args.wsize)
    model = ProfileMining(dataset=d,args=args)

    model.bag_of_words()
    model.remove_low_occ()
    model.remove_top_words()


    topics = model.topic_modelling()
    profile = model.profile_clustering(topics)



    """ Tests and visualization """
    
    grades = get_grades()
    
    normed_topics = model.topic_model.components_/np.sum(model.topic_model.components_,axis=1)[:,np.newaxis]
    H_topics = np.apply_along_axis(lambda x: -np.sum(x*np.log2(x)),1,normed_topics)

    normed_profile = topics/np.sum(topics,axis=1)[:,np.newaxis]
    H_profile = np.apply_along_axis(lambda x: -np.sum(x*np.log2(x)),1,normed_profile)



    ## LDA components' H






    ## Sessions topics' H

    plt.figure()

    for k in range(30):
        plt.annotate(str(profile[k]),(k+0.2,H_profile[k]+0.02),color='green',weight='bold')

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


    plt.figure()
    # for k in range(model.K):
    #     plt.scatter(grades[ordered][profile==k],H_profile[ordered][profile==k],s=20,label=str(k))
    # plt.legend()

    plt.scatter(grades[ordered],H_profile[ordered],s=20)


    plt.xlabel("Grade")
    plt.ylabel("Topics' entropy")
    
    plt.grid()



    plt.show()