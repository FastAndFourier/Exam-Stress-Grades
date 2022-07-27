import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from Dataset import *
from ProfileMining import get_grades

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(2)

class Classifcation():

    def __init__(self,label_cat,fold=1) -> None:
        
        dataset = Dataset()

        self.feature = dataset.feature
        self.index = np.insert(dataset.index.ravel(),0,0)
        self.grades = np.array(list(map(int,get_grades().ravel())))
        

        self.folds = fold
        self.split_features()   
        
   

        self.l_cat = label_cat
        self.labels = np.block([np.array([int(g)]*(self.index[i+1]-self.index[i]))  for i,g in enumerate(self.grades)])
        self.discretize_grades()


        self.clf = RandomForestClassifier(criterion='entropy',n_estimators=200)
        

    def split_features(self):

        index = []

        for i in range(len(self.index)-1):

            split_ = np.array_split(np.arange(self.index[i],self.index[i+1]),self.folds)
            split_ = split_ if split_[-1].shape[0] >= int(0.75*split_[0].shape[0]) else split_[:-1]
            
            for s in split_:
                index += s.shape #.append(sum([s.shape[0] for s in split_]))

        
        index = np.array([0]+index)
        
        
        for i in range(1,len(index)):
            index[i] = index[i]+index[i-1]


        self.index = index
        self.grades = np.repeat(self.grades,self.folds)
        


    def discretize_grades(self):

        id_label = []

        for i in range(len(self.l_cat)-1):

            low_limit = np.argwhere(self.grades>=self.l_cat[i])
            up_limit = np.argwhere(self.grades<self.l_cat[i+1])

            id_label.append(np.intersect1d(low_limit,up_limit))
            self.labels[(self.labels>=self.l_cat[i]) & (self.labels<self.l_cat[i+1])] = i

        print(id_label)
        plt.bar(np.arange(0,len(self.l_cat)-1),[len(l) for l in id_label],align='edge')
        plt.show()

        
   


    def training_dataset(self,test_index) -> list:

        train_index = np.setdiff1d(np.arange(30),test_index)


        X_train = np.vstack([self.feature[self.index[i]:self.index[i+1]] for i in train_index])
        y_train = np.block([self.labels[self.index[i]:self.index[i+1]] for i in train_index])
        

        # X_test = np.vstack([self.feature[self.index[i]:self.index[i+1]] for i in test_index])
        y_test = np.block([self.labels[self.index[i]:self.index[i+1]] for i in test_index])


            

       
        print("Train set size = ",X_train.shape[0],"| Test set size",y_test.shape[0])

        # print("Test set's students:",end=" ")
        # print([f"Student {1 + i//3} Session {1 + i%3}" for i in test_index])

        return X_train, y_train, train_index



    def fit(self,X,y):

        self.clf.fit(X,y)


    def kFoldCV(self,k=5,mixed=False):

        if not mixed:
            
            fold_shuffle = np.block([np.arange(i*3,i*3+3) for i in np.random.permutation(np.arange(10))])
            if k not in [5,10]:
                print("Incorrect number of folds (should be 5 or 10 to keep triplets)")
                return None
        else:
            fold_shuffle = np.random.permutation(np.arange(len(self.index)-1))
            

        print(len(fold_shuffle))
        folds = np.array_split(fold_shuffle,k)



        accuracy = []
        agreement = []
        agreement_true = []

        for i,test_index in enumerate(folds):
            print()
            print("-"*(os.get_terminal_size().columns//2-2),end="")
            print("Fold",str(i),end="")
            print("-"*(os.get_terminal_size().columns//2-3))
            X_train, y_train, _  = self.training_dataset(test_index)
            print()
            print(f"Distribution of labels in the training set = ",[f"{100*len(y_train[y_train==l])/len(y_train):.2f}" for l in np.unique(self.labels)])
            self.clf.fit(X_train,y_train)

            acc, agr, agr_true = self.classification_report(test_index)

            accuracy.append(acc)
            agreement.append(agr)
            agreement_true.append(agr_true)


            self.clf = RandomForestClassifier(criterion='entropy',n_estimators=200)


        print(f"Accuracy: mean = {np.mean(accuracy)*100:.2f} | std = {np.std(accuracy)*100:.2f}")
        print(f"Agreement: mean = {np.mean(agreement):.2f} | std = {np.std(agreement):.2f}")
        print(f"Agreement on true predictions: mean = {np.mean(agreement_true):.2f} | std = {np.std(agreement_true):.2f}")


    def predict(self,y_true,y_pred):

        return np.argmax(np.bincount(y_pred))==np.argmax(np.bincount(y_true))




    def classification_report(self,id_) -> list:

        print()

        agreement= []
        accuracy = []

        for i in range(len(self.index)-1):
            if i in id_:
                l = self.labels[self.index[i]:self.index[i+1]]
                pred = self.clf.predict(self.feature[self.index[i]:self.index[i+1]])

                l_pred = np.argmax(np.bincount(pred))
                
                #print("[Train] |" if i in id_train else "[Test]  |",end=" ")

                prediction_res = self.predict(l,pred)

                # print(f"{i},{3*self.folds}")

                print(f"Student {1 + i//(3*self.folds)} Session {1 + (i%(3*self.folds))%3}",end="\t")
                print("Classification:",prediction_res,end="\t")
                print(f"Predicted grade: [{self.l_cat[l_pred]}:{self.l_cat[l_pred+1]}]","| True grade:",self.grades[i],end=" | ")
                
              

                agreement.append(np.max(np.bincount(pred))*100/len(pred))
                accuracy.append(prediction_res)

                print(f"Level of agreement: {np.max(np.bincount(pred))*100/len(pred):.2f}")

        accuracy = np.array(accuracy)
        agreement = np.array(agreement)

        true_agreement = agreement[accuracy==True] if any(accuracy) else 0


        print()
        print(f"Mean level of agreement: {np.mean(agreement):.2f} %")
        print(f"Mean level of agreement for true predictions: {np.mean(true_agreement):.2f} %")
        print(f"Accuracy on test set: {np.sum(accuracy)*100/len(accuracy):.2f} % (Random guess = {100/(len(self.l_cat)-1):.2f} %)")
        print()

    
        return np.mean(accuracy), np.mean(agreement), np.mean(true_agreement)


if __name__ == "__main__":

    l_cat = [0,70,80,100]
    model = Classifcation(l_cat,fold=4)


    model.kFoldCV(k=10,mixed=True)

    # id_test = np.array([0,1,2,15,16,17,24,25,26])#np.array([25,16,10,0,3,27,7])
    # X_train, y_train, id_train = model.training_dataset(id_test)


    # model.fit(X_train,y_train)

    # print("")
    # print("="*os.get_terminal_size().columns)
    # print("-"*(os.get_terminal_size().columns//2 - 20)+\
    #     "Test set classification summary"+"-"*(os.get_terminal_size().columns//2 - 10))
    # print("="*os.get_terminal_size().columns)
    # print("")

    # model.classification_report(id_test)


    # print("")
    # print("="*os.get_terminal_size().columns)
    # print("-"*(os.get_terminal_size().columns//2 - 21)+\
    #     "Train set classification summary"+"-"*(os.get_terminal_size().columns//2 - 10))
    # print("="*os.get_terminal_size().columns)
    # print("")
    
    # model.classification_report(id_train)

    



    """ Regression """
    


    # model_lin = LinearRegression().fit(X_train,y_train)
    # y_pred = model_lin.predict(X_test)
    # l1_error = abs(y_pred-y_test)


    # print(np.mean(l1_error),np.std(l1_error),np.max(l1_error),np.min(l1_error))

    # model_svm = LinearSVR().fit(X_train,y_train)
    # y_pred = model_svm.predict(X_test)
    # l1_error = abs(y_pred-y_test)


    # print(np.mean(l1_error),np.std(l1_error),np.max(l1_error),np.min(l1_error))

    """ Classification """


    # model_svm = RandomForestClassifier().fit(X_train,y_train)
    # y_pred = model_svm.predict(X_test)

    # print()
    # print(confusion_matrix(y_test,y_pred))
    # print()
    
    
    # agreement= []
    # accuracy = []

    # for i in range(len(index)-1):
    #     if i in id_test:
    #         l = labels[index[i]:index[i+1]]
    #         pred = model_svm.predict(feature[index[i]:index[i+1]])

    #         l_pred = np.argmax(np.bincount(pred))
            
    #         print("[Train] |" if i in id_train else "[Test]  |",end=" ")
    #         print("Classification:",np.argmax(np.bincount(pred))==np.argmax(np.bincount(l)),end="\t")
    #         print(f"Predicted grade: [{l_cat[l_pred]}:{l_cat[l_pred+1]}]","| True grade:",get_grades().ravel()[i],end=" | ")
    #         agreement.append(np.max(np.bincount(pred))*100/len(pred))


    #         accuracy.append(np.argmax(np.bincount(pred))==np.argmax(np.bincount(l)))

    #         print(f"Level of agreement: {np.max(np.bincount(pred))*100/len(pred):.2f}")

    # accuracy = np.array(accuracy)
    # agreement = np.array(agreement)

    # print()
    # print(f"Mean level of agreement: {np.mean(agreement):.2f} %")
    # print(f"Mean level of agreement for true predictions: {np.mean(agreement[accuracy==True]):.2f} %")
    # print(f"Accuracy on test set: {np.sum(accuracy)*100/len(accuracy):.2f} %")
    # print()
   

    """-------------------------------------------------------------------------------------------"""


    # dataset = Dataset(modalities=['eda','hr','acc'],load=1)

    # feature = dataset.feature
    # index = dataset.index.ravel()

    # index = np.insert(index,0,0)

    # labels = [np.array([int(g)]*(index[i+1]-index[i]))  for i,g in enumerate(get_grades().ravel())] #+ np.random.randint(-1,1,size=(index[i+1]-index[i]))
    # labels = np.block(labels)

    # # plt.hist(labels,bins=np.arange(0,100,step=2))
    # # plt.show()

    # print(np.percentile(labels,25),np.percentile(labels,50),np.percentile(labels,75))

    # grade = np.array(list(map(int,get_grades().ravel())))


    # l_cat = [0,60,70,80,90,100]#[0,60,65,70,75,80,85,90,100]
    # id_label = []
    # for i in range(len(l_cat)-1):
    #     id_label.append(np.intersect1d(np.argwhere(grade>=l_cat[i]),np.argwhere(grade<l_cat[i+1])))
    #     labels[(labels>=l_cat[i]) & (labels<l_cat[i+1])] = i

    # print(id_label)

    # id_test = np.array([25,16,10,0,3,24,7])#np.array([19,20,25,16,18,12,26,11,0,10,14,9,3,1,22,27,4,5,8,23])
    # id_train = np.setdiff1d(np.arange(30),id_test)

    # print("Test set's students:")
    # print([f"Student {1 + i//3} Session {1 + i%3}" for i in id_test])

    # X_train = np.vstack([feature[index[i]:index[i+1]] for i in id_train])
    # y_train = np.block([labels[index[i]:index[i+1]] for i in id_train])

    # X_test = np.vstack([feature[index[i]:index[i+1]] for i in id_test])
    # y_test = np.block([labels[index[i]:index[i+1]] for i in id_test])

    # #X_train, X_test, y_train, y_test = train_test_split(feature,labels,test_size=0.3,random_state=42)

    # print("Train set size = ",X_train.shape[0],"| Test set size",X_test.shape[0])

    # plt.bar(np.arange(0,len(l_cat)-1),[len(l) for l in id_label])
    # plt.show()