import numpy as np
import matplotlib.pyplot as plt

from Dataset import *
from ProfileMining import get_grades

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier



# def session_functionnal(f):

#     func_ = []

#     for s in f:

#         func.append





if __name__ == "__main__":


    dataset = Dataset(modalities=['eda','hr','acc'],load=1)

    feature = dataset.feature
    index = dataset.index.ravel()

    index = np.insert(index,0,0)

    labels = [np.array([int(g)]*(index[i+1]-index[i]))  for i,g in enumerate(get_grades().ravel())] #+ np.random.randint(-1,1,size=(index[i+1]-index[i]))
    labels = np.block(labels)

    # plt.hist(labels,bins=np.arange(0,100,step=2))
    # plt.show()

    print(np.percentile(labels,25),np.percentile(labels,50),np.percentile(labels,75))

    grade = np.array(list(map(int,get_grades().ravel())))


    l_cat = [0,60,70,80,90,100]#[0,60,65,70,75,80,85,90,100]
    id_label = []
    for i in range(len(l_cat)-1):
        id_label.append(np.intersect1d(np.argwhere(grade>=l_cat[i]),np.argwhere(grade<l_cat[i+1])))
        labels[(labels>=l_cat[i]) & (labels<l_cat[i+1])] = i

    print(id_label)

    id_test = np.array([0,1,9,11,21,23,27,29])#np.array([19,20,25,16,18,12,26,11,0,10,14,9,3,1,22,27,4,5,8,23])
    id_train = np.setdiff1d(np.arange(30),id_test)

    X_train = np.vstack([feature[index[i]:index[i+1]] for i in id_train])
    y_train = np.block([labels[index[i]:index[i+1]] for i in id_train])

    X_test = np.vstack([feature[index[i]:index[i+1]] for i in id_test])
    y_test = np.block([labels[index[i]:index[i+1]] for i in id_test])

    #X_train, X_test, y_train, y_test = train_test_split(feature,labels,test_size=0.3,random_state=42)

    print("Train set size = ",X_train.shape[0],"| Test set size",X_test.shape[0])

    plt.bar(np.arange(0,len(l_cat)-1),[len(l) for l in id_label])
    plt.show()

    



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


    grade_label = ["<60","60 -> 65","65 -> 70","70 -> 75","75 -> 80",
                   "80 -> 85","85 -> 90",">90"]
    
    
    
    #["Under 60","Between 60 and 75","Between 75 and 85","Above 85"]


    model_svm = RandomForestClassifier().fit(X_train,y_train)
    y_pred = model_svm.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print()
    
    
    agreement= []
    accuracy = []

    for i in range(len(index)-1):
        if i in id_test:
            l = labels[index[i]:index[i+1]]
            pred = model_svm.predict(feature[index[i]:index[i+1]])

            l_pred = np.argmax(np.bincount(pred))
            predicted_ = grade_label[l_pred]
            print("[Train] |" if i in id_train else "[Test]  |",end=" ")
            print("Classification:",np.argmax(np.bincount(pred))==np.argmax(np.bincount(l)),end="\t")
            print(f"Predicted grade: [{l_cat[l_pred]}:{l_cat[l_pred+1]}]","| True grade:",get_grades().ravel()[i],end=" | ")
            agreement.append(np.max(np.bincount(pred))*100/len(pred))


            accuracy.append(np.argmax(np.bincount(pred))==np.argmax(np.bincount(l)))

            print(f"Level of agreement: {np.max(np.bincount(pred))*100/len(pred):.2f}")

    print()
    print(f"Mean level of agreement: {np.mean(agreement):.2f} %")
    print(f"Accuracy: {np.sum(accuracy)*100/len(accuracy):.2f} %")
   