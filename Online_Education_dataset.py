
#====================
# imports
#====================
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, show
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from seaborn import countplot
from sklearn.model_selection import train_test_split

# class which contain classification algorithm function and will predict the suitable Algorithm
class AlgorithmPredictor:
    def __init__(self, data_train, data_test, target_train,
                 target_test):  # init method accepting testing and training data
        self.data_train = data_train  # instance variable
        self.data_test = data_test
        self.target_train = target_train
        self.target_test = target_test
        self.AlgoDict = {}

    # Training and Testing of Data for Decision Tree
    def DecisionTree(self):
        cobj = tree.DecisionTreeClassifier()
        cobj.fit(self.data_train, self.target_train)
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test, output)
        print("Accuracy of Decision Tree : ",Accuracy*100)
        self.AlgoDict["Decision Tree"] = Accuracy * 100

    # Training and Testing of Data for KNN
    def KNN(self):
        cobj = KNeighborsClassifier(n_neighbors=7)
        cobj.fit(self.data_train, np.ravel(self.target_train))
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test, output)
        print("Accuracy of KNN : ", Accuracy * 100)
        self.AlgoDict["K Nearest Neighbour"] = Accuracy * 100

    # Training and Testing of Data for Naive Bayes
    def NaiveBayes(self):
        cobj = GaussianNB()
        cobj.fit(self.data_train, self.target_train)
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test, output)
        print("Accuracy of NaiveBayes : ", Accuracy * 100)
        self.AlgoDict["Naive Bayes"] = Accuracy * 100

    # Training and Testing using Logistic Regression
    def LogisticRegression1(self):
        cobj = LogisticRegression(max_iter=2000)
        cobj.fit(self.data_train, np.ravel(self.target_train))
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test, output)
        print("Accuracy of LogisticRegression : ", Accuracy * 100)
        self.AlgoDict["Logistic Regression"] = Accuracy * 100

    # Training and Testing using Random Forest
    def RandomForest(self):
        cobj = RandomForestClassifier(n_estimators=100)
        cobj.fit(self.data_train, np.ravel(self.target_train))
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test, output)
        print("Accuracy of RandomForest : ", Accuracy * 100)
        self.AlgoDict["Random Forest Classifier"] = Accuracy * 100

    # Method to prdict algorithm
    def predictAlgorithm(self):
        self.DecisionTree()
        self.KNN()
        self.NaiveBayes()
        self.LogisticRegression1()
        self.RandomForest()

        max_key = max(self.AlgoDict, key=self.AlgoDict.get)

        print("Preferable Algorithm for your Data Set is: ", max_key)

def main():
    line = "*"*50

    # loading dataset
    dataset = pd.read_csv("Online_Education_4.csv")
    print("First five rows of dataset : ")
    print(dataset.head())
    print(line)

    # performing exploratory data analysis on the dataset
    # 1 : first step would be knowing about the datatypes
    print("Datatypes of the attributes : ")
    print(dataset.dtypes)
    print(line)
    print("Information of dataset")
    print(dataset.info())
    print(line)

    # statistical analysis on ordinal datatypes present in the dataset
    # analysing physical health of students
    print("Visulization of Online Education Dataset : ")
    print(line)

    figure()
    countplot(data=dataset, x="Preferance").set_title("Visualization of Online education preference")
    show()

    # analysing area from which students belong
    print("Visualization according to State : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="State").set_title("Visualization according to State")
    show()

    # analysing the courses from which the students belong
    print("Visualization according to education : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="education").set_title("Visualization according to education")
    show()

    # analysing gender
    print("Visualization according to Gender : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="gender").set_title("Visualization according to Gender")
    show()

    # analysing according to area
    print("Visualization according to area: ")
    figure()
    countplot(data=dataset, x="Preferance", hue="area").set_title("Visualization according to area")
    show()

    # analysing according to device
    print("Visualization according to device : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="device").set_title("Visualization according to device")
    show()

    # analysing according to internet service provider
    print("Visualization according to isp : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="isp").set_title("Visualization according to isp")
    show()

    # analysing according to networking device
    print("Visualization according to networkingdevice : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="networkingdevice").set_title("Visualization according to networkingdevice")
    show()

    # analysing according to powercut
    print("Visualization according to powercut : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="powercut").set_title("Visualization according to powercut")
    show()

    # analysing according to powerbackup
    print("Visualization according to powerbackup : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="powerbackup").set_title("Visualization according to powerbakup")
    show()

    # analysing according to hour
    print("Visualization according to hour : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="hour").set_title("Visualization according to hour")
    show()

    # analysing according to phi_ear
    print("Visualization according to phi_ear : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="phi_ear").set_title("Visualization according to phi_ear")
    show()

    # analysing according to phi_eye
    print("Visualization according to phi_eye : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="phi_eye").set_title("Visualization according to phi_eye")
    show()

    # analysing according to phi_joint
    print("Visualization according to phi_joint : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="phi_joint").set_title("Visualization according to phi_joint")
    show()

    # analysing according to phi_obesity
    print("Visualization according to phi_obesity : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="phi_obesity").set_title("Visualization according to phi_obesity")
    show()

    # analysing according to device_hunged
    print("Visualization according to device_hunged : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="device_hunged").set_title("Visualization according to device_hunged")
    show()

    # analysing according to answering_question
    print("Visualization according to answering_question : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="answering_question").set_title("Visualization according to answering_question")
    show()

    # analysing according to completing_assignment
    print("Visualization according to completing_assignment : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="completing_assignment").set_title("Visualization according to completing_assignment")
    show()

    # analysing according to submitting_assignment
    print("Visualization according to submitting_assignment : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="submitting_assignment").set_title("Visualization according to submitting_assignment")
    show()

    # analysing according to understanding_online
    print("Visualization according to understanding_online : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="understanding_online").set_title("Visualization according to understanding_online")
    show()

    # analysing according to lab_problem
    print("Visualization according to lab_problem : ")
    figure()
    countplot(data=dataset, x="Preferance", hue="lab_problem").set_title("Visualization according to lab_problem")
    show()

    # converting the object datatypes into numeric values
    # cols = dataset.columns[dataset.dtypes.eq('object')]
    # dataset[cols] = dataset[cols].apply(pd.to_numeric, errors='coerce')
    print("Encoded Data")
    dataset.drop(['college','name','State','area'], axis=1, inplace=True)
    cols = pd.get_dummies(dataset, drop_first=True,
                          columns=['education', 'gender', 'device', 'isp', 'networkingdevice', 'powerbackup',
                                   'phi_ear', 'phi_eye', 'phi_joint', 'phi_obesity', 'Preferance'])

    print(cols)
    print(line)

    X_Data =  cols.iloc[:,0:20]
    Y_Data = cols.iloc[:,20]

    #np.isnan(X_Data)

    X_train, X_test, Y_train, Y_test = train_test_split(X_Data,Y_Data,random_state=0,test_size=0.20)
    obj = AlgorithmPredictor(X_train, X_test, Y_train, Y_test)
    obj.predictAlgorithm()

if __name__=="__main__":
    main()
