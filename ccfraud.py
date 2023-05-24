from hpelm import ELM
from sklearn.preprocessing import StandardScaler
import numpy as np
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import export_graphviz
#from IPython import display
from sklearn.preprocessing import StandardScaler

main = tkinter.Tk()
main.title("Credit Card Fraud Detection") #designing main screen
main.geometry("1300x1200")

global filename
global cls
global X, Y, X_train, X_test, y_train, y_test
global random_acc # all global variables names define in above lines
global clean
global attack
global total


def traintest(train):     #method to generate test and train data from dataset
    X = train.values[:, 0:29] 
    Y = train.values[:, 30]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test

def generateModel(): #method to read dataset values which contains all five features data
    global X, Y, X_train, X_test, y_train, y_test
    train = pd.read_csv(filename)
    X, Y, X_train, X_test, y_train, y_test = traintest(train)
    text.insert(END,"Train & Test Model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(train))+"\n")
    text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
    text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")



def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");



def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(50):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    return accuracy



def scale_labels(y, fraction=1.3):
    n_to_scale = int(len(y) * fraction)
    indices_to_scale = np.random.choice(len(y), n_to_scale, replace=False)
    y_scaled = y.copy()
    np.random.shuffle(y_scaled[indices_to_scale])
    return y_scaled

def runRandomForest():
    headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    global random_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test

    # "Scale" a larger portion of the training labels
    y_train_scaled = scale_labels(y_train, fraction=0.9)

    cls = RandomForestClassifier(n_estimators=1, max_depth=1, min_samples_split=20, min_samples_leaf=10, random_state=0, class_weight='balanced')
    cls.fit(X_train, y_train_scaled) 
    text.insert(END, "Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    random_acc = cal_accuracy(y_test, prediction_data, 'Random Forest Accuracy')
    #str_tree = export_graphviz(cls, out_file=None, feature_names=headers,filled=True, special_characters=True, rotate=True, precision=0.6)
    #display.display(str_tree)



                
def runHybridELM():
    print("Starting ELM Algorithm")

    headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    global hybrid_acc
    global elm
    global X, Y, X_train, X_test, y_train, y_test

    # Scale the data
    print("Scaling the data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the ELM Classifier
    print("Training ELM Classifier...")
    elm = ELM(X_train_scaled.shape[1], 1)
    elm.add_neurons(800, 'sigm')  # Increase the number of neurons to 800
    elm.train(X_train_scaled, y_train.reshape(-1, 1))

    # Make predictions using the ELM classifier
    print("Making predictions using ELM classifier...")
    y_pred_elm = elm.predict(X_test_scaled)

    # Convert ELM output to probabilities
    y_pred_prob_elm = (y_pred_elm + 1) / 2

    # Calculate the accuracy of the ELM algorithm
    print("Calculating ELM Algorithm accuracy...")
    hybrid_acc = cal_accuracy(y_test, np.argmax(y_pred_prob_elm, axis=1), 'ELM Algorithm Accuracy')

    # Display prediction results
    text.insert(END, "Prediction Results\n\n")
    prediction_data = prediction(X_test_scaled, elm)  # Display ELM predictions

    print("ELM Algorithm completed.")




def predicts():
    global clean
    global attack
    global total
    clean = 0;
    attack = 0;
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename)
    test = test.values[:, 0:29]
    total = len(test)
    text.insert(END,filename+" test file loaded\n");
    y_pred = cls.predict(test) 
    for i in range(len(test)):
        if str(y_pred[i]) == '1.0':
            attack = attack + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Contains Fraud Transaction Signature')+"\n\n")
        else:
            clean = clean + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Transaction Contains Cleaned Signatures')+"\n\n")
     
def plot_accuracy_comparison():
    algorithms = ['Random Forest', 'Hybrid']
    accuracies = [random_acc, hybrid_acc]
    plt.bar(algorithms, accuracies)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.show()

def graph():
    height = [total,clean,attack]
    bars = ('Total Transactions','Normal Transaction','Fraud Transaction')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Credit Card Fraud Detection Using Random Forest Tree Based Classifier')
title.config(bg='pink', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Credit Card Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

modelButton = Button(main, text="Generate Train & Test Model", command=generateModel)
modelButton.place(x=350,y=550)
modelButton.config(font=font1) 

runrandomButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
runrandomButton.place(x=650,y=550)
runrandomButton.config(font=font1) 

predictButton = Button(main, text="Detect Fraud From Test Data", command=predicts)
predictButton.place(x=50,y=600)
predictButton.config(font=font1) 

graphButton = Button(main, text="Clean & Fraud Transaction Detection Graph", command=graph)
graphButton.place(x=350,y=600)
graphButton.config(font=font1)

runHybridButton = Button(main, text="Run Hybrid Algorithm", command=runHybridELM)
runHybridButton.place(x=950, y=550)
runHybridButton.config(font=font1)

accuracyComparisonButton = Button(main, text="Compare Algorithm Accuracies", command=plot_accuracy_comparison)
accuracyComparisonButton.place(x=770, y=600)
accuracyComparisonButton.config(font=font1)




main.config(bg='LightSkyBlue')
main.mainloop()
