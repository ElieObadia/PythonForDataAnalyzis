from sklearn import datasets
import pickle
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  
y = iris.target

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)

# Save the model as a pkl file
filename = 'ml_model/iris_model.pkl'
pickle.dump(clf, open(filename, 'wb'))

data = pd.read_csv("ml_model/diabetic_data.csv")
data.drop(['encounter_id','patient_nbr'], axis=1, inplace=True)
data.drop(['weight'],axis=1,inplace=True)
data.drop(['payer_code', 'medical_specialty'],axis=1,inplace=True)
for col in data.columns:
    if len(data[col].unique()) == 1:
        data.drop(col,inplace=True,axis=1)
data.replace(['?'],[np.nan], inplace=True)
data = data.dropna()
data.replace(["[0-10)", "[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"],[0,10,20,30,40,50,60,70,80,90], inplace = True)
data = data[data['gender']!='Unknown/Invalid']
data['gender'].replace(['Male','Female'],[0,1],inplace = True)
data['max_glu_serum'].replace(['None', '>300', 'Norm', '>200'],[-1,300,0,200], inplace =True)
data['race'].replace(['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic'],[1,2,3,4,5], inplace = True)
id_column = ['insulin','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

for i in id_column:
  data[i].replace(['No','Up','Down','Steady'],[0,1,2,3],inplace = True)

data['change'].replace(['No','Ch'],[0,1], inplace = True)
data['diabetesMed'].replace(['No','Yes'],[0,1], inplace = True)
data['A1Cresult'].replace(['None','>8','Norm','>7'],[0,8,1,7], inplace = True)
data.replace(['NO','<30','>30'],[0,1,0], inplace=True)

data.drop(['diag_1','diag_2','diag_3'], axis = 1, inplace =True)
data_copy = data.copy()
dataX=data_copy.drop(['readmitted'],axis=1)
dataY=data_copy['readmitted']
X_train, X_test, Y_train, Y_test = sk.train_test_split(dataX, dataY, test_size=0.20, random_state=0)

logReg = LogisticRegression()
logReg.fit(X_train, Y_train)

logit_pred = logReg.predict(X_test)
acc_logreg = accuracy_score(Y_test, logit_pred)
print(acc_logreg)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, Y_train)
tree_pred = tree.predict(X_test)

acc_tree = accuracy_score(Y_test, tree_pred)
print(acc_tree)

# Save the model as a pkl file
filename = 'ml_model/diab_model_log_reg.pkl'
pickle.dump(logReg, open(filename, 'wb'))

filename = 'ml_model/diab_model_dtree.pkl'
pickle.dump(tree, open(filename, 'wb'))
#save train as a csv file
f = open("ml_model/acc_logreg.txt", "w")
f.write(str(acc_logreg))
f.close()
print('acc_logreg compile')
np.savetxt("ml_model/logit_pred.csv", logit_pred, delimiter=',')
print('logit_pred compile')

f = open("ml_model/acc_tree.txt", "w")
f.write(str(acc_tree))
f.close()
print('acc_tree compile')
np.savetxt("ml_model/tree_pred.csv", tree_pred, delimiter=',')
print('tree_pred compile')

Y_test.to_csv('ml_model/Ytest.csv')

print('File compile')
