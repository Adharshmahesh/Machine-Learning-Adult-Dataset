import numpy as np
import pandas as pd
import scipy
#from loaddata import data
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

columns= ['age','workclass','fnlwgt','education','education-num','marital-status',
                    'occupation','relationship','race','sex','capital-gain','capital-loss',
                    'hours-per-week','native-country','income']

#loading dataset
adult= pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',sep=",",names=columns, header= None)
#to consider cat data as type category
for col in set(adult.columns) - set(adult.describe().columns):
    adult[col] = adult[col].astype('category')
#adult.info()
#display last few rows
#adult.tail()
#display first few rows 
#adult.head(30)
#since this column is not much usefull for analysis
adult=adult.drop('fnlwgt', axis=1)
#adult
#making target variable simple in terms of 0 and 1
adult['income'] =[0 if x==' <=50K' else 1 for x in adult['income']]
adult.shape #shape after converting target as 0 and 1
# Remove invalid data from table
adult= adult[(adult.astype(str) != ' ?').all(axis=1)]
adult.shape #shape after removal of ?
#Separate categorical and numberical columns
cat= adult.dtypes[adult.dtypes == 'object']
num= adult.dtypes[adult.dtypes != 'object']
#sns.heatmap(adult[list(adult.dtypes.index)].corr(),annot = True,square = True);
# Use one-hot encoding on categorial columns
adult = pd.get_dummies(adult, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country'],drop_first=True)
#print(adult.iloc[:,8])
x=adult.drop(['income'], axis = 1)
y=adult['income']
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#w =  0.1*np.ones(x.shape[1])

from sklearn.linear_model import LogisticRegression   
# create logistic regression object 
reg = LogisticRegression() 
   
# train the model using the training sets 
reg.fit(x_train, y_train) 
  
# making predictions on the testing set 
y_pred = reg.predict(x_test) 

w = reg.coef_ 
# comparing actual response values (y_test) with predicted response values (y_pred) 
#print(w)
print("Logistic Regression model accuracy(in %):",  
metrics.accuracy_score(y_test, y_pred)*100) 


