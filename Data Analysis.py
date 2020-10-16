#import libraries
import pandas as pd
import numpy as np
#matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
#giving headers to column
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
sns.heatmap(adult[list(adult.dtypes.index)].corr(),annot = True,square = True);
plt.show()
# Use one-hot encoding on categorial columns
adult = pd.get_dummies(adult, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country'],drop_first=True)

corr = adult.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.5:
            print(i,j)
            
        else:
            #print(i)
            continue
#Correlation with output variable
cor_target = abs(corr.iloc[:,-1])#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)

sns.boxplot(data=adult['hours-per-week'])
plt.show()