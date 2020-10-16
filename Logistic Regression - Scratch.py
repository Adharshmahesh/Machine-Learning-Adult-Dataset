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
#sns.heatmap(adult[list(adult.dtypes.index)].corr(),annot = True,square = True);
# Use one-hot encoding on categorial columns
adult = pd.get_dummies(adult, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country'],drop_first=True)
#adult
#print(adult['income'])
#print(adult)

class Logistic_Regression:

	def __init__(self, w):
		self.w = w

	def sigmoidal(self, l):
	
		return (1/(1+np.exp(-l)))
	
	def gradient(self, xtrain, ytrain):
		z = self.sigmoidal(np.transpose(self.w).dot(np.transpose(xtrain)))
		deltaW = np.transpose(xtrain).dot(np.transpose(ytrain - z))

		return deltaW
		

	def fit(self, xtrain, ytrain, lr=0.01, iter=50000, eps=0.001, normal = True):

		
		cost = 0
		cost1 =list()
		const = 1e-5 #To prevent zero
		ytrain = ytrain[np.newaxis]
		numiter = 1
		#print(ytrain.shape)
		for i in range(iter):
			
			cost = -1*(ytrain.dot(np.log(self.sigmoidal(np.transpose(w).dot(np.transpose(xtrain))).T+const)) + (1-ytrain).dot(np.log(1-self.sigmoidal(np.transpose(self.w).dot(np.transpose(xtrain))).T+const)))
			cost1.append(cost)

			g = self.gradient(xtrain,ytrain)
			
			self.w = self.w+(lr * g)
			
			if(np.linalg.norm(g)<eps):
				break
			numiter = numiter + 1
		
		return self.w

	def predict(self, xtest, normal = True):
		
		pred = (self.sigmoidal((self.w.T.dot(xtest.T))))
		for i in range(len(pred)):
			for j in range(len(pred[i])):
				if pred[i][j]==0:
					pred[i][j] = 0
				elif pred[i][j] ==1:
					pred[i][j] = 1
				elif pred[i][j]<0.5:
					pred[i][j] = 0
				else:
					pred[i][j] = 1 
		
		
		return pred
		

	def calc_accuracy(self, ytest, pred):
		

		return np.mean(ytest == pred)
		 

	def conf_matrix(self, ytest, pred):

		cm = np.zeros(shape = (2,2))

		for i in range(len(pred)):
			for j in range(len(pred[i])):
				if ytest[i] == 0:
					if pred[i][j] == 0:
							cm[1][1] += 1
					else:
						cm[1][0] += 1

				elif ytest[i] == 1:
					if pred[i][j] == 1:
						cm[0][0] += 1

					else:
						cm[0][1] += 1
		positive = cm[0][0] + cm[1][0]
		negative = cm[0][1] + cm[1][1]

		accuracy_cm = (cm[0][0] + cm[1][1]) / (positive + negative)
		#precision = cm[0][0] / (cm[0][0] + cm[0][1])
		#recall = cm[0][0] / positive
		#f_measure = (2*recall*precision)/ (recall + precision)
		
		return accuracy

	def crossvalidation(self, xtrain, ytain, k, alpha=0.01 , iter=50000, eps = 0.01):
		
		size = int(len(xtrain)/k)
		cv_accuracy = 0

		for i in range(k):

			valstart = i*size
			valend = valstart + size

			if i!=(k-1):
				valend = size

				xval = xtrain[:valend,:]
				yval = ytrain[:valend]

				kxtrain = xtrain[valend:,:]
				kytrain = ytrain[valend:]

			else:
		
				xval = xtrain[valstart:,:]
				yval = ytrain[valstart:]

				kxtrain = xtrain[:valstart,:]
				kytrain = ytrain[:valstart]

				kxtrain = np.concatenate((xtrain[:valstart,:],xtrain[valend:,:]), axis = 0)
				kytrain = np.concatenate((ytrain[:valstart],ytrain[valend:]))

			w_kfold = self.fit(kxtrain, kytrain, alpha, iter)
			
			predy = self.predict(xval)
			
			cv_accuracy = cv_accuracy+self.calc_accuracy(yval, predy)
			print(cv_accuracy)

		cv_accuracy = cv_accuracy / k

		return cv_accuracy

a = np.ones((len(adult),1),dtype = int)
adult.insert(0,0,a, True)

	
train_data = adult.sample(frac = 0.8)

xtrain = np.array(train_data.drop(columns = ['income']))
ytrain = np.array(train_data['income'])	
test_data = adult.drop(train_data.index)
xtest = np.array((test_data.drop(columns = ['income'])))
ytest = np.array((test_data['income']))
	
w = np.array(np.transpose(np.zeros((xtrain.shape[1]))[np.newaxis]))
LR = Logistic_Regression(w)
w = LR.fit(xtrain, ytrain)


pred = LR.predict(xtest)
	
accuracy = LR.calc_accuracy(ytest,pred)
print("Accuracy is:", accuracy)

#Function call for k-fold validation
#accuracy_kfold = LR.crossvalidation(xtrain, ytrain, 5)
#print("Accuracy using k-fold is:", accuracy_kfold)
#accuracy_cm, precision, recall, f_measure= LR.conf_matrix(ytest, pred)

#print("Accuracy using confusion matrix is:", accuracy_cm)
#print("Precision is:", precision)
#print("Recall is:", recall)
#print("F - measure is:", f_measure)

#Alpha and Number of Iterations plot
'''
alpha_vector = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
accuracy_cv = []

for j in alpha_vector:

	w = np.array(np.transpose(np.zeros((xtrain.shape[1]))[np.newaxis]))
	LR = Logistic_Regression(w)
	accuracy_cv.append(LR.crossvalidation(xtrain, ytrain, 5, j, 500))
print(accuracy_cv)
plt.plot(alpha_vector, accuracy_cv, '.-')
plt.title('Accuracy vs Learning rate alpha for adult data ')
plt.xlabel('Learning rate')
plt.ylabel('Accuracy using cross validation')
plt.show()'''

#No of instances and accuracy plot
'''
instance_vector = [100, 500, 1000, 3000, 5000, 7000]
accuracy = []

for k in instance_vector:
	xtr = np.array(train_data.iloc[:k,:-1])
	ytr = np.array(train_data.iloc[:k,-1])
	xte = np.array((test_data.iloc[:k,:-1]))
	yte = np.array((test_data.iloc[:k,-1]))
	w = np.array(np.transpose(np.zeros((xtrain.shape[1]))[np.newaxis]))
	LR = Logistic_Regression(w)
	w = LR.fit(xtr, ytr)
	p = LR.predict(xte)
	a1 = LR.calc_accuracy(yte, p)
	accuracy.append(a1)
	print(accuracy)

print(accuracy)'''