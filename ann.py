# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('index.csv')
X = dataset.iloc[:, 1:21].values
y = dataset.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras import backend as K
import os
from importlib import reload

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("theano")

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 20))

# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 9, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results

def predictPercentage(input):
	global classifier
	global X_test
	y_pred = classifier.predict(X_test)
	# y_pred_new = classifier.predict(sc.transform(np.array([[4, 11, 4, 3, 1154, 2, 1, 4, 2, 1, 4, 1, 57, 3, 2, 3, 2, 1, 1, 1]])))
	y_pred_new = classifier.predict(sc.transform(np.array([input])))
	y_pred_new = y_pred_new * 100
	print ("                             ")
	# print ("Your credit score in percentage is %d."%y_pred_new)
	return y_pred_new
	

#print ("Prediction directly from model is %f" %y_pred_new)

print ("                             ")
#y_pred_new = classifier.predict(sc.transform(np.array([[1, 10, 1, 48, 1, 60, 120, 1, 30, 739]])))  
#y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/dataSubmit', methods = ['POST'])
def parse_request():
	d1 = request.form.get('1')
	d2 = int(request.form.get('2'))
	d3 = int(request.form.get('3'))
	d4 = int(request.form.get('4'))
	d5 = int(request.form.get('5'))
	d6 = int(request.form.get('6'))
	d7 = int(request.form.get('7'))
	d8 = int(request.form.get('8'))
	d9 = int(request.form.get('9'))
	d10 = int(request.form.get('10'))
	d11 = int(request.form.get('11'))
	d12 = int(request.form.get('12'))
	d13 = int(request.form.get('13'))
	d14 = int(request.form.get('14'))
	d15 = int(request.form.get('15'))
	d16 = int(request.form.get('16'))
	d17 = int(request.form.get('17'))
	d18 = int(request.form.get('18'))
	d19 = int(request.form.get('19'))
	d20 = int(request.form.get('20'))
	d21 = int(request.form.get('21'))
	input = [d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21]
	# print predictPercentage(input)
	res = predictPercentage(input)
	score = 'undefined'
	if res[0][0] >=80:
		score = 'Excellent'
	elif res[0][0] >=60 and res[0][0] < 80:
		score = 'Good'
	elif res[0][0] >=50 and res[0][0] < 60:
		score = 'Poor'
	else:
		score = 'Bad'
	result = str(res[0][0])
	return '''<h1>Score : {0}</h1><p>Prediction : {1}</p>'''.format(score, result)

if __name__ == '__main__':
	app.run()
