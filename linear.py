"""linear.py"""
import pandas as pd
import matplotlib.pyplot as PLT
import numpy as np

DF = pd.read_csv("data/boston_housing.csv")

def train_model(INPUTS, print_result=True):
	INPUTS = np.concatenate((INPUTS, -np.ones((np.shape(INPUTS)[0], 1))), axis=1)
	TARGETS = DF.as_matrix(columns=['MEDIAN VALUE'])
	WEIGHTS = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(INPUTS), INPUTS)),
		                       np.transpose(INPUTS)), TARGETS)
	OUTPUTS = np.dot(INPUTS, WEIGHTS)
	ERROR = sum((TARGETS-OUTPUTS)**2)
	if print_result:
		print('Weights:')
		print(WEIGHTS)
		print('Error:')
		print(ERROR)
	return OUTPUTS, WEIGHTS, ERROR

#1 Low Status

INPUTS = DF.as_matrix(columns=['LOW STATUS'])
OUTPUTS, _, _ = train_model(INPUTS)
PLT.scatter(DF['LOW STATUS'], DF['MEDIAN VALUE'])
PLT.plot(DF['LOW STATUS'], OUTPUTS, c="orange")
PLT.show()


#2 Rooms

INPUTS = DF.as_matrix(columns=['ROOMS'])
print("2:")
OUTPUTS, _, _ = train_model(INPUTS)

PLT.scatter(DF['ROOMS'], DF['MEDIAN VALUE'])
PLT.plot(DF['ROOMS'], OUTPUTS, c="orange")
PLT.show()

#3 Rooms & Low Status

INPUTS = DF.as_matrix(columns=['ROOMS', 'LOW STATUS'])
print("3:")
train_model(INPUTS)


#4 Low & Low^2

DF['LOW STATUS^2'] = DF['LOW STATUS']**2
INPUTS = DF.as_matrix(columns=['LOW STATUS', 'LOW STATUS^2'])
print("4:")
OUTPUTS, _, _ = train_model(INPUTS)


#5 Rooms & Rooms^2

DF['ROOMS^2'] = DF['ROOMS']**2
INPUTS = DF.as_matrix(columns=['ROOMS', 'ROOMS^2'])
print("5:")
OUTPUTS, _, _ = train_model(INPUTS)


#6 Rooms, Low, Rooms^2, Low^2

INPUTS = DF.as_matrix(columns=['ROOMS', 'LOW STATUS', 'LOW STATUS^2', 'ROOMS^2'])
print("6:")
OUTPUTS, _, _ = train_model(INPUTS)


#7 Linear reg with Rooms, Low, Rooms^2, Low^2. & A 'Low Rooms' column

DF['LOW ROOMS'] = DF['LOW STATUS'] * DF['ROOMS']
INPUTS = DF.as_matrix(columns=['ROOMS', 'LOW STATUS', 'LOW STATUS^2', 'ROOMS^2', 'LOW ROOMS'])
print("7:")
OUTPUTS, _, _ = train_model(INPUTS)

#8 Add one of the other colums to number six

COLUMN_LIST = list(DF.columns)
VALID_INPUTS = list(set(COLUMN_LIST) - set(['ROOMS', 'LOW STATUS', 'LOW STATUS^2',
	                                           'ROOMS^2', 'MEDIAN VALUE']))

MIN_ERROR = None
MIN_WEIGHTS = None
VAL = None

for v in VALID_INPUTS:
	INPUTS = DF.as_matrix(columns=['ROOMS', 'LOW STATUS', 'LOW STATUS^2', 'ROOMS^2', v])
	_, WEIGHTS, ERROR = train_model(INPUTS, False)
	#print(v)
	#print(ERROR)
	if MIN_ERROR is None or MIN_ERROR > ERROR:
		MIN_ERROR = ERROR
		MIN_WEIGHTS = WEIGHTS
		VAL = v

#Prop Tax Rate, Afr Amer, and River changed the error by about 400.

print('8:')
print(VAL)
print('Weights:')
print(MIN_WEIGHTS)
print('Error:')
print(MIN_ERROR)

#9 Find MPG with AUTO_MPG dataset

AUTO_MPG = pd.read_csv("data/AUTO_MPG.csv")
AUTO_MPG['horsepower^2'] = AUTO_MPG['horsepower']**2
AUTO_MPG['1/horsepower^2'] = 1 / (AUTO_MPG['horsepower']**2)

INPUTS = AUTO_MPG.as_matrix(columns=['1/horsepower^2'])
INPUTS = np.concatenate((INPUTS, -np.ones((np.shape(INPUTS)[0], 1))), axis=1)
TARGETS = AUTO_MPG.as_matrix(columns=['mpg'])
WEIGHTS = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(INPUTS), INPUTS)),
	                       np.transpose(INPUTS)), TARGETS)
OUTPUTS = np.dot(INPUTS, WEIGHTS)
ERROR = sum((TARGETS-OUTPUTS)**2)

print("9:")
print('Weights:')
print(WEIGHTS)
print('Error:')
print(ERROR)

PLT.scatter(AUTO_MPG['horsepower'], AUTO_MPG['mpg'])
PLT.scatter(AUTO_MPG['horsepower'], OUTPUTS, c="orange")
PLT.show()
