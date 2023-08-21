# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Here the basic neural network model has been created with one input layer, one hidden layer and one output layer.The number of neurons(UNITS) in each layer varies the 1st input layer has 16 units and hidden layer has 8 units and output layer has one unit.

In this basic NN Model, we have used "relu" activation function in input and hidden layer, relu(RECTIFIED LINEAR UNIT) Activation function is a piece-wise linear function that will output the input directly if it is positive and zero if it is negative.

## Neural Network Model


![261610854-8320fd72-384f-4400-b792-6fc4543a653b](https://github.com/naramala-niharika/basic-nn-model/assets/94165377/1b550b62-b38c-42c4-a7f0-61146f53ad18)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.
``
### PROGRAM:

## Importing modules

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## Authenticate & Create data frame using data in sheets

from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('ex1').sheet1
data = worksheet.get_all_values()
dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype({'input':'float'})
dataset1=dataset1.astype({'output':'float'})
dataset1.head()

## Assign X & Y Values

X = dataset1[['input']].values
y = dataset1[['output']].values
X

## Normalize the values and split the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
## Create a neural network and train it.
ai_brain=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=200)

## Plot the loss

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

## Predict for some value

X_test1 = Scaler.transform(X_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)



## Dataset Information:

![261318641-989cd3fb-98b7-40a4-9194-178d1c9cf83b](https://github.com/naramala-niharika/basic-nn-model/assets/94165377/7146e241-de92-4137-a861-58dc8c8233bb)
## OUTPUT:

### Training Loss Vs Iteration Plot:

![261318823-76268b7a-8fae-4ff6-bef6-eba02a8e1887](https://github.com/naramala-niharika/basic-nn-model/assets/94165377/ae7a2e9f-b8f2-4161-8d43-0ebaef65804d)


### Test Data Root Mean Squared Error:

![261319159-74bb9f89-13a8-4c31-94fe-0873558afa8a](https://github.com/naramala-niharika/basic-nn-model/assets/94165377/2404cb52-9315-40cd-a2de-6454352c4420)

### New Sample Data Prediction:

![261319444-09027049-c0d9-4595-92a1-c15ea28e272f](https://github.com/naramala-niharika/basic-nn-model/assets/94165377/e4ff8dfc-c36d-422b-8065-7d04691fbd11)

## RESULT :

A Basic neural network regression model for the given dataset is developed successfully.
