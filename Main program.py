#Importing statements
import tensorflow as tf
import numpy as np

# Loading and filtering data
raw_data = open('Raw data.txt','r')
lines = raw_data.readlines()
X=[]
CD1=[]
CD2=[]
CL1=[]
CL2=[]
for line in lines:
    x1,x2,x3,cd1,cd2,cl1,cl2 = map(float,line.split(' '))
    x = [x1,x2,x3]
    X.append(x)
    CD1.append(cd1)
    CD2.append(cd2)
    CL1.append(cl1)
    CL2.append(cl2)

#visualizing data
print(X,CD1,CD2,CL1,CL2)

# Converting data into numpy arrays
x_train=np.asarray(X)
cd1_train=np.asarray(CD1)
cd2_train=np.asarray(CD2)
cl1_train=np.asarray(CL1)
cl2_train=np.asarray(CL2)

#visualising data
for a ,b,c,d ,e in zip(x_train,cd1_train,cd2_train,cl1_train,cl2_train):
    print(a,b,c,d,e)

# Creating a sequential model
model=tf.keras.Sequential([
    tf.keras.layers.Dense(3,input_shape=[3],activation='sigmoid'),
    tf.keras.layers.Dense(12,activation='sigmoid'),
    tf.keras.layers.Dense(1)
])
# compiling the model using loss and optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(0.1),loss=tf.keras.losses.mean_squared_error)

# we fit the model four different times to predict cd1, cd2, cl1, cl2 individually
#model.fit(x_train,cd1_train,epochs=1000,validation_split=0.2)
model.fit(x_train,cd2_train,epochs=10000,validation_split=0.2)
#model.fit(x_train,cl1_train,epochs=1000,validation_split=0.2)
#model.fit(x_train,cl2_train,epochs=1000,validation_split=0.2)

#and filnally we predict the crack locations and crack depths 
pradict=np.asarray([[0.9519,0.9577,0.954]])
print(model.predict(pradict))







