import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf


# In[271]:


def model(num_units, dropout = 0.0):
    
    classifier = Sequential()
    classifier.add(Dense(4*num_units,activation='relu',kernel_initializer='uniform', input_dim = 4039))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(2*num_units, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(num_units, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    return classifier


# In[272]:


classifier = model(5, 0.3)


# In[273]:


optimizer = keras.optimizers.Adam(lr = 0.0001)


# In[274]:


classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])


# In[275]:


classifier.fit(xTrain,yTrain,batch_size = 128, epochs = 200)


# In[276]:


y_p = classifier.predict(xTrain)


# In[277]:


y_p_1 = classifier.predict(xval)


# In[278]:


error = 0
#y_pred_1 = y_pred[:, 1]
j = 0
for i in yTrain:
    #print(j)
    error1 = i - y_p[j][0]
    #print(str(i) + " " + str(y_pred_1[j])+" " + str(abs(error1)))
    error = error + abs(error1)
    j = j +1

mse = error/len(y_p)
print(mse)
print(1 /(1+mse))


# In[279]:


error = 0
#y_pred_1 = y_pred[:, 1]
j = 0
for i in yval:
    #print(j)
    error1 = i - y_p_1[j][0]
    #print(str(i) + " " + str(y_pred_1[j])+" " + str(abs(error1)))
    error = error + abs(error1)
    j = j +1

mse = error/len(y_p)
print(mse)
print(1 /(1+mse))


# In[280]:


y_p_2 = classifier.predict(X_test)


# In[281]:


print(y_p_2)


# In[299]:


print(print(id_test))

print(len(y_p_2))
id_test.count()


# In[301]:


id_test.head()


# In[313]:


ids = id_test.iloc[:].values
print(ids)


# In[323]:


v_id = []
j = 0

for i in ids:
    if not pd.isnull(i):
        v_id.append(j)
    j = j+1


# In[324]:


print(v_id)


# In[325]:


pout = y_p_2[v_id]
iout = ids[v_id]


# In[326]:


predictions = pd.DataFrame(pout, columns=['is_goal'])
idss = pd.DataFrame(iout, columns=['shot_id_number'])
predictions = pd.concat((idss, predictions), axis = 1)
predictions.to_csv('result1.csv', sep=",", index = False)


