
# coding: utf-8

# In[116]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[117]:


dataset = pd.read_csv('data.csv')


# In[118]:


dataset.head()


# In[119]:


dataset.shape


# In[120]:


dataset.describe()


# In[121]:


dataset.team_name.unique()


# In[122]:


dataset.type_of_shot.unique()


# In[123]:


dataset.type_of_shot.count()


# In[124]:


dataset.team_id.unique()


# In[125]:


df = dataset.drop(columns = "team_id")
df = df.drop(columns = "team_name")


# In[126]:


df.shape


# In[127]:


df['home/away'].nunique()


# In[128]:


df.columns


# In[129]:


df['knockout_match'].nunique()


# In[130]:


df['knockout_match.1'].describe()


# In[131]:


df['power_of_shot.1'].unique()


# In[132]:


df['remaining_min.1'].describe()


# In[133]:


df2 = df.drop(columns = ['remaining_min.1', 'power_of_shot.1', 'knockout_match.1',
       'remaining_sec.1', 'distance_of_shot.1'])


# In[134]:


df2.shape


# In[135]:


df2['match_event_id'].nunique()


# In[136]:


df2['match_id'].count()


# In[137]:


label_encoder = LabelEncoder()
df2['match_id'] = label_encoder.fit_transform(df2['match_id'])


# In[138]:


df2['match_id'].unique()


# In[139]:


print(df2['match_id'][0])


# In[140]:


mapl = {}
mapha = {}


for i in range(0,df2.shape[0]):
    key = df2['match_id'][i]
    if key not in mapl:
        if not pd.isnull(df2['lat/lng'][i]):
            mapl[key] = df2['lat/lng'][i]
            
    if key not in mapha:
        if not pd.isnull(df2['home/away'][i]):
            mapha[key] = df2['home/away'][i]
    

print(mapl)
print(mapha)
    


# In[141]:


print(df2['lat/lng'].count())
print(df2['home/away'].count())


# In[143]:



for i in range(0,df2.shape[0]):
    if pd.isnull(df2['lat/lng'][i]):
        key = df2['match_id'][i]
        df2['lat/lng'][i] = mapl[key]
    if pd.isnull(df2['home/away'][i]):
        key = df2['match_id'][i]
        df2['home/away'][i] = mapha[key]
        


# In[144]:


print(df2['lat/lng'].count())
print(df2['home/away'].count())


# In[145]:


label_en_lat = LabelEncoder()
df2['lat/lng'] = label_en_lat.fit_transform(df2['lat/lng'])


# In[146]:


df2.head()


# In[147]:


df3 = df2


# In[148]:


symb = []
t2 = []

for i in range(0, df3.shape[0]):
    symb.append(df3['home/away'][i].split()[1])
    t2.append(df3['home/away'][i].split()[2])

df3['symb'] = symb
df3['t2'] = t2


# In[149]:


df3


# In[150]:


df3.loc[158 , :]


# In[151]:


label_en_ha = LabelEncoder()
label_en_symb = LabelEncoder()
label_en_t2 = LabelEncoder()

df3['home/away'] = label_en_ha.fit_transform(df3['home/away'])
df3['symb'] = label_en_symb.fit_transform(df3['symb'])
df3['t2'] = label_en_t2.fit_transform(df3['t2'])


# In[152]:


df3.loc[158, :]


# In[153]:


mapd = {}

for i in range(0,df3.shape[0]):
    key = df3['match_id'][i]
    if key not in mapd:
        if not pd.isnull(df3['date_of_game'][i]):
            mapd[key] = df3['date_of_game'][i]


# In[154]:


print(label_encoder.inverse_transform(1083))


# In[155]:


mapd[1080] = '1996-11-04'


# In[156]:


for i in range(0,df3.shape[0]):
    if pd.isnull(df3['date_of_game'][i]):
        key = df3['match_id'][i]
        df3['date_of_game'][i] = mapd[key]


# In[157]:


year = []
month = []
day = []

for i in range(0,df3.shape[0]):
    date = df3['date_of_game'][i].split('-')
    year.append(date[0])
    month.append(date[1])
    day.append(date[2])
    
df3['year'] = year
df3['month'] = month
df3['day'] = day


# In[158]:


df3.head()


# In[159]:


mapgs = {}

for i in range(0,df3.shape[0]):
    key = df3['match_id'][i]
    if key not in mapgs:
        if not pd.isnull(df3['game_season'][i]):
            mapgs[key] = df3['game_season'][i]
            


# In[160]:


print(label_encoder.inverse_transform(1123))


# In[161]:


mapgs[1083] = '1996-97'


# In[162]:


mapgs[1123] = '1996-97'


# In[164]:


for i in range(0,df3.shape[0]):
    if pd.isnull(df3['game_season'][i]):
        key = df3['match_id'][i]
        df3['game_season'][i] = mapgs[key]


# In[165]:


df3.to_csv("processed.csv", index=False, encoding='utf8')


# In[166]:


time_rem = []

for i in range(0,df3.shape[0]):
    mint = df3['remaining_min'][i]
    sec = df3['remaining_sec'][i]
    time_rem.append(60*mint + sec)

df3['time_rem'] = time_rem


# In[167]:


df3.head()


# In[168]:


df3.to_csv("processed.csv", index=False, encoding='utf8')


# In[169]:


df3 = df3.drop(columns = 'Unnamed: 0')


# In[170]:


df3.head()


# In[171]:


df4 = pd.read_csv('processed.csv')


# In[172]:


mapkm = {}

for i in range(0,df4.shape[0]):
    key = df4['match_id'][i]
    if key not in mapkm:
        if not pd.isnull(df4['knockout_match'][i]):
            mapkm[key] = df4['knockout_match'][i]


# In[173]:


for i in range(0,df4.shape[0]):
    if pd.isnull(df4['knockout_match'][i]):
        key = df4['match_id'][i]
        df4['knockout_match'][i] = mapkm[key]


# In[174]:


df4.to_csv("processed1.csv", index=False, encoding='utf8')


# In[175]:


df4.head()


# In[176]:


df3.columns


# In[177]:


df4['m_id_null'] = pd.isnull(df4['match_event_id'])
df4['l_x_null'] = pd.isnull(df4['location_x'])
df4['l_y_null'] = pd.isnull(df4['location_y'])
df4['rem_min_null'] = pd.isnull(df4['remaining_min'])
df4['rem_sec_null'] = pd.isnull(df4['remaining_sec'])
df4['p_shot_null'] = pd.isnull(df4['power_of_shot'])
df4['d_shot_null'] = pd.isnull(df4['distance_of_shot'])
df4['a_shot_null'] = pd.isnull(df4['area_of_shot'])
df4['b_shot_null'] = pd.isnull(df4['shot_basics'])
df4['r_shot_null'] = pd.isnull(df4['range_of_shot'])
df4['t_shot_null'] = pd.isnull(df4['type_of_shot'])
df4['c_shot_null'] = pd.isnull(df4['type_of_combined_shot'])


# In[178]:


df4.head()


# In[179]:


df4.to_csv("processed2.csv", index=False, encoding='utf8')


# In[180]:


test = []
train = []

for i in range(0, df4.shape[0]):
    if pd.isnull(df4['is_goal'][i]):
        test.append(i)
    else :
        train.append(i)


# In[181]:


df_test = df4.loc[test, :]
df_train = df4.loc[train, :]


# In[182]:


df_train.to_csv("train.csv", index=False, encoding='utf8')
df_test.to_csv("test.csv", index=False, encoding='utf8')


# In[183]:


df_train.shape


# In[184]:


df_train['is_goal'].count()


import numpy as np
import pandas as pd
train = pd.read_csv('processed2.csv')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# In[241]:


train = train.drop(columns = 'Unnamed: 0')
#train = train.drop(columns = 'match_event_id')


# In[242]:


train['area_of_shot'].replace(np.nan,'NaN',inplace=True)
train['shot_basics'].replace(np.nan,'NaN',inplace=True)
train['range_of_shot'].replace(np.nan,'NaN',inplace=True)
train['type_of_shot'].replace(np.nan,'NaN',inplace=True)
train['type_of_combined_shot'].replace(np.nan,'NaN',inplace=True)
train['match_event_id'].replace(np.nan,-999,inplace=True)


# In[243]:


#game_season, area_of_shot, shot_basics, range_of_shot, date_of_game, type_of_shot, type_of_combined_shot
label_en_gs = LabelEncoder()
label_en_as = LabelEncoder()
label_en_bs = LabelEncoder()
label_en_rs = LabelEncoder()
label_en_dg = LabelEncoder()
label_en_ts = LabelEncoder()
label_en_cs = LabelEncoder()
label_en_meid = LabelEncoder()


# In[244]:


train['game_season'] = label_en_gs.fit_transform(train['game_season'])
train['date_of_game'] = label_en_dg.fit_transform(train['date_of_game'])
train['area_of_shot'] = label_en_as.fit_transform(train['area_of_shot'])
train['shot_basics'] = label_en_bs.fit_transform(train['shot_basics'])
train['range_of_shot'] = label_en_rs.fit_transform(train['range_of_shot'])
train['type_of_shot'] = label_en_ts.fit_transform(train['type_of_shot'])
train['type_of_combined_shot'] = label_en_cs.fit_transform(train['type_of_combined_shot'])
train['match_event_id'] = label_en_meid.fit_transform(train['match_event_id'])


# In[245]:


print(label_en_as.transform(['NaN']))
print(label_en_bs.transform(['NaN']))
print(label_en_rs.transform(['NaN']))
print(label_en_ts.transform(['NaN']))
print(label_en_cs.transform(['NaN']))
print(label_en_meid.transform([-999]))


# In[246]:


train['area_of_shot'].replace(4,'NaN',inplace=True)
train['shot_basics'].replace(5,'NaN',inplace=True)
train['range_of_shot'].replace(5,'NaN',inplace=True)
train['type_of_shot'].replace(0,'NaN',inplace=True)
train['type_of_combined_shot'].replace(0,'NaN',inplace=True)
train['match_event_id'].replace(0,'NaN',inplace=True)


# In[247]:


test = []
train1 = []

for i in range(0, train.shape[0]):
    if pd.isnull(train['is_goal'][i]):
        test.append(i)
    else :
        train1.append(i)


# In[248]:


df_test = train.loc[test, :]
df_train = train.loc[train1, :]


# In[249]:


y = df_train.loc[:, 'is_goal']
id_train = df_train.loc[:, 'shot_id_number']
id_test =  df_test.loc[:, 'shot_id_number']
x_train = df_train.drop(columns = ['is_goal', 'shot_id_number'])
x_test = df_test.drop(columns = ['is_goal', 'shot_id_number'])


# In[250]:


x_train.columns


# In[251]:


x_test.columns


# In[252]:


x2_train = x_train.iloc[:, 0:24].values
x2_test = x_test.iloc[:, 0:24].values


# In[253]:


print(x2_train.shape)
print(x2_test.shape)


# In[254]:


imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer.fit(x2_train[:, :])
x2_train[:, :] = imputer.transform(x2_train[:, :])
x2_test[:, :] = imputer.transform(x2_test[:, :])


# In[255]:


imputer1 = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
imputer1.fit(x2_train[:, :])
x2_train[:, :] = imputer1.transform(x2_train[:, :])
x2_test[:, :] = imputer1.transform(x2_test[:, :])


# In[256]:


print(x2_train.shape)
print(x2_test.shape)


# In[257]:


x2_train.dtype


# In[259]:


x_big = np.concatenate([x2_train, x2_test])
onehotencoder = OneHotEncoder(categorical_features = [0,9,10,11,12,13,14,15,16,17,18,19,20,21,22])
onehotencoder.fit(x_big)
X_big = onehotencoder.transform(x_big).toarray()


# In[260]:


X_train = X_big[0:24429, :]
X_test = X_big[24429: , :]


# In[261]:


print(X_train.shape)
print(X_test.shape)


# In[262]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)


# In[263]:


print(X_train.shape)
print(X_test.shape)


# In[264]:


xTrain, xval, yTrain, yval = train_test_split(X_train, y, test_size = 0.1, random_state = 0)


# In[265]:


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


