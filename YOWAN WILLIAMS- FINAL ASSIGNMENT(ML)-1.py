#!/usr/bin/env python
# coding: utf-8

# In[14]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_rows', 115)
pd.set_option('display.max_columns', 500)


# In[16]:


#
data = pd.read_csv("Housing_dataset.csv")


# In[17]:


data.head(10)


# In[18]:


data.info()


# In[19]:


data.describe()


# In[20]:


data[data.FIREPLACES == 293920]


# In[21]:


data.dtypes


# In[22]:


data.isnull().sum()


# In[23]:


data.nunique()


# In[24]:


data[(data.PRICE.isnull()) & (data.QUALIFIED == 'Q')]


# # DATA PREPROCESSING

# In[25]:


data.drop(['Unnamed: 0', "CMPLX_NUM", "LIVING_GBA" , "ASSESSMENT_SUBNBHD", "CENSUS_TRACT", 
         "CENSUS_BLOCK", "GIS_LAST_MOD_DTTM", "SALE_NUM", "USECODE", "CITY", 
         "STATE", "NATIONALGRID",'X','Y','SALEDATE'],axis=1,inplace=True)


# In[26]:


data


# In[27]:


total = data.isnull().sum().sort_values(ascending=False)
percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis=1, keys = ['Total','Percent'])
missing_data


# In[34]:


data[(data.PRICE.isnull()) & (data.KITCHENS.isnull())]['PRICE'].value_counts()


# ## FILLING YEAR REMODEL DATA

# In[46]:


data.dropna(subset=['AYB'],inplace=True)
group_remodel= data.groupby(['EYB','AYB']).mean()['YR_RMDL']
group_remodel.dtypes


# In[39]:


data['AYB'].isnull().sum()


# In[42]:


data['AYB'].dtypes
data['EYB'].dtypes


# In[43]:


# First convert this then groupmodel
data['YR_RMDL'] = data['YR_RMDL'].astype('Int64')


# In[44]:


data['YR_RMDL'].dtypes


# In[47]:


group_remodel.astype('Int64')


# In[49]:


#Converting float to integer
data['AYB'] = data['AYB'].astype(int)


# In[50]:


data.isnull().sum().sort_values(ascending = False)


# In[55]:


def applyRemodel(x):
    if pd.isnull(x['YR_RMDL']):
        return (group_remodel.loc[x['EYB']][x['AYB']])
    else:
        return x['YR_RMDL']


# In[56]:


data['YR_RMDL'] = data[['YR_RMDL','EYB','AYB']].apply(applyRemodel, axis = 1)


# In[57]:


data.dropna(subset=['YR_RMDL'],inplace=True)


# In[58]:


data


# In[59]:


data.drop(index=56600,inplace=True)


# In[60]:


data.reset_index().drop('index',axis=1,inplace=True)


# In[61]:


data.isnull().sum()


# # Dropping the following values- NaN values at PRICE, LONGITUDE, LATITUDE, ZIPCODE, WARD, ASSESSMENT_NBHD

# In[62]:


data.dropna(subset=['PRICE', 'LONGITUDE', 'ASSESSMENT_NBHD', 'ZIPCODE', 'WARD', 'LATITUDE'], inplace = True)


# In[63]:


data = data.reset_index().drop('index',axis=1)


# In[64]:


data


# In[65]:


## Drop all NaN values


# In[66]:


data.dropna(subset=['STRUCT','STYLE','FULLADDRESS','QUADRANT'],inplace=True)


# In[67]:


data['AYB'] = data['AYB'].values.astype(int)
data['YR_RMDL'] = data['YR_RMDL'].values.astype(int)
data['NUM_UNITS'] = data['NUM_UNITS'].values.astype(int)
data['KITCHENS'] = data['KITCHENS'].values.astype(int)
data['ZIPCODE'] = data['ZIPCODE'].values.astype(int)


# # Change column FULLADDRESS to SUBADRESS for grouping property location 

# In[70]:


temp = []
for item in data.FULLADDRESS.values:
    splt = item.split()[1:]
    sub_address = ' '.join(splt[:(len(splt)-1)])
    temp.append(sub_address)
data.FULLADDRESS = temp


# In[69]:


data['FULLADDRESS'].dtype


# In[71]:


data.rename(columns={'FULLADDRESS':'SUBADDRESS'},inplace=True)


# In[72]:


data.isnull().sum().sort_values(ascending = False)


# In[73]:


data['SUBADDRESS'].value_counts()


# In[74]:


data.nunique()


# In[75]:


data.dtypes


# In[76]:


data


# In[77]:


data["HEAT"].value_counts()


# In[79]:


#Checking the rooms
data['ROOMS'].value_counts().sort_values(ascending = False)


# In[80]:


data['CNDTN'].value_counts()


# In[81]:


data['SOURCE'].value_counts()


# In[82]:


data['AC'].value_counts()


# # Creating a function for AC 0 values¶

# In[83]:


# function for changing AC value from 0 to No.
def changeAc(x):
    for item in x.AC:
        if item !='0':
            return(item)
        else:
            return('N')


# In[85]:


data['AC'] = data.apply(changeAc, axis = 1)


# In[86]:


data['AC'].value_counts()


# In[87]:


#retriving every unique value in each of the column of the csv file
for col in list(data):
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    
    print(col)
    print(data[col].unique())


# In[89]:


# filling the missing values with mode in stories
data['STORIES'] = data['STORIES'].fillna(data['STORIES'].mode()[0])


# In[90]:


data.isnull().sum()


# In[91]:


data


# In[92]:


data.dtypes


# In[93]:


data.nunique()


# In[94]:


max_value = data["PRICE"].max
max_value


# In[95]:


# Dropping Square and SUBADDRESS
data.drop(['SQUARE','SUBADDRESS'],axis=1,inplace=True)


# In[96]:


# Converting categorical values into numerical
columnsToEncode=data.select_dtypes(include=[object]).columns
New_data = pd.get_dummies(data, columns=columnsToEncode, drop_first=True)


# In[97]:


New_data


# In[98]:


New_data.nunique()


# In[99]:


New_data.isnull().sum()


# # PART-B: DIMENSIONALITY REDUCTION

# In[101]:


New_data.keys()


# In[102]:


New_data.iloc[:,9]


# In[103]:


New_data.columns.get_loc("PRICE")


# In[104]:


#SVD
from sklearn.decomposition import TruncatedSVD


# In[105]:


svd = TruncatedSVD()
svd.fit(New_data)
svd_transformed_data = svd.transform(New_data)


# In[106]:


svd_transformed_data[0]


# In[107]:


#PCA
from sklearn.decomposition import PCA


# In[108]:


pca = PCA(n_components=2)
pca.fit(New_data)
pca_transformed_data = pca.transform(New_data)
pca.explained_variance_


# In[109]:


pca_transformed_data[0]


# # SCREE PLOT 

# In[111]:


#### Construct Convariance Matrix
cov_mat = np.cov(New_data.T)

#### Get Eigen Values and Vectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)


# In[112]:


SVD_values = np.arange(svd.n_components) + 1
plt.plot(SVD_values, svd.explained_variance_ratio_, 'o-', linewidth=2, color='red')
plt.title('Scree Plot')
plt.xlabel('Singular Value Decomposition')
plt.ylabel('Variance Explained')
plt.show()


# In[113]:


PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()


# In[116]:


pca = PCA().fit(New_data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# # PART C

# In[117]:


#A:
X = New_data.drop(columns = 'PRICE').values
y = New_data['PRICE'].values


# In[118]:


from sklearn.model_selection import train_test_split


# In[119]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[120]:


from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept = True)


# In[121]:


model.fit(X_train, y_train)


# In[122]:


#testibg with test data
y_pred = model.predict(X_test)


# In[123]:


np.column_stack((y_test, y_pred.astype(int)))


# In[124]:


from sklearn.metrics import r2_score


# In[125]:


r2_score(y_test, y_pred)


# # Converting price into categorical values

# In[126]:


value_list = []
for value in New_data['PRICE']:
  if value < 30000:
    value_list.append(0)
  elif 30000 <= value <= 60000:
    value_list.append(1)
  elif 60000 <= value <= 90000:
    value_list.append(2)
  elif 90000 <= value <= 130000:
    value_list.append(3)
  elif 130000 < value:
    value_list.append(4)

New_data['PRICE'] = value_list


# In[127]:


New_data


# In[128]:


y = New_data['PRICE'].values


# # PART C- 3
# 

# In[131]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[132]:


logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)


# In[133]:


# Logistic Regression accuracy
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)
print(score)


# # B. NEURAL NETWORK
# 

# In[134]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy


# In[135]:


model = Sequential([
    Dense(units = 186, input_shape = (186,), activation = 'relu'),
    Dense(units = 5, activation = 'softmax')
])


# In[136]:


model.summary()


# In[137]:


model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[140]:


model.fit(x = X_train, y = y_train, validation_split = 0.1, batch_size = 15, epochs = 15)


# In[141]:


# Neural Network test validation and accuracy
model.evaluate(X_test, y_test)


# # NAIVE BAYES

# In[142]:


from sklearn.naive_bayes import GaussianNB


# In[143]:


nb = GaussianNB()


# In[144]:


nb.fit(X_train, y_train)


# In[145]:


# Naïve Bayes accuracy
y_pred = nb.predict(X_test)
accuracy_score(y_test, y_pred)


# # END
