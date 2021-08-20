#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import panda library for explore and manipulating
#Pandas has powerful methods for most things you want to do with data

import pandas as pd


# In[5]:


#Read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv('melb_data.csv')
#print a summary of the data in Melbourne data
melbourne_data.describe()


# In[25]:


#Visually checking data with these commands is an important part as you will frequently find surprises in the datasets that deserve further inspection
melbourne_data.head()


# In[26]:


#Visually checking data with these commands is an important part as you will frequently find surprises in the datasets that deserve further inspection
melbourne_data.tail()


# In[6]:


#To choose variables/columns, lets check all lists in a column
melbourne_data.columns


# In[9]:


#The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
#I will use dropna which is basically usded to drop missing values( think of naas not available)

melbourne_data = melbourne_data.dropna(axis=0)


# In[10]:


#I will select a subset from this data using two approaches for now
#Dot notation (for prediction target) and selecting with a column list, which we use to select the "features"
#By convention,i will have the target as y


# In[13]:


#Dot notation(selecting the prediction target)

y = melbourne_data.Price


# In[14]:


#Selecting multiple features by providing a list of column names inside brakets.

melbourne_features = ['Rooms', 'Bathroom', 'Landsize','Lattitude','Longtitude']


# In[17]:


#By convection, this data i will call it X

X = melbourne_data[melbourne_features]


# In[18]:


#I will explore the data I will be using in this project to predict house prices using describe method and head method, which shows top few rows

X.describe()


# In[19]:


#Visually checking data with these commands is an important part as you will frequently find surprises in the datasets that deserve further inspection
X.head()


# In[20]:


#Visually checking data with these commands is an important part as you will frequently find surprises in the datasets that deserve further inspection
X.tail()


# # Building Model Archicture

# In[ ]:


#I will use scikit-learn (sklearn) to create models
#Scikit-learn is apopular libraryfor modeling the types of data stored in DataFrame
#Here i will 
#1 Define the model
#2 Fit,to capture patterns from provided data. Heart of modeling
#3 Predict 
#4 Evaluate to determine how accurate the model's predictions are




# In[27]:


#Below is ane example of defining a DECISION TREE MODEL with Scikit-learn and fitting it with features and target variable


from sklearn.tree import DecisionTreeRegressor

#Define model. Specify a number for random_state to ensure same results each run
#When i specify a number for random_state esnures i get same results in each run
melbourne_model = DecisionTreeRegressor(random_state=1)

#Fit model

melbourne_model.fit(X, y)


# In[53]:


#I will amke predictions for the first few rows of the training data to see how the predict function works

print("Making predictions for the following 10 houses:")
print(X.head(10))
print("The prediction are")
print(melbourne_model.predict(X.head()))


# In[52]:


#I will amke predictions for the first few rows of the training data to see how the predict function works
print("Making prediction for bottom 10 houses")
print(X.tail(10))
print(melbourne_model.predict(X.tail()))


# # Model Validation

# In[ ]:


#The model is bult but how good is it?
#I will use model validation to measure the quality of the model
#Relevant measure of model quality is predictive accuracy
#In other words will the model's prediction be close to what actually happens

#Metric for summerizing model quality (1. Mean Absolute Error)

# Error = actual- predicted


# In[44]:


#Data loading code hidden here

import pandas as pd

#load data

melbourne_data =pd.read_csv("melb_data.csv")

#filter rows with missing price values

filtered_melbourne_data = melbourne_data.dropna(axis=0)

#Choose target and features

y = filtered_melbourne_data.Price
melbourne_features = ['Rooms','Bathroom','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
    
X = filtered_melbourne_data[melbourne_features]
    
    
from sklearn.tree import DecisionTreeRegressor
    
#Define model

melbourne_model = DecisionTreeRegressor()

#Fit model

melbourne_model.fit(X,y)


# In[51]:


#I will calculate the mean absolute error/In-Sample score
#In-sample might be limited and may not give accurate valuation

from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[ ]:


#The most straightforward way to do this is to exclude some data
#from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. 
#This data is called validation data.


# In[54]:


#i will use scikit-learn library that 
#has train_test_split function to break data into two pieces
#I willuse some of this data as training data to fit the model and calc mean_absolute_error

from sklearn.model_selection import train_test_split

#1 split data into training and validation data, 
# for both features and target
#2 The split is based on a random number generator Supplying a numeric value to
#3 The random_state argument guarentees we get the same split every time we 
#run this script

train_X, val_X, train_y, val_y =train_test_split(X,y,random_state =0)

#Fit model

melbourne_model.fit(train_X, train_y)

#get predicted prices on valiidation data

val_predictions = melbourne_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))


# In[ ]:


#CONCLUSION
#Your mean absolute error for the in-sample data was about 500 dollars. Out-of-sample it is more than 250,000 dollars.
#This is the difference between a model that is almost exactly right, and one that is unusable for most practical purposes
#As a point of reference, the average home value in the validation data is 1.1 million dollars. So the error in new data is about a quarter of the average home value.



# # EXPERIMENT WITH DIFFERNT MODELS

# In[ ]:


#What is overfitting a model matches the training data almost perfectly, but does poorly in validation and other new data. 
#Whats is underfitting? When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data


# In[67]:


# I will use utilty function to help compare MAE scores from diffrent values for max_leaf_nodes:

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model= DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[59]:


#Data loading code runs at this point

import pandas as pd

#load data

melbourne_data = pd.read_csv("melb_data.csv")

#filter rows with missing values

filtered_melbourne_data = melbourne_data.dropna(axis=0)

#choose target and features

y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom','Landsize', 'BuildingArea','YearBuilt','Lattitude','Longtitude']

X = filtered_melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

#split data into training and validation data for both features and target

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[68]:


# I will use for loop to comapre the accuracy of models built with different values for max_leaf_nodes

#compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))


# In[ ]:


#Of the options listed, 500 is the optimal number of leaves.

#CONCLUSION

# 1. Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
# 2. Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.
# 3. validation data,lets us try many candidate models and keep the best one.


# # Random Forests

# In[72]:


# The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree
import pandas as pd

#Load data

melbourne_data = pd.read_csv("melb_data.csv")

#Filter rows with missing values

melbourne_data = melbourne_data.dropna(axis=0)

#Choose target and features
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
X = melbourne_data[melbourne_features] 

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    


# In[74]:


#this time using the RandomForestRegressor class instead of DecisionTreeRegressor.

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# In[ ]:


#Conclusion

#There is likely room for further improvement, but this is a big improvement over the best decision tree error of 250,000. 
#Best features of Random Forest models is that they generally work reasonably even without this tuning.


# In[ ]:




