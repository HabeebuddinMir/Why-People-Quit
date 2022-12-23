#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 
# * [Approaching Categorical Features](#1)
# * [Various Approaches to Handle Missing values in Categorical Features](#2)
# * [k-Nearest Neighbour Imputation](#3)
# * [Evaluation Metrics](#4)
# * [Model](#5)
# * [Learning Curve](#6)
# * [Oversampling using SMOTE](#7)
# * [Hyperparameter Tunning](#8)
# * [Reference](#9)

# ## Import important libraries and packages

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, log_loss, plot_roc_curve, auc, precision_recall_curve
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
from xgboost import XGBClassifier
from functools import partial
from skopt import gp_minimize
from skopt import space
from skopt.plots import plot_convergence

sns.set_style('whitegrid')



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df_train = pd.read_csv('../input/hr-analytics-job-change-of-data-scientists/aug_train.csv')
df_test = pd.read_csv('../input/hr-analytics-job-change-of-data-scientists/aug_test.csv')


# In[3]:


df_train.head()


# In[4]:


df_test.head()


# Features
# 
# * enrollee_id : Unique ID for candidate
# * city: City code
# * city_ development _index : Developement index of the city (scaled)
# * gender: Gender of candidate
# * relevent_experience: Relevant experience of candidate
# * enrolled_university: Type of University course enrolled if any
# * education_level: Education level of candidate
# * major_discipline :Education major discipline of candidate
# * experience: Candidate total experience in years
# * company_size: No of employees in current employer's company
# * company_type : Type of current employer
# * lastnewjob: Difference in years between previous job and current job
# * training_hours: training hours completed
# * target: 0 – Not looking for job change, 1 – Looking for a job change

# In[5]:


df_train.info()


# In[6]:


df_test.info()


# # Approaching Categorical Features<a id = "1" ></a>
# 
# 
# Categorical variables/features are any feature type can be classified into two major
# types:
# *  Nominal
# *  Ordinal
# 
# **Nominal variables** are variables that have two or more categories which do not
# have any kind of order associated with them. For example, if gender is classified
# into two groups, i.e. male and female, it can be considered as a nominal variable.
# 
# **Ordinal variables** on the other hand, have “levels” or categories with a particular
# order associated with them. For example, an ordinal categorical variable can be a
# feature with three different levels: low, medium and high. Order is important.

# **List of ordinal variables in this data**
# 
# 1. education_level
# 2. company_size
# 3. experience
# 4. last_new_job
# 5. company_type

# In[7]:


df_train.head()


# ## Various Approaches to Encode Categorical Features
# 

# In[8]:


#lets combine train and test sets to preprocess the data

#First i suggest to create a fake target feature in test set with some same value for every single element
#By this it will be easy for us to combine and seprate our training and test data after data preprocessing

#can plot count plot for more intution

df_test['target'] = -1 #remeber that we have to drop this column later

df_pre = pd.concat([df_train, df_test], axis = 0).reset_index(drop = True)
# Just a Tip always reset the indices whenever you join or disjoin two or more datasets


# In[9]:


df_pre.info()


# 
# **Label Encoding** refers to converting the labels into numeric form so as to convert it into the machine-readable form. Machine learning algorithms can then decide in a better way on how those labels must be operated. It is an important pre-processing step for the structured dataset in supervised learning.
# ![](https://ekababisong.org/assets/seminar_IEEE/LabelEncoder.png)  
# 
# We can do label Encoding From LabelEncoder of scikit-Learn but to do so first we have to impute missing values in data 

# In[10]:


from sklearn.preprocessing import LabelEncoder

#Making Copy of data just for example
df_lb = df_pre.copy()
df_lb['education_level'].value_counts()


# The feature column education_level of data is in categorical form as we can see above output

# In[11]:


#Fill nan values
df_lb.loc[:, "education_level"] = df_lb['education_level'].fillna("NONE")


# In above code cell i just create null values as new category "NONE"

# In[12]:


# initialize LabelEncoder
lbl_enc = LabelEncoder()

# fit label encoder and transform values on ord_2 column
df_lb.loc[:, "education_level"] = lbl_enc.fit_transform(df_lb['education_level'].values)

df_lb['education_level'].value_counts()


# The feature column education_level of data is now transformed into numarical form as we can see above
# 
# But in this Notebook i am not going to use scikit-Learn LabelEncoder Due to following reasons
# 
# 1. Label Encoder encode data on basis of count but as mentioned above this data have lots of ordinal features means categories of some features might depend      upon some levels like in education_level feature\ 
# 
#    We know that we should encode data in this order but label encoder encodes it on basis of count
#    
#       Primary School  
#       High School            
#       Graduate          
#       Masters           
#       Phd
#     
# 2. To use label encoder first we have to create NULL values as new category and Our data have so many NULL values so we can not just create new Category for      NULL values because due to this data distribution could change
# 
# 

# In[13]:


# Making Dictionaries of ordinal features

gender_map = {
        'Female': 2,
        'Male': 1,
        'Other': 0
         }

relevent_experience_map = {
    'Has relevent experience':  1,
    'No relevent experience':    0
}

enrolled_university_map = {
    'no_enrollment'   :  0,
    'Full time course':    1, 
    'Part time course':    2 
}
    
education_level_map = {
    'Primary School' :    0,
    'Graduate'       :    2,
    'Masters'        :    3, 
    'High School'    :    1, 
    'Phd'            :    4
    } 
    
major_map ={ 
    'STEM'                   :    0,
    'Business Degree'        :    1, 
    'Arts'                   :    2, 
    'Humanities'             :    3, 
    'No Major'               :    4, 
    'Other'                  :    5 
}
    
experience_map = {
    '<1'      :    0,
    '1'       :    1, 
    '2'       :    2, 
    '3'       :    3, 
    '4'       :    4, 
    '5'       :    5,
    '6'       :    6,
    '7'       :    7,
    '8'       :    8, 
    '9'       :    9, 
    '10'      :    10, 
    '11'      :    11,
    '12'      :    12,
    '13'      :    13, 
    '14'      :    14, 
    '15'      :    15, 
    '16'      :    16,
    '17'      :    17,
    '18'      :    18,
    '19'      :    19, 
    '20'      :    20, 
    '>20'     :    21
} 
    
company_type_map = {
    'Pvt Ltd'               :    0,
    'Funded Startup'        :    1, 
    'Early Stage Startup'   :    2, 
    'Other'                 :    3, 
    'Public Sector'         :    4, 
    'NGO'                   :    5
}

company_size_map = {
    '<10'          :    0,
    '10/49'        :    1, 
    '100-500'      :    2, 
    '1000-4999'    :    3, 
    '10000+'       :    4, 
    '50-99'        :    5, 
    '500-999'      :    6, 
    '5000-9999'    :    7
}
    
last_new_job_map = {
    'never'        :    0,
    '1'            :    1, 
    '2'            :    2, 
    '3'            :    3, 
    '4'            :    4, 
    '>4'           :    5
}


# I am using mapping to transform categorical features into numarical features

# In[14]:


# Transforming Categorical features into numarical features

df_pre.loc[:,'education_level'] = df_pre['education_level'].map(education_level_map)
df_pre.loc[:,'company_size'] = df_pre['company_size'].map(company_size_map)
df_pre.loc[:,'company_type'] = df_pre['company_type'].map(company_type_map)
df_pre.loc[:,'last_new_job'] = df_pre['last_new_job'].map(last_new_job_map)
df_pre.loc[:,'major_discipline'] = df_pre['major_discipline'].map(major_map)
df_pre.loc[:,'enrolled_university'] = df_pre['enrolled_university'].map(enrolled_university_map)
df_pre.loc[:,'relevent_experience'] = df_pre['relevent_experience'].map(relevent_experience_map)
df_pre.loc[:,'gender'] = df_pre['gender'].map(gender_map)
df_pre.loc[:,'experience'] = df_pre['experience'].map(experience_map)

#encoding city feature using label encoder
lb_en = LabelEncoder()

df_pre.loc[:,'city'] = lb_en.fit_transform(df_pre.loc[:,'city']) 



# In[15]:


df_pre.head()


# In[16]:


df_pre.info()


# We can use this directly in many tree-based models like:
# *  Decision trees
# *  Random forest
# *  Extra Trees
# *  Or any kind of boosted trees model
# 
#    * XGBoost
#    * GBM
#    * LightGBM
#    
# Generally, in tree-based models the scale of the features does not matter. This is because at each tree level, the score of a possible split will be equal whether the respective feature has been scaled or not.
# 
# You can think of it like here: We're dealing with a binary classification problem and the feature we're splitting takes values from 0 to 1000. If you split it on 300, the samples <300 belong 90% to one category while those >300 belong 30% to one category. Now imaging this feature is scaled between 0 and 1. Again, if you split on 0.3, the sample <0.3 belong 90% to one category while those >0.3 belong 30% to one category.
# 
# So you've changed the splitting point but the actual distribution of the samples remains the same regarding the target variable.
# 
# Above example is taken from : [Why Normalization is not required for tree based models](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/160613)
# 
# 
# This type of encoding cannot be used in linear models, support vector machines or neural networks as they expect data to be normalized (or standardized). For these types of models, we can binarize the data.
# 
# 
# 

# #### Approch For Categorical Features(Summary)
# 
# 1. Fill nan values with some value like NONE to which we can deal later 
# 2. Then convert categorical features in numarical using label encoder
# 
# **NOTE : If you want to use some tree based algorithm than these much steps are sufficent**
# 
# 3. In case of non tree based model do One Hot Encoding of numarical feature acieved from step 2 and make sparse = True in      One Hot Encoding
# 4. Now select model and train your data

# # Various Approaches to Handle Missing values in Categorical Features <a id = "2" ></a>
# 
# 1. You can simply drop columns having very large amount of null values
# 2. Drop entire row if it has some null values (But this approach is **not Recommendable** because then we will lose lots of precious data)
# 3. First convert none null categorical data into numarical form and then simply fill mean, mode or median value inplace of null values
# 4. First convert none null categorical data into numarical form and then with the help of k-Nearest Neighbors algorithm find missing values and impute them in    data
# 5. Another way of imputing missing values in a column would be to train a regression model that tries to predict missing values in a column based on other        columns.
# 
# **In This notebook i am using k-Nearest Neighbors algorithm to fill missing values**
# 

# Below i am plotting count of values per columns of dataset. I also have sorted columns based on missing values.

# In[17]:


missingno.bar(df_pre,color="dodgerblue", sort="ascending", figsize=(10,5), fontsize=12);


# In[18]:


#Just to check number of null values of every column in data

for col in df_pre.columns:
    null_val = df_pre[col].isnull().sum()
    null_prec = (null_val * 100) / df_pre.shape[0]
    print('> %s , Missing: %d (%.1f%%)' % (col, null_val, null_prec))


# Below we are plotting heatmap showing nullity correlation between various columns of dataset.
# 
# The nullity correlation ranges from -1 to 1.
# 
# * -1 - Exact Negative correlation represents that if the value of one variable is present then the value of other variables is definitely absent.
# * 0 - No correlation represents that variables values present or absent do not have any effect on one another.
# * 1 - Exact Positive correlation represents that if the value of one variable is present then the value of the other is definitely present.

# In[19]:


missingno.heatmap(df_pre, cmap="RdYlGn", figsize=(10,5), fontsize=12);


# # <center> k-Nearest Neighbour Imputation <a id = "3" ></a> </center>
# 
# 
# A fancy way of filling in the missing values would be to use a **k-nearest neighbour** method. You can select a sample with missing values and find the nearest
# neighbours utilising some kind of distance metric, for example, Euclidean distance. Then you can take the mean of all nearest neighbours and fill up the missing value. You can use the KNN imputer implementation for filling missing values like this.
# 
# ![image1](https://static.javatpoint.com/tutorial/machine-learning/images/k-nearest-neighbor-algorithm-for-machine-learning2.png)
# 
# 
# [K-Nearest Neighbors (KNN) Algorithm for Machine Learning](https://medium.com/capital-one-tech/k-nearest-neighbors-knn-algorithm-for-machine-learning-e883219c8f26)
# 
# 
# ### How it works?
# 
# Step-1: Select the K number of the neighbors
#         
# let say we select K = 5
#         
# ![image2](https://static.javatpoint.com/tutorial/machine-learning/images/k-nearest-neighbor-algorithm-for-machine-learning3.png) 
# 
# 
# Step-2: Calculate the Euclidean distance of K number of neighbors
# 
# In this step we search for those k = 5 neighbors having minimum Euclidean Distance from unknown data point
# 
# 
# ![Image4](https://static.javatpoint.com/tutorial/machine-learning/images/k-nearest-neighbor-algorithm-for-machine-learning4.png)
# 
#         
# Step-3: Among these k neighbors, count the number of the data points in each category.
# 
# ![image3](https://static.javatpoint.com/tutorial/machine-learning/images/k-nearest-neighbor-algorithm-for-machine-learning5.png)
# 
# 
# Step-4: Assign the new data points to that category for which the number of the neighbor is maximum.
# 
# You can also visit below given youtube video link to understand it bit nicely
# 
# [Step-by-Step procedure of KNN Imputer for imputing missing values](https://www.youtube.com/watch?v=AHBHMQyD75U)

# In[20]:


df_pre1 = df_pre.copy()


# In[21]:


knn_imputer = KNNImputer(n_neighbors = 3)

X = np.round(knn_imputer.fit_transform(df_pre1))
df_pre1 = pd.DataFrame(X, columns = df_pre1.columns)


# In[22]:


df_pre1.info()


# In[23]:


df_pre1.head()


# But wait as you can observe city_development_index feature values were in continues form before imputation and after imputation now they are in discrete form.
# As you may also noticed i was rounding values after imputation because of this city_development_index values converted into discrete form
# 
# I am plotting kernel density estimate plot just to check distribution of city_development_index feature before and after imputation. You can plot KDE plots for every feature to check effect of imputaion on data

# In[24]:


plt.figure(figsize = (15,8))
plt.title('Before Imputation')
df_pre['city_development_index'].plot(kind = 'kde')
plt.show()


# In[25]:


plt.figure(figsize = (15,8))
plt.title('After Imputation')
df_pre1['city_development_index'].plot(kind = 'kde')
plt.show()


# So rather than using imputation on whole dataset just use it on those features having missing values.

# In[26]:


# missing columns

missing_cols = df_pre.columns[df_pre.isna().any()].tolist()
missing_cols


# Above shown columns have missing values and all 8 are categorical features
# 
# Now i would like make two different dataframes one having features with missing values and second having features without missing values. But there will be one common column enrollee_id so that later we can perform inner join on both dataframes

# In[27]:


#dataframe having features with missing values
df_missing = df_pre[['enrollee_id'] + missing_cols]

#dataframe having features without missing values
df_non_missing = df_pre.drop(missing_cols, axis = 1)


# In[28]:


#k-Nearest Neighbour Imputation

knn_imputer = KNNImputer(n_neighbors = 3)

X = np.round(knn_imputer.fit_transform(df_missing))
#Rounding them because these are categorical features

df_missing = pd.DataFrame(X, columns = df_missing.columns)


# In[29]:


#now lets join both dataframes 

df_pre2 = pd.merge(df_missing, df_non_missing, on = 'enrollee_id')


# If you remember i did concatenation between train and test data before preprocessing. Now after preprocessing of data we can seprate train and test data

# In[30]:


train = df_pre2[df_pre2['target'] != -1].reset_index(drop = True)
test = df_pre2[df_pre2['target'] == -1].reset_index(drop = True)

X = train.drop(['enrollee_id', 'target'], axis = 1)
Y = train['target']

# drop fake target feature from test data 
test = test.drop('target', axis = 1)


# # Evaluation Metrics <a id = "4" ></a>
# 
# I think before selecting an optimal model for given data first we have to analayze target feature. Target Feature can be discrete in case of classification problem or continuous in case of Regression Problem
# 
# If we talk briefly about classification problems, the most common metrics used are:
# 
# 
# - **[Accuracy](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)** : It is one of the most straightforward metrics used in machine learning. It defines how accurate your model is. For example, if you build a model that classifies 90 samples accurately, your accuracy is 90% or 0.90. If only 83 samples are classified correctly, the accuracy of your model is 83% or 0.83. Simple.
#              
#      **[Scikit-learn user guide for accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)**
#      
# 
# - **[Precision](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)** :  Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate. We have got 0.788 precision which is pretty good
# 
#      **[True Positives (TP)](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)** - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted  class is also yes. E.g. if actual class value indicates that this passenger survived and predicted class tells you the same thing.
# 
#      **[True Negatives (TN)](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)** - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no. E.g. if actual class says this passenger did not survive and predicted class tells you the same thing.
# 
#      False positives and false negatives, these values occur when your actual class contradicts with the predicted class.
# 
#      **[False Positives (FP)](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)** – When actual class is no and predicted class is yes. E.g. if actual class says this passenger did not survive but predicted class tells you that this passenger will survive.
# 
#      **[False Negatives (FN)](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)** – When actual class is yes but predicted class in no. E.g. if actual class value indicates that this passenger survived and predicted class tells you that passenger will die.
# 
#      **[Scikit-learn user guide for Precision ](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)**
#      
#      ![](https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg)     
#      
# 
# - **[Recall(Sensitivity)](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)** : Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. The question recall answers is: Of all the passengers that truly survived, how many did we label? We have got recall of 0.631 which is good for this model as it’s above 0.5.
# 
#      **[Scikit-learn user guide for Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)**
# 
# 
# - **[Confusion Matrix](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/)** : A much better way to evaluate the performance of a classifier is to look at the confusion matrix. The general idea is to count the number of times instances of class A are classified as class B. For example, to know the number of times the classifier confused images of 5s with 3s, you would look in the 5th row and 3rd column of the confusion matrix.
# 
#     **[Scikit-Learn user guide for Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)**
#      
#      
# - **[F1 score (F1)](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)** : F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. In our case, F1 score is 0.701. 
# 
#      **[Scikit-learn user guide for F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)**
# 
# 
# - **[Area under the ROC (Receiver Operating Characteristic) curve](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)** : AUC - ROC curve is a performance measurement for the classification problems at various threshold settings. ROC is a probability curve and AUC represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. Higher the AUC, the better the model is at predicting 0s as 0s and 1s as 1s. By analogy, the Higher the AUC, the better the model is at distinguishing between patients with the disease and no disease.
# 
#      **[Scikit-learn user guide for AUC under the ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)**
# 
# 
# 
# 
# 
# 
# **When it comes to regression, the most commonly used [evaluation metrics](https://towardsdatascience.com/evaluation-metrics-model-selection-in-linear-regression-73c7573208be) are:**
# 
# 
# - Mean absolute error (MAE)
# - Mean squared error (MSE)
# - Root mean squared error (RMSE)
# - Root mean squared logarithmic error (RMSLE)
# - Mean percentage error (MPE)
# - Mean absolute percentage error (MAPE)
# - R2
# 
#  **[Scikit-learn user guide for regression evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)**

# In[31]:


sns.countplot(train['target'], edgecolor = 'black')


# We see that the target is **skewed** and thus the best metric for this binary classification problem would be Area Under the ROC Curve (AUC). We can use precision and recall too, but AUC combines these two metrics. Thus, we will be using AUC to evaluate the model that we build on this dataset.

# # Model <a id = "5"></a>
# 
# In this notebook i would like to use [Extreme Gradient Boosting (XGBoost)](https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/) Classifier

# In[32]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2 ,random_state = 42)

clf = XGBClassifier()

clf.fit(X_train, y_train)

y_train_pred = clf.predict_proba(X_train)
y_train_pred_pos = y_train_pred[:,1]

y_val_pred = clf.predict_proba(X_val)
y_val_pred_pos = y_val_pred[:,1]

auc_train = roc_auc_score(y_train, y_train_pred_pos)
auc_test = roc_auc_score(y_val, y_val_pred_pos)

print(f"Train AUC Score {auc_train}")
print(f"Test AUC Score {auc_test}")

fpr, tpr, _ = roc_curve(y_val, y_val_pred_pos)


# As we can see model is overfitting the data, we can do various things to resolve this problem like we can increase data set size in balanced manner and we can also tune hyperparameters of model 

# Let's plot AUC Curve

# In[33]:


def plot_auc_curve(fpr, tpr, auc):
    plt.figure(figsize = (16,6))
    plt.plot(fpr,tpr,'b+',linestyle = '-')
    plt.fill_between(fpr, tpr, alpha = 0.5)
    plt.ylabel('True Postive Rate')
    plt.xlabel('False Postive Rate')
    plt.title(f'ROC Curve Having AUC = {auc}')


# In[34]:


plot_auc_curve(fpr, tpr, auc_test)


# # Learning Curve <a id = "6"></a>
# 
# 
# Training 3 examples will easily have 0 errors because we can always find any curve that exactly touches 3 points.
# 
# * As the training set gets larger, the error for a function increases. 
# * The error value will plateau out after a certain m, or training set size.
# 
# 
# **With high bias**
# 
# 
# * Low training set size: causes training cost to be low and cross validation cost to be high
# * Large training set size: causes both training cost and cross validation cost to be high with training cost = cross validation cost
# 
# 
# ![img1](https://www.dataquest.io/wp-content/uploads/2019/01/low_high_var.png)
# 
# **If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.**
# 
# 
# For high variance, we have the following relationships in terms of the training set size:
# 
# 
# **With high variance**
# 
# 
# * Low training set size: training cost will be low and cross validation cost will be high
# * Large training set size: training cost increases with training set size and cross validation cost decreases without leveling off. Also, training cost < cross   validation cost but the difference between them remains significant.
# 
# **If a learning algorithm is suffering from high variance, getting more training data is likely to help.**
# 
# You can visit below given link for more detailed intuition
# 
# [Learning Curves for machine learning](https://www.dataquest.io/blog/learning-curves-machine-learning/)

# In[35]:


# funtion to plot learning curves

def plot_learning_cuve(model, X, Y):
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 11)
    train_loss, test_loss = [], []
    
    for m in range(200,len(x_train),200):
        
        model.fit(x_train.iloc[:m,:], y_train[:m])
        y_train_prob_pred = model.predict_proba(x_train.iloc[:m,:])
        train_loss.append(log_loss(y_train[:m], y_train_prob_pred))
        
        y_test_prob_pred = model.predict_proba(x_test)
        test_loss.append(log_loss(y_test, y_test_prob_pred))
        
    plt.figure(figsize = (15,8))
    plt.plot(train_loss, 'r-+', label = 'Training Loss')
    plt.plot(test_loss, 'b-', label = 'Test Loss')
    plt.xlabel('Number Of Batches')
    plt.ylabel('Log-Loss')
    plt.legend(loc = 'best')



    plt.show()
        


# In[36]:


plot_learning_cuve(XGBClassifier(), X, Y)


# It's a high variance problem 

# In[37]:


sns.countplot(Y, edgecolor = 'black')


# Let's try to increase data in balanced manner using Synthetic Minority Oversampling Technique (SMOTE) 

# # Oversampling using SMOTE <a id = "7"></a>
# 
# [SMOTE for Imbalanced Classification](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)\
# [SMOTE implementation](https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.SMOTE.html)

# In[38]:


from imblearn.over_sampling import SMOTE


# In[39]:


smote = SMOTE(random_state = 402)
X_smote, Y_smote = smote.fit_resample(X,Y)


sns.countplot(Y_smote, edgecolor = 'black')


# In[40]:


print(X_smote.shape)


# In[41]:


X_train, X_val, y_train, y_val = train_test_split(X_smote, Y_smote, test_size = 0.2 ,random_state = 42)

clf = XGBClassifier()

clf.fit(X_train, y_train)

y_train_pred = clf.predict_proba(X_train)
y_train_pred_pos = y_train_pred[:,1]

y_val_pred = clf.predict_proba(X_val)
y_val_pred_pos = y_val_pred[:,1]

auc_train = roc_auc_score(y_train, y_train_pred_pos)
auc_test = roc_auc_score(y_val, y_val_pred_pos)

print(f"Train AUC Score {auc_train}")
print(f"Test AUC Score {auc_test}")


# In[42]:


plot_learning_cuve(XGBClassifier(), X_smote, Y_smote)


# Let's try to increase more data to conquer overfitting using SMOTE

# In[43]:


smote = SMOTE(random_state = 446)
X_smote1, Y_smote1 = smote.fit_resample(X,Y)


X_final = pd.concat([X_smote, X_smote1], axis = 0).reset_index(drop = True)
Y_final = pd.concat([Y_smote, Y_smote1], axis = 0).reset_index(drop = True)

sns.countplot(Y_final, edgecolor = 'black')


# In[44]:


print(X_final.shape)


# In[45]:


X_train, X_val, y_train, y_val = train_test_split(X_final, Y_final, test_size = 0.2 ,random_state = 42)

clf = XGBClassifier()


clf.fit(X_train, y_train)

y_train_pred = clf.predict_proba(X_train)
y_train_pred_pos = y_train_pred[:,1]

y_val_pred = clf.predict_proba(X_val)
y_val_pred_pos = y_val_pred[:,1]

auc_train = roc_auc_score(y_train, y_train_pred_pos)
auc_test = roc_auc_score(y_val, y_val_pred_pos)

print(f"Train AUC Score {auc_train}")
print(f"Test AUC Score {auc_test}")


# In[46]:


plot_learning_cuve(XGBClassifier(), X_final, Y_final)


# Now as we can see gap decreases that mean we are going good but this was only for illustration in further notebook i will use smote only once because using it twice may change distribution of data too much

# # Hyperparameter Tunning <a id = "8"></a>
# 
# The parameters that the model has here are known as hyper-parameters, i.e. the parameters that control the training/fitting process of the model.
# 
# 
# In this notebook i am using **Bayesian optimization with gaussian process**
# 

# Bayesian optimization algorithm need a function they can optimize. Most of the time, it’s about the minimization of this function, like we minimize loss.

# In[47]:


def optimize(params, param_names, x, y):
   

    # convert params to dictionary
    params = dict(zip(param_names, params))

    # initialize model with current parameters
    clf = XGBClassifier(tree_method = 'hist', **params)
    
    # initialize stratified k fold
    kf = StratifiedKFold(n_splits = 5)
    
    i = 0
    
    # initialize auc scores list
    auc_scores = []
    
    #loop over all folds
    for index in kf.split(X = x, y = y):
        train_index, test_index = index[0], index[1]
        
        
        
        x_train = x.iloc[train_index,:]
        y_train = y[train_index]

        smote = SMOTE(random_state = 446)
        x_train, y_train = smote.fit_resample(x_train,y_train)
        
        x_test = x.iloc[test_index,:]
        y_test = y[test_index]
        
        #fit model
        clf.fit(x_train, y_train)
        
        y_pred = clf.predict_proba(x_test)
        y_pred_pos = y_pred[:,1]
        
        auc = roc_auc_score(y_test, y_pred_pos)
        print(f'Current parameters of fold number {i} -> {params}')
        print(f'AUC score of test {i} f {auc}')

        i = i+1
        auc_scores.append(auc)
        
    return -1 * np.mean(auc_scores)
    
    


# So, let’s say, you want to find the best parameters for best accuracy and obviously, the more the accuracy is better. Now we cannot minimize the accuracy, but we can minimize it when we multiply it by -1. This way, we are minimizing the negative of accuracy, but in fact, we are maximizing accuracy. Using Bayesian optimization with gaussian process can be accomplished by using [gp_minimize function from scikit-optimize (skopt) library](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html). Let’s take a look at how we can tune the parameters of our xgboost model using this
# function.
# 
# [Parameters for XGBoost Classifier](https://xgboost.readthedocs.io/en/latest/parameter.html)
# 
# I am try to optimize the model with 4 hyperparameters but you can try many more given in above mentioned link

# In[48]:


#define a parameter space

param_spaces = [space.Integer(100, 2000, name = 'n_estimators'),
                space.Real(0.01,100, name = 'min_child_weight'),
                space.Real(0.01,1000, name = 'gamma'),
                space.Real(0.1, 1, prior = 'uniform', name = 'colsample_bytree'),
]

# make a list of param names this has to be same order as the search space inside the main function
param_names = ['n_estimators' ,'min_child_weight', 'gamma', 'colsample_bytree']

# by using functools partial, i am creating a new function which has same parameters as the optimize function except 
# for the fact that only one param, i.e. the "params" parameter is required. 
# This is how gp_minimize expects the optimization function to be. 
# You can get rid of this by reading data inside the optimize function or by defining the optimize function here.

optimize_function = partial(optimize, param_names = param_names, x = X, y = Y)


# In[49]:


# output of this cell is very large that's why it is hidden

result = gp_minimize(optimize_function, dimensions = param_spaces, n_calls = 20, n_random_starts = 5, verbose = 10)


# In[50]:


best_params = dict(zip(param_names, result.x))
print(f'Best Parameters : {best_params}')
print(f'Best AUC score : {result.fun}')


# let's again plot learning cuve with hyperparameters this time

# In[51]:


# splitting train and validation data

X_train, X_val, y_train, y_val = train_test_split(X,Y, test_size = 0.2, random_state = 24)

smote = SMOTE(random_state = 446)
X_train, y_train = smote.fit_resample(X_train,y_train)


# In[52]:


# initialize model with best parameters
clf = XGBClassifier(**best_params)

# fit model
clf.fit(X_train, y_train)

# predicting probabilities of training data
y_train_pred = clf.predict_proba(X_train)


y_train_pred_pos = y_train_pred[:,1]

y_val_pred = clf.predict_proba(X_val)
y_val_pred_pos = y_val_pred[:,1]

auc_train = roc_auc_score(y_train, y_train_pred_pos)
auc_test = roc_auc_score(y_val, y_val_pred_pos)

print(f"Train AUC Score {auc_train}")
print(f"Test AUC Score {auc_test}")

fpr, tpr, _ = roc_curve(y_val, y_val_pred_pos)


# In[53]:


plot_learning_cuve(XGBClassifier(**best_params),X_smote,Y_smote)


# In[54]:


plot_auc_curve(fpr, tpr, auc_test)

