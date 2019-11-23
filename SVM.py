#!/usr/bin/env python
# coding: utf-8

# <h3 align="center">TEXT CLASSIFICATION USING SVM</h3> 

# <img src="svm2.PNG" width="400" height="250">

# # What is text classification?
# Given a truck-load of textual data, it is a huge task to analyse what is inside it owing to its lack of structure. 
# If machines can help us to automate this mechanical process of grouping the text, it is indeed substantial! 
# 
# Text classification is a way of identifying the category in which the contents of a text belong.
# Depending on the scenario, this classification can be binary (positive/negative, spam/non-spam) or 
# categorical (politics, technology business, fashion, sports etc). 
# 
# Classification of textual data is a means to clean it, organize it, make it user-friendly and to make sense out of the unstructured data.
# It marks its applications in the field of spam detection, sentiment analysis, tagging, language detection and a lot more.

# # How is it done?
# There are numerous machine learning algorithms like Naive Bayes, Random Forest, Support Vector Machines, 
# Neural Models which make use of the training data to arrive at the conclusive category for the new, 
# previously unseen text data.
# 
# Steps:
# 1. Data download.
# 2. Data pre-processing: xml-parsing, lower-case conversion, punctuation and special character removals, stopwords removal, lemmatization etc, all depending upon the requirement!
# 3. Word-to-vector Conversion: Machine do not understand text, so we vectorize every word into numerics and feed the vectors thus formed in the machine learning model.
# 4. Implementing the algorithm.
# 
# 
# Here in this article, we will discuss one of these algorithms, Support Vector Machines from scratch.
# 
# 

# <img src="svm.PNG" width="800" height="400">

# # What is Support Vector Machine (SVM) ?
# Support Vector Machines or SVM as we call them, are based on the concept of ‘lines’ or ‘hyperplanes’ dividing the space of
# vectorized data into two or multiple subspaces. 
# 
# In layman terms, SVM first analyses the given points in the n-dimensional space, then figures out the line/hyperplane that separates the points. Then, points belonging to one category falls onto one side of the line/hyperplane and points of another category falls exactly into its opposite side. 
# SVM model, once trained over the training data, puts the test point in the vector space and analyses its respective position with the line/hyperplane and hence decides its category!
# 
# A linear classifier has the form:
# 
# 

# <img src="linear.PNG" width="400" height="250">

# In 3D it is plane, while in nD we call it a hyperplane.
# But data points are rarely linearly separable or they are so intricately mixed that they are not even separable. In that complex case, say data points are not separable in x-y plane, then we add another dimension z, and
# plot the 2D space points into 3D use a hyperplane to separate and transform that hyperplane 
# back to the original 2D space, thereby getting a separation in the original space. We call these transformations as ‘Kernels’.
# 
# As a matter of fact, there can be multiple hyperplanes separating the data-points. Which one should be chosen as the decision boundary?
# The one that maximizes the smallest distance between the data points of both the classes seems to be a natural choice,
# providing better margins. Support vectors are the sample data points which lie clode to the decision boundary.
# 
# The loss function, which is a function of data point, prediction class and actual label, is actually a measure of the penaly for wrong prediction.
# SVM uses Hinge loss as the loss function, given as below:

# <img src="hinge.PNG" width="400" height="250">

# It tells us how better we are doing with respect to one training example. When we sum over all the training examples, what we get is the cost function. 
# 
# The optimization problem finally looks like:

# <img src="cost.PNG" width="400" height="250">

# where R(w) is the regularization function used to tackle overfiiting, 
# C is the scalar constant,
# and L is the loss fuction.
# 
# Below is the python implementation of SVM using scikit learn library

# # Importing the libraries

# In[ ]:


import csv 
import re
import xml.etree.ElementTree as ET 
import os
import pandas as pd
import string
import nltk
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import metrics


# In[5]:


data = r'C:\Users\Aditi Sethia\HTML\training\pan12.xml'


# # Parsing the xml file

# In[7]:


tree = ET.parse(data)
myroot = tree.getroot()


# In[10]:


def getvalueofnode(node):
    return node.text if node is not None else None
def rem_num(text):
    output = re.sub('[0-9]+', '', text)
    return output
def use(x):
    o = 0
    for i in x:
        if i in lis:
            o = o + 1
            return 1
    if o == 0:
        return 0    

def rem_stopword(text):
    tokens = text.split()
    result = [i for i in tokens if not i in stop_words]
    return (' '.join(result))


# In[13]:


k = ''
d = 0
df = pd.DataFrame(columns = ['Users','Conversation_ID','text'])
for i in myroot:
    user = {}
    k = ''
    p = i.attrib.get('id')
    for j in i:
        l = j.find('text')
        k = k + ' ' + str(getvalueofnode(l))
        auth = j.find('author')
        auth = getvalueofnode(auth)
        if auth in user:
            user[auth] = user[auth] + 1
        else:
            user[auth] = 1
    list_user = [i for i in user]
    df = df.append({'Users' :list_user  ,'Conversation_ID': p,'text':k},ignore_index=True)


# # Text-preprocessing

# In[14]:


def text_preprocessing(df):
    df["text"] = df['text'].apply(lambda x: x.lower())
    df["text"] = df['text'].apply(lambda x: ''.join(c for c in x if c not in punctuation))
    df['text'] = df['text'].apply(rem_num)
text_preprocessing(df = df)


# In[17]:


file1 = open(r"\training\tcorpus.txt","r+")
lis = file1.readlines()
lis = [i[:-1] for i in lis]


# In[20]:


def work(df):
    df['Predator_Present'] = df['Users'].apply(use) 
work(df = df)


# In[22]:


from sklearn.utils import resample


# In[31]:


df_majority = df[df['Predator_Present'] == 0]
df_minority = df[df['Predator_Present'] == 1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=4000,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])


# In[33]:


df_downsampled.tail()


# In[35]:


df_downsampled.shape


# # Downsampling of data 

# In[37]:


from sklearn.utils import resample

df_majority = df[df['Predator_Present'] == 0]
df_minority = df[df['Predator_Present'] == 1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False, 
                                 n_samples=4032,    
                                 random_state=123) 
 

df_downsampled = pd.concat([df_majority_downsampled, df_minority])

y = df_downsampled['Predator_Present']
y = y.values

corpus_DS = []
for i in df_downsampled["text"]:
    corpus_DS += [i]


# # Converting the data-set into vectors (Tf-idf)

# In[43]:


vectorizer = TfidfVectorizer(stop_words = 'english')
X_vect = vectorizer.fit_transform(corpus_DS).toarray()
X_vect.shape


# # Splitting the data into training and testing samples
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size = 0.2, random_state = 0)


# # Fitting the model

# In[ ]:


classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# # Results

# In[47]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(metrics.f1_score(y_test, y_pred, average='macro'))


# References:
# 1. https://monkeylearn.com/text-classification/
# 2. https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
