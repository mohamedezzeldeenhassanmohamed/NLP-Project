#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[51]:


df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")


# In[52]:


df_fake.head(5)


# In[53]:


df_true.head(5)


# In[54]:


df_fake["class"] = 0
df_true["class"] = 1


# In[55]:


df_fake.shape, df_true.shape


# In[56]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


# In[57]:


df_fake.shape, df_true.shape


# In[58]:


df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1


# In[59]:


df_fake_manual_testing.head(10)


# In[60]:


df_true_manual_testing.head(10)


# In[61]:


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")


# In[62]:


df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)


# In[63]:


df_marge.columns


# In[64]:


df = df_marge.drop(["title", "subject","date"], axis = 1)


# In[65]:


df.isnull().sum()


# In[66]:


df = df.sample(frac = 1)


# In[67]:


df.head()


# In[68]:


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)


# In[69]:


df.columns


# In[70]:


df.head()


# In[71]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[72]:


df["text"] = df["text"].apply(wordopt)


# In[73]:


x = df["text"]
y = df["class"]


# In[74]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[75]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[76]:


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[77]:


from sklearn.linear_model import LogisticRegression


# In[78]:


LR = LogisticRegression()
LR.fit(xv_train,y_train)


# In[79]:


pred_lr=LR.predict(xv_test)


# In[80]:


LR.score(xv_test, y_test)


# In[81]:


print(classification_report(y_test, pred_lr))


# In[82]:


from sklearn.tree import DecisionTreeClassifier


# In[83]:


DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[84]:


pred_dt = DT.predict(xv_test)


# In[85]:


DT.score(xv_test, y_test)


# In[86]:


print(classification_report(y_test, pred_dt))


# In[87]:


from sklearn.ensemble import GradientBoostingClassifier


# In[88]:


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)


# In[89]:


pred_gbc = GBC.predict(xv_test)


# In[90]:


GBC.score(xv_test, y_test)


# In[91]:


print(classification_report(y_test, pred_gbc))


# In[92]:


from sklearn.ensemble import RandomForestClassifier


# In[93]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)


# In[94]:


pred_rfc = RFC.predict(xv_test)


# In[95]:


RFC.score(xv_test, y_test)


# In[96]:


print(classification_report(y_test, pred_rfc))


# In[97]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return "\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0]))

