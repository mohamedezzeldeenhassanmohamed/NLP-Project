#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import pytextrank


# In[2]:


nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")


# In[3]:


def summarization(example_text):
    doc = nlp(example_text)
    txt=""
    for sent in doc._.textrank.summary(limit_phrases=2, limit_sentences=2):
          txt+=sent.text
    return txt


# In[ ]:




