#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:45:37 2018

@author: sahana
"""
import re
import os
import itertools
from itertools import chain
import pickle
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import corpora,models
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

homeDir = os.getenv("HOME")

rootDir = os.path.join(homeDir,"core_data")
outputDir = os.path.join(rootDir,"pickles")


with open(os.path.join(outputDir,"topic_distr.pkl"),"rb") as f:
    topic_distr = pickle.load(f)
    
with open(os.path.join(outputDir,"doc_corpus_codes.pkl"),"rb") as f:
    doc_corpus_codes = pickle.load(f)


def kld(p,q):
    KLD = []
    for i in range(100):
        kld_i =p[i] * math.log2(p[i]/q[i])
        KLD.append(kld_i)
    return sum(KLD)
    
kld_all = []
for i in range(len(topic_distr)):
    for j in range(len(topic_distr)):
        p = [x[1] for x in topic_distr[i]]
        q = [x[1] for x in topic_distr[j]]
        doc1 = doc_corpus_codes[i]
        doc2 = doc_corpus_codes[j]
        kld_all.append((doc1,doc2,kld(p,q)))
        
        
doc1 = []
doc2 = []
kld_value = []

for i in kld_all:
    doc1.append(i[0])
    doc2.append(i[1])
    kld_value.append(i[2])

counter = 0
kld_heat = []
while counter <len(kld_value):
   kld_heat.append(kld_value[counter:counter+45])
   counter+=45

#reodering order by year
   
with open(os.path.join(rootDir,"corpus_year_order")) as f:
    content = f.readlines()
by_year_all = [x.strip() for x in content]


#all documents heat map

all_df = pd.DataFrame({"doc1":doc1,"doc2":doc2,"kld":kld_value})
res_all = all_df.pivot("doc1","doc2","kld")
res_all = res_all.reindex(index = by_year_all, columns = by_year_all)
plt.figure(figsize = (20,20))
sns.heatmap(res_all,cmap = "magma_r")

#only textbooks
txtbook_codes = doc_corpus_codes[0:13]
txt1 = []
txt2 = []
kld_txt = []
for i, e in enumerate(doc1):
    if e in set(txtbook_codes) and doc2[i] in set(txtbook_codes):
        txt1.append(e)
        txt2.append(doc2[i])
        kld_txt.append(kld_value[i])
txt_df = pd.DataFrame({"text1":txt1,"text2":txt2,"kld":kld_txt})
res_txt = txt_df.pivot("text1","text2","kld")
by_year_txt = ["smith","ricardo","mill","marx","marshall","keynes","samuelson","hayek",
                 "samuelson_nordhaus","krugman","acemoglu","core","mankiw"]
res_txt = res_txt.reindex(index = by_year_txt, columns = by_year_txt)
plt.figure(figsize = (10,10))
sns.heatmap(res_txt, annot= True,cmap = "magma_r")
