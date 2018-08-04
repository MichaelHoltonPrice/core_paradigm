#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 13:04:46 2018

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

homeDir = os.getenv("HOME")

rootDir = os.path.join(homeDir,"core_data")


model =  models.LdaModel.load(os.path.join(rootDir,'lda_output/ldamodel_output100'))

topics100 =  model.print_topics(num_topics=100, num_words=10)




for i,j in enumerate(topics100):
    print("topic %s" %i)
    print(" ")
    print(j)
    print(" ")
    print("=====================================================================")
    print(" ")


