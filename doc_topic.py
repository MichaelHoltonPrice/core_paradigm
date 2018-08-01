#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:09:06 2018

@author: sahana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 13:19:07 2018

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

homeDir = os.getenv("HOME")

rootDir = os.path.join(homeDir,"core_data")
outputDir = os.path.join(rootDir,"pickles")

model =  models.LdaModel.load(os.path.join(rootDir,'lda_output/ldamodel_output100'))
"""
#getting textbooks

with open(os.path.join(outputDir,"textbooks.pkl"),"rb") as f:
    textbooks = pickle.load(f)

with open(os.path.join(rootDir,"stopwords.dat"),"r") as f:
    stopwords = f.readlines()
    stopwords = [l.strip('\n\r') for l in stopwords]
    
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
cleaned_texts = []

#bigramming

sentence_split = [i.split(" ") for i in textbooks]

phrases = Phrases(sentence_split)


bigrammed = Phraser(phrases)

bigrammed = list(bigrammed[sentence_split])

bigrammed_textbooks = [' '.join(i) for i in bigrammed if len(i) > 1]

#stem toke stop
for i in range(len(bigrammed_textbooks)):
    #tokenize document
    #lower_case = training_flat[i].lower()
    tokens = tokenizer.tokenize(bigrammed_textbooks[i])
    
    #stopwords
    stopped = [i for i in tokens if not i in stopwords]
    
    #stemmed
    stemmed = [stemmer.stem(i) for i in stopped]
    
    cleaned_texts.append(stemmed)

doc_corpus = cleaned_texts

with open(os.path.join(outputDir,"doc_corpus.pkl"),"wb") as f:
    pickle.dump(doc_corpus,f)

"""
with open(os.path.join(outputDir,"doc_corpus.pkl"),"rb") as f:
    doc_corpus = pickle.load(f)

doc_corpus_codes = ["samuelson","marx","hayek","krugman","sam_nord","acemoglu","smith","keynes","CORE","marshall","mill","mankiw"]
#tokenizing
tokenizer = RegexpTokenizer(r'\w+')

#doc_corpus = [tokenizer.tokenize(i) for i in doc_corpus]

txtbook_dictionary = corpora.Dictionary(doc_corpus)
corpus = [txtbook_dictionary.doc2bow(t)for t in doc_corpus]

for i,j in enumerate(corpus):
    #topic_dist = model.get_document_topics(j)
    #doc_topic = topic_dist.sort(key= lambda x: x[1])
    print("doc {0}: {1}".format(i, doc_corpus_codes[i]))
    print(model.get_document_topics(j))
    print("====================================")