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

dropboxDir = os.path.join(homeDir,"Dropbox")

rootDir = os.path.join(dropboxDir,"core_data")
outputDir = os.path.join(rootDir,"pickles")

with open(os.path.join(outputDir,"dictionary.pkl"),"rb") as f:
    dictionary = pickle.load(f)   

with open(os.path.join(outputDir,"corpus.pkl"),"rb") as f:
    corpus = pickle.load(f)   

ldamodel100 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 100,id2word = dictionary ,passes = 5)
ldamodel100.save(os.path.join(rootDir,'lda_output/ldamodel_output100'))
