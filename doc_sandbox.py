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
outputDir = os.path.join(rootDir,"pickles")

model =  models.LdaModel.load(os.path.join(rootDir,'lda_output/ldamodel_output100'))

# This is the full mapping of integers to words that model uses internally
wordHashFull = model.id2word

#getting textbooks

with open(os.path.join(outputDir,"textbooks.pkl"),"rb") as f:
    textbooks = pickle.load(f)

with open(os.path.join(outputDir,"articles.pkl"),"rb") as f:
    articles = pickle.load(f)

with open(os.path.join(outputDir,"core_chapters.pkl"),"rb") as f:
    core_chapters = pickle.load(f)

echo_corpus = []
echo_corpus.extend(textbooks)
echo_corpus.extend(articles)
echo_corpus.extend(core_chapters)

paths = [i[0] for i in echo_corpus]
echo_corpus = [i[1] for i in echo_corpus]
#CLEANING
with open(os.path.join(rootDir,"stopwords.dat"),"r") as f:
    stopwords = f.readlines()
    stopwords = [l.strip('\n\r') for l in stopwords]

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
cleaned_texts = []

#bigramming

#stem toke stop
for i in range(len(echo_corpus)):
    #tokenize document
    #lower_case = training_flat[i].lower()
    tokens = tokenizer.tokenize(echo_corpus[i])
    
    #stopwords
    stopped = [i for i in tokens if not i in stopwords]
    
    #stemmed
    stemmed = [stemmer.stem(i) for i in stopped]
    
    cleaned_texts.append(stemmed)

sent_text = [" ".join(i) for i in cleaned_texts]
sentence_split = [i.split(" ") for i in sent_text]

phrases = Phrases(sentence_split)

bigrammed = Phraser(phrases)

bigrammed = list(bigrammed[sentence_split])

#bigrammed_textbooks = [' '.join(i) for i in bigrammed if len(i) > 1]


doc_corpus = bigrammed



#tokenizing
tokenizer = RegexpTokenizer(r'\w+')

#doc_corpus = [tokenizer.tokenize(i) for i in doc_corpus]

#txtbook_dictionary = corpora.Dictionary(doc_corpus)

doc_corpus = [wordHashFull.doc2bow(t) for t in doc_corpus]

with open(os.path.join(outputDir,"doc_corpus.pkl"),"wb") as f:
    pickle.dump(doc_corpus,f)




#getting corpus codes
#with open(os.path.join(rootDir,"echo_corpus_order")) as f:
#   content = f.readlines()
#doc_corpus_codes = [x.strip() for x in content]
    
    
doc_corpus_codes = [os.path.splitext(os.path.basename(i))[0] for i in paths]
doc_corpus_codes[24:46] =["chapter_"+ doc_corpus_codes[i] for i in range(24,46)]

with open(os.path.join(outputDir,"doc_corpus_codes.pkl"),"wb") as f:
    pickle.dump(doc_corpus_codes,f) 
    
with open(os.path.join(outputDir,"doc_corpus.pkl"),"rb") as f:
    doc_corpus = pickle.load(f)
    
#get shannon info
def shannon(doc_topics):
    shannon_inf = 0
    for i in doc_topics:
        p = i[1]
        if p != 0:
            log_p = math.log2(p)
            entropy = -p*log_p
        else:
            entropy = 0
        shannon_inf += entropy
    return shannon_inf


topic_distr = []

for i,j in enumerate(doc_corpus):
    topic_dist = model.get_document_topics(j,minimum_probability = 0.00)
    topic_distr.append(topic_dist)
    #doc_topic = topic_dist.sort(key= lambda x: x[1])
    print("doc {0}: {1}, shannon entropy: {2}".format(i, doc_corpus_codes[i],shannon(topic_dist)))
    print(" ")
    print(model.get_document_topics(j))
    print("====================================")

with open(os.path.join(outputDir,"topic_distr.pkl"),"wb") as f:
    pickle.dump(topic_distr,f)

