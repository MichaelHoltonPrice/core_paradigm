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

with open(os.path.join(outputDir,"journals.pkl"),"rb") as f:
    journals = pickle.load(f)

#unlisting journals
journals_flat = [i for i in list(chain.from_iterable(journals))]  


with open(os.path.join(outputDir,"nber_part1.pkl"),"rb") as f:
    nber_part1 = pickle.load(f)   

with open(os.path.join(outputDir,"textbooks.pkl"),"rb") as f:
    textbooks = pickle.load(f)

training_data = []
training_data.extend((journals_flat,nber_part1,textbooks))
print("training data appended")

training_flat = [i for i in list(chain.from_iterable(training_data))]  
print("training data flattened")

#bigramming

sentence_split = [i.split(" ") for i in training_flat]
print("sentence split")

phrases = Phrases(sentence_split)

print("phrases made")

# Switch ordering of stemming and bi-gramming
bigrammed = Phraser(phrases)
print("bigrams made")

bigrammed = list(bigrammed[sentence_split])
print("bigrams saved")

training_data_bigrammed = [' '.join(i) for i in bigrammed]
print("training_data_bigrammed made")

#loading stopwords
with open(os.path.join(rootDir,"stopwords.dat"),"r") as f:
    stopwords = f.readlines()
    stopwords = [l.strip('\n\r') for l in stopwords]
print("stopwords made")

# cleaning
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
cleaned_texts = []

for i in range(len(training_data_bigrammed)):
    if i%1000 == 0:
        print(i)
    #tokenize document
    #lower_case = training_flat[i].lower()
    tokens = tokenizer.tokenize(training_data_bigrammed[i])
    
    
    #stopwords
    stopped = [i for i in tokens if not i in stopwords]
    
    #stemmed
    stemmed = [stemmer.stem(i) for i in stopped]
    
    cleaned_texts.append(stemmed)
ldamodel100 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 100,id2word = txtbook_dictionary,passes = 5)

print("cleaning done")

with open("cleaned_texts.pkl","wb") as f:
    pickle.dump(cleaned_texts,f)

print("pickled")  

#make it into a dictionary

txtbook_dictionary = corpora.Dictionary(cleaned_texts)
print("corpora made")
#creating a document-term matrix
corpus = [txtbook_dictionary.doc2bow(text) for text in cleaned_texts]
print("corpus ready")
#LDA model
#ldamodel10 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, 
#                                           id2word = txtbook_dictionary,
#                                           passes = 5)
#ldamodel10.save('ldamodel_output10')


#ldamodel100 = gensim.models.ldamulticore.LdaMulticore(corpus,num_topics = 100,id2word = txtbook_dictionary,passes = 5)

ldamodel100 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 100,id2word = txtbook_dictionary,passes = 5)
ldamodel100.save('ldamodel_output100')

#ldamodel1000 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 1000, 
#                                           id2word = txtbook_dictionary,
#                                           passes = 5)
#ldamodel1000.save('ldamodel_output1000')


print("LDA MODEL READY")
#to get results 
#ldamodel.print_topics(num_topics=10, num_words=10)



print("AAAND WE'RE DONE")



