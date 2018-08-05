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

# for cleaning
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
cleaned_texts = []

with open(os.path.join(outputDir,"journals.pkl"),"rb") as f:
    journals = pickle.load(f)

#with open(os.path.join(outputDir,"journals_sand.pkl"),"rb") as f:
   # journals_sand = pickle.load(f)
#unlisting journals
journals_flat = [i for i in list(chain.from_iterable(journals))]  
#j_sand_flat = [i for i in list(chain.from_iterable(journals_sand))]

with open(os.path.join(outputDir,"nber_part1.pkl"),"rb") as f:
    nber_part1 = pickle.load(f)   

with open(os.path.join(outputDir,"textbooks.pkl"),"rb") as f:
    textbooks = pickle.load(f)

training_data = []
#training_data_sand = []
#training_data_sand.extend((j_sand_flat,textbooks))
#tr_s_flat = [i for i in list(chain.from_iterable(training_data_sand))]
training_data.extend((journals_flat,nber_part1,textbooks))
print("training data appended")

training_flat = [i for i in list(chain.from_iterable(training_data))]  
print("training data flattened")

paths = [i[0] for i in training_flat]

with open(os.path.join(outputDir,"paths_to_data.pkl"),"wb") as f:
    pickle.dump(paths,f)

training_texts = [i[1] for i in training_flat]

#loading stopwords
with open(os.path.join(rootDir,"stopwords.dat"),"r") as f:
    stopwords = f.readlines()
    stopwords = [l.strip('\n\r') for l in stopwords]
print("stopwords made")

for i in range(len(training_texts)):
    if i%1000 == 0:
        print(i)
    #tokenize document
    #lower_case = training_flat[i].lower()
    tokens = tokenizer.tokenize(training_texts[i])
    
    
    #stopwords
    stopped = [i for i in tokens if not i in stopwords]
    
    #stemmed
    stemmed = [stemmer.stem(i) for i in stopped]
    
    cleaned_texts.append(stemmed)

print("cleaning done")

#bigramming 

cleaned_sent = [' '.join(i) for i in cleaned_texts]
#bigramming

sentence_split = [i.split(" ") for i in cleaned_sent]
print("sentence split")

phrases = Phrases(sentence_split)

print("phrases made")

bigrammed = Phraser(phrases)
print("bigrams made")

bigrammed = list(bigrammed[sentence_split])
print("bigrams made into list")

training_data_bigrammed = [' '.join(i) for i in bigrammed if len(i) > 1]
print("training_data_bigrammed made")

with open(os.path.join(outputDir,"training_data_bigrammed.pkl"),"wb") as f:
    pickle.dump(training_data_bigrammed,f)

    training_data_bigrammed = pickle.load(f)

print("bigrammed training data pickled")  

#tokenizing
training_data_final = [tokenizer.tokenize(i) for i in training_data_bigrammed]

with open(os.path.join(outputDir,"training_data_final.pkl"),"wb") as f:
    pickle.dump(training_data_final,f)

#make it into a dictionary
dictionary = corpora.Dictionary(training_data_final)
print("corpora made")

dictionary.filter_extremes(no_below = 0.01*len(training_data_final))

with open(os.path.join(outputDir,"dictionary.pkl"),"wb") as f:
    pickle.dump(dictionary,f)

#creating a document-term matrix
corpus = [dictionary.doc2bow(text) for text in training_data_final]
print("corpus ready")

with open(os.path.join(outputDir,"corpus.pkl"),"wb") as f:
    pickle.dump(corpus,f)
#LDA model
#ldamodel10 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, 
#                                           id2word = txtbook_dictionary,
#                                           passes = 5)
#ldamodel10.save('ldamodel_output10')


#ldamodel100 = gensim.models.ldamulticore.LdaMulticore(corpus,num_topics = 100,id2word = txtbook_dictionary,passes = 5)

ldamodel100 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 100,id2word = dictionary ,passes = 5)
ldamodel100.save(os.path.join(rootDir,'lda_output/ldamodel_output100'))

#ldamodel1000 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 1000, 
#                                           id2word = txtbook_dictionary,
#                                           passes = 5)
#ldamodel1000.save('ldamodel_output1000')


print("LDA MODEL READY")
#to get results 
#ldamodel100.print_topics(num_topics=10, num_words=10)



print("AAAND WE'RE DONE")



