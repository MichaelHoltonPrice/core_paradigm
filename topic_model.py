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

rootDir = os.path.join(homeDir,"core_paradigm")
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

bigrammed = Phraser(phrases)
print("bigrams made")

bigrammed = list(bigrammed[sentence_split])
print("bigrams saved")

training_data_bigrammed = [' '.join(i) for i in bigrammed]
print("training_data_bigrammed made")

# cleaning
tokenizer = RegexpTokenizer(r'\w+')
stopwords = get_stop_words('en')
stemmer = PorterStemmer()
#additional to extend
additional = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
              "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", 
              "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", 
              "their", "theirs", "themselves", "what", "which", "who", "whom", "this", 
              "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
              "being", "have", "has", "had", "having", "do", "does", "did", "doing",
              "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
              "while", "of", "at", "by", "for", "with", "about", "against", "between", 
              "into", "through", "during", "before", "after", "above", "below", "to", 
              "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
              "further", "then", "once", "here", "there", "when", "where", "why",
              "how", "all", "any", "both", "each", "few", "more", "most", "other",
              "some", "such", "no", "nor", "not", "only", "own", "same", "so",
              "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
              "now", "may", "one","upon","great","bigger","also","year"]
stopwords.extend([i for i in additional if i not in stopwords])

print("stopwords made")

cleaned_texts = []

for i in range(len(training_flat)):
    if i%1000 == 0:
        print(i)
    #tokenize document
    #lower_case = training_flat[i].lower()
    tokens = tokenizer.tokenize(training_flat[i])
    
    
    #stopwords
    stopped = [i for i in tokens if not i in stopwords]
    
    #stemmed
    stemmed = [stemmer.stem(i) for i in stopped]
    
    cleaned_texts.append(stemmed)

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

ldamodel100 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 100, 
                                           id2word = txtbook_dictionary,
                                           passes = 5)
ldamodel100.save('ldamodel_output100')

#ldamodel1000 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 1000, 
#                                           id2word = txtbook_dictionary,
#                                           passes = 5)
#ldamodel1000.save('ldamodel_output1000')


print("LDA MODEL READY")
#to get results 
#ldamodel.print_topics(num_topics=10, num_words=10)



print("AAAND WE'RE DONE")



