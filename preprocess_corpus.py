import os
import re
import itertools
from itertools import chain
import pickle
import paradigm

# print("done imports")

# clean fn: txt file to lower case alphabet only string

# """Collecting .txt files to clean"""



to_clean = []
error_files = []
homeDir = os.getenv("HOME")
rootDir = os.path.join(homeDir,"core_data")
journalRootDir = os.path.join(rootDir,"journal_data")
outputDir = os.path.join(rootDir,"pickles")

#ukFile = ""
#usFile = ""

#cleaner = paradigm.CoreCleaner(rootDir,usFile,ukFile)
ukFile = os.path.join(homeDir,"core_data","UK_words.dat")
usFile = os.path.join(homeDir,"core_data","US_words.dat")
cleaner = paradigm.CoreCleaner(rootDir,ukFile,usFile)


#p = os.path.join(rootDir,"NBER_txt","w7104.txt")
#t0 = cleaner.cleanTextFile(p,fixSpelling=False)
#t1 = cleaner.cleanTextFile(p)

JEP = cleaner.preprocessFiles(os.path.join(journalRootDir,"JEP"))
QJE = cleaner.preprocessFiles(os.path.join(journalRootDir,"QJE"))
AER = cleaner.preprocessFiles(os.path.join(journalRootDir,"AER"))
JPE = cleaner.preprocessFiles(os.path.join(journalRootDir,"JPE"))
EJ = cleaner.preprocessFiles(os.path.join(journalRootDir,"EJ"))

textbooks = cleaner.preprocessFiles(os.path.join(rootDir,"textbooks_txt"))
articles = cleaner.preprocessFiles(os.path.join(rootDir,"articles"))
core_chapters = cleaner.preprocessFiles(os.path.join(rootDir,"core_chapters"))

nber_part0 = cleaner.preprocessFiles(os.path.join(rootDir,"NBER_txt"))
nber_part1 = cleaner.cleanNBER(nber_part0)

# pickling files

journals = []
#journals_sand = []
#journals_sand.extend((JEP,articles))
journals.extend((JEP,QJE,AER,JPE,EJ))

#for home computer:

with open(os.path.join(outputDir,"textbooks.pkl"),"wb") as f:
    pickle.dump(textbooks,f)

with open(os.path.join(outputDir,"articles.pkl"),"wb") as f:
    pickle.dump(articles,f)

with open(os.path.join(outputDir,"core_chapters.pkl"),"wb") as f:
    pickle.dump(core_chapters,f)
    
with open(os.path.join(outputDir,"nber_part1.pkl"),"wb") as f:
    pickle.dump(nber_part1,f)

with open(os.path.join(outputDir,"journals.pkl"),"wb") as f:
    pickle.dump(journals,f)


#with open(os.path.join(outputDir,"journals_sand.pkl"),"wb") as f:
#    pickle.dump(journals_sand,f)
