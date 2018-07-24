import os
import re
import itertools
from itertools import chain
import pickle
import lxml
from lxml.html import fromstring
from lxml.html import tostring
import paradigm

# print("done imports")

# clean fn: txt file to lower case alphabet only string

# """Collecting .txt files to clean"""

to_clean = []
error_files = []
homeDir = os.getenv("HOME")
rootDir = os.path.join(homeDir,"core_paradigm")
journalRootDir = os.path.join(rootDir,"journal_data")

JEP = paradigm.preprocessFiles(os.path.join(journalRootDir,"JEP"))
QJE = paradigm.preprocessFiles(os.path.join(journalRootDir,"QJE"))
AER = paradigm.preprocessFiles(os.path.join(journalRootDir,"AER"))
JPE = paradigm.preprocessFiles(os.path.join(journalRootDir,"JPE"))
EJ = paradigm.preprocessFiles(os.path.join(journalRootDir,"EJ"))

textbooks = paradigm.preprocessFiles(os.path.join(rootDir,"textbooks_txt"))

nber_part0 = paradigm.preprocessFiles(os.path.join(rootDir,"NBER_txt"))
nber_part1 = paradigm.cleanNBER(nber_part0)

# pickling files

journals = []
journals.extend((JEP,QJE,AER,JPE,EJ))

#for home computer:

outputDir = os.path.join(rootDir,"pickles")
with open(os.path.join(outputDir,"textbooks.pkl"),"wb") as f:
    pickle.dump(textbooks,f)

with open(os.path.join(outputDir,"nber_part1.pkl"),"wb") as f:
    pickle.dump(nber_part1,f)

with open(os.path.join(outputDir,"journals.pkl"),"wb") as f:
    pickle.dump(journals,f)
