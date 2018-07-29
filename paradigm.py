import os
import re
import itertools
import lxml
from lxml.html import fromstring
from lxml.html import tostring
import unicodedata
from spellchecker import SpellChecker

class CoreCleaner():
    # A class to do the cleaning for the Core Econ paradigm text analysis
    # project.
    
    def __init__(self,rootDir,ukFile=None,usFile=None):
        self.rootDir = rootDir
        if ukFile is not None:
            with open(uk_dir, 'r') as f:    
                self.UK = f.read().replace('\u2028',' ').replace('\xa0','').splitlines()
                self.UK = self.UK[0].split()
    
        if usFile is not None:
            with open(us_dir, 'r') as f:    
                self.US = f.read().replace('\u2028',' ').replace('\xa0','').splitlines()
                self.US = self.US[0].split()
        self.spell = SpellChecker()

    def cleanTextFile(self,textFilePath,regex=None,fixSpelling=True):
        print("Cleaning " + textFilePath)
        if regex is None:
            regex = re.compile(r'([^a-z\- ])|((^| ). )')

        output = [line.strip() for line in open(textFilePath,"r")] 
        output = [[word.lower() for word in text.split()] for text in output]
        output = list(itertools.chain.from_iterable(output))
        output = ' '.join(str(x) for x in output)
        output = fromstring(output).text_content() #removing weird html stuff
        output = regex.sub('',output)
        output = regex.sub('',output)
        tok = output.split() # tokenize
        if fixSpelling:
            # Search for merged words (lacking space between them due to OCR)
            badWords = self.spell.unknown(tok)
            for bad in badWords:
                fix = self.checkForMissingSpace(bad)
                # Remove the word
                ind = [i for i,x in enumerate(tok) if x==bad]
                while len(ind) is not 0:
                    ind = ind[0]
                    del tok[ind] # remove the value
                    if fix is not None:
                        # If a fix was found, add the fix
                        w1,w2 = fix.split()
                        tok.insert(ind,w1)
                        tok.insert(ind+1,w2)
                    ind = [i for i,x in enumerate(tok) if x==bad]
            output = ' '.join(str(x) for x in tok)
        return output

    def preprocessFiles(self,targetDir):
        text = []
        #for dirname, subdirname, files in os.walk(self.rootDir):
        for dirname, subdirname, files in os.walk(targetDir):
            for f in files:
                if f.endswith('.txt'):
                    try:
                        path = os.path.join(dirname,f)
                        text.append(self.cleanTextFile(path,fixSpelling=False))
                    except:
                        print(path)
        return text

    def cleanNBER(self,textList):
        pattern=re.compile(".*?abstract(.*?)(?:references|bibliography|$).*") 
        problem = [] #697/17647 nber papers were problems

        for i in range(len(textList)):
            try:
                textList[i] = re.match(pattern,textList[i]).group(1)
            except:
                problem.append(i)
        return(textList)

    def checkForMissingSpace(self,word):
        # For the input, misspelled word iterate over locations where a space could
        # go to identify merged words. The first successful candidate is returned.
        for i in range(1,len(word)):
            w1 = word[0:i]
            w2 = word[i:]
            if len(self.spell.unknown([w1])) == 0 and len(self.spell.unknown([w2])) == 0:
                return(w1 + " " + w2)
        return None
