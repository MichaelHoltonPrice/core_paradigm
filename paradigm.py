import os
import re
import itertools
import lxml
from lxml.html import fromstring
from lxml.html import tostring
import unicodedata
#from spellchecker import SpellChecker

class CoreCleaner():
    # A class to do the cleaning for the Core Econ paradigm text analysis
    # project.
    
    def __init__(self,rootDir,ukFile,usFile):
        self.rootDir = rootDir
        with open(ukFile, 'r') as f:    
            self.UK = f.read().replace('\u2028',' ').replace('\xa0','').splitlines()
            self.UK = self.UK[0].split()
    
        with open(usFile, 'r') as f:    
            self.US = f.read().replace('\u2028',' ').replace('\xa0','').splitlines()
            self.US = self.US[0].split()
        #self.spell = SpellChecker()

    def cleanTextFile(self,textFilePath,regex=None,fixMistakes=True):
        print("Cleaning " + textFilePath)
        if regex is None:
            regex = re.compile(r'([^a-z\- ])|((^| ).( |$))')

        output = [line.strip() for line in open(textFilePath,"r")] 
        output = [[word.lower() for word in text.split()] for text in output]
        output = list(itertools.chain.from_iterable(output))
        output = ' '.join(str(x) for x in output)
        output = fromstring(output).text_content() #removing weird html stuff
        output = regex.sub('',output)
        output = regex.sub('',output)
        if fixMistakes:
            output = self.fixMistakes(output)
        output = self.standardizeSpelling(output)
        return output

    def fixMistakes(self,text):
        # Fix mistakes, such as missing spaces between and mis-spellings
        # text is an input string of words
        tok = text.split() # tokenize the text into individual words

        # (1) Search for merged words (lacking space between them due to OCR)
        badWords = self.spell.unknown(tok)
        for bad in badWords:
            fix = self.checkForMissingSpace(bad)
            ind = [i for i,x in enumerate(tok) if x==bad]
            while len(ind) is not 0:
                ind = ind[0]
                del tok[ind] # remove the value
                if fix is not None:
                    # If a fix was found, apply it
                    w1,w2 = fix.split()
                    tok.insert(ind,w1)
                    tok.insert(ind+1,w2)
                ind = [i for i,x in enumerate(tok) if x==bad]

        # (2) Search for word-parts separated by one hyphen (almost certainly
        #     words spilling over two lines in the original PDF
        # [Not implementing this, at least for now]
        output = ' '.join(str(x) for x in tok)
        return(output)

    def preprocessFiles(self,targetDir):
        text = []
        #for dirname, subdirname, files in os.walk(self.rootDir):
        for dirname, subdirname, files in os.walk(targetDir):
            for f in files:
                if f.endswith('.txt'):
                    try:
                        path = os.path.join(dirname,f)
                        text.append((path,self.cleanTextFile(path,fixMistakes=False)))
                        #text.append(self.cleanTextFile(path))
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

    def standardizeSpelling(self,text):
        tok = text.split()
        for i,usWord in enumerate(self.US):
            if usWord in tok:
                tok = [self.UK[i] if w==usWord else w for w in tok]
        output = ' '.join(str(x) for x in tok)
        return(output)
            
    def checkForMissingSpace(self,word):
        # For the input, misspelled word iterate over locations where a space could
        # go to identify merged words. The first successful candidate is returned.
        for i in range(1,len(word)):
            w1 = word[0:i]
            w2 = word[i:]
            if len(self.spell.unknown([w1])) == 0 and len(self.spell.unknown([w2])) == 0:
                return(w1 + " " + w2)
        return None
