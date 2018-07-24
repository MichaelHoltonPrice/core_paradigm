import os
import re
import itertools
import lxml
import unicodedata

def clean_txt(txt_path,regex=None):
    if regex is None:
        regex = re.compile(r'([^a-z\- ])|((^| ). )')

    output = [line.strip() for line in open(txt_path,"r")] 
    output = [[word.lower() for word in text.split()] for text in output]
    output = list(itertools.chain.from_iterable(output))
    output = ' '.join(str(x) for x in output)
    output = lxml.html.fromstring(output).text_content() #removing weird html stuff
    output = regex.sub('',output)
    output = regex.sub('',output)
    return output

def preprocessFiles(rootDir):
    text = []
    for dirname, subdirname, files in os.walk(rootDir):
        for f in files:
            if f.endswith('.txt'):
                try:
                    path = os.path.join(dirname,f)
                    text.append(clean_txt(path))
                except:
                    print(path)
    return text

def cleanNBER(textList):
    pattern=re.compile(".*?abstract(.*?)(?:references|bibliography|$).*") 
    problem = [] #697/17647 nber papers were problems

    for i in range(len(textList)):
        try:
            textList[i] = re.match(pattern,textList[i]).group(1)
        except:
            problem.append(i)
    return(textList)
