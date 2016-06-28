import json
import codecs
import os
import sys
import re
import numpy
import gensim
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from gensim import models,corpora,similarities
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def nameDoc2vec_representation(textfile,dictfile,representationfile):
    appname=[]
    with open(textfile,'r') as fin:
        for line in fin:
            line=json.loads(line)
            appname.append(line['name'])
    print("Preprocessing Name Text done!")

    tokenizer=RegexpTokenizer(r'\w+')
    appnametext=[]
    count=0
    for namestr in appname:
        wordlist=tokenizer.tokenize(namestr)
        wordlist_rmstopword=[word for word in wordlist if word not in stopwords.words('english')]
        # wordlist_rmstopword=wordlist
        for i in range(len(wordlist_rmstopword)):
            wordlist_rmstopword[i]=nltk.PorterStemmer().stem_word(wordlist_rmstopword[i])
        Lnumber=[word for word in wordlist_rmstopword if re.match(r'\d+$',word)]
        Lothers=[word for word in wordlist_rmstopword if re.match(r'^_+',word)]
        wordlist_rmLnumber=[word for word in wordlist_rmstopword if word not in Lnumber]
        wordlist_rmLothers=[word for word in wordlist_rmLnumber if word not in Lothers]
        appnametext.append(wordlist_rmLothers)
        if count%1000==0:
            # print(str(count)+"...",end='')
            print count
        count=count+1

        
    # appname.clear()
    print("Obtain Name Wordlist done!")
    
    name_dict=corpora.Dictionary(appnametext)
    once_ids=[wordid for wordid,docfreq in name_dict.dfs.items() if docfreq==1]
    name_dict.filter_tokens(once_ids)
    name_dict.save_as_text(dictfile)
    print("Obtain Name dictionary done!")


    # corpus_tf=[name_dict.doc2bow(eachappname) for eachappname in appnametext]
    # #tfidf transform
    # tfidf=models.TfidfModel(corpus_tf)
    # corpus_tfidf=tfidf[corpus_tf]

    KList=[50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]

    # documents = []
    # for i in range(len(appnametext)):
    #     string = "SENT_" + str(i)
    #     sentence = models.doc2vec.LabeledSentence(appnametext[i], labels = [string])
    #     documents.append(sentence)
    file='doc2vec/apptext.txt'
    apptext=open(file,'w')
    for line in appnametext:
        # print line
        apptext.write(' '.join(line)+'\n')
    apptext.close()
    for k in KList:
        #lsi transform
        # lsi=models.LsiModel(corpus=corpus_tfidf,id2word=name_dict,num_topics=k)
        documents=models.doc2vec.TaggedLineDocument(file)
        doc2vec=models.Doc2Vec(documents,size=k,window=2,min_count=0,workers=4)
        featurefile=representationfile+str(k)+".txt"
        fout=open(featurefile,'w')
        for i in range(15282):
            # print type()
            valueList=doc2vec.docvecs[i].tolist()
            for j in range(k):

                fout.write(str(valueList[j])+'\t')
            fout.write('\n')
        fout.close()


def descriptionDoc2vecrepresentation(descriptionFile,representationfile):

    # nameFile='ADMM/alphaNone/appNameText.txt'
    appdescriptiontext=getappFeatureTest(descriptionFile)
    KList=[50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
    file='doc2vec/appdescriptiontext_tmp.txt'
    apptext=open(file,'w')
    for line in appdescriptiontext:
        # print line
        apptext.write(' '.join(line)+'\n')
    apptext.close()
    for k in KList:
        #lsi transform
        # lsi=models.LsiModel(corpus=corpus_tfidf,id2word=name_dict,num_topics=k)
        print k
        documents=models.doc2vec.TaggedLineDocument(file)
        doc2vec=models.Doc2Vec(documents,size=k,window=2,min_count=1,workers=4)
        featurefile=representationfile+str(k)+".txt"
        fout=open(featurefile,'w')
        for i in range(15282):
            # print type()
            valueList=doc2vec.docvecs[i].tolist()
            for j in range(k):

                fout.write(str(valueList[j])+'\t')
            fout.write('\n')
        fout.close()

def getappFeatureTest(featureFile):
    appdescriptiontext=[]
    fin=open(featureFile,'r')
    for line in fin:
        appdescriptiontext.append(eval(line.strip()))
    return appdescriptiontext
    
if __name__=="__main__":
    corpusfile='Corpus_small.txt'
    nameDict='doc2vec/nameDict.txt'
    nameRepresentation='doc2vec/name/nameVector'
    descriptionRepresentation='doc2vec/desc/descVector'
    descriptionFile='doc2vec/appDescriptionText_small.txt'
    # nameDoc2vec_representation(corpusfile,nameDict,nameRepresentation)
    descriptionDoc2vecrepresentation(descriptionFile,descriptionRepresentation)


