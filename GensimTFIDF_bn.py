import json
import codecs
import os
import sys
import re
import numpy
import gensim
from gensim import models,corpora,similarities
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def writecorpustofile(corpus,filename):
    mata=gensim.matutils.corpus2csc(corpus).transpose()
    (docnum,wordnum)=mata.shape
    fout=open(filename,'w')
    for i in range(docnum):
        tmp=mata.getrow(i)
        if tmp.nnz>0:
            L=tmp.nonzero()
            for icol in L[1]:
                a=tmp.getcol(icol).toarray()
                fout.write(str(icol)+":"+str(a[0][0])+' ')
            fout.write('\r\n')
        else:
            fout.write('0:0 \r\n')
    fout.close()
            

def nameTFIDF(textfile,dictfile,namefile):
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
        for i in range(len(wordlist_rmstopword)):
            wordlist_rmstopword[i]=nltk.PorterStemmer().stem_word(wordlist_rmstopword[i])
        Lnumber=[word for word in wordlist_rmstopword if re.match(r'\d+$',word)]
        Lothers=[word for word in wordlist_rmstopword if re.match(r'^_+',word)]
        wordlist_rmLnumber=[word for word in wordlist_rmstopword if word not in Lnumber]
        wordlist_rmLothers=[word for word in wordlist_rmLnumber if word not in Lothers]
        appnametext.append(wordlist_rmLothers)
        if count%1000==0:
            print count
        count=count+1     
    # appname.clear()
    print("Obtain Name Wordlist done!")
    
    name_dict=corpora.Dictionary(appnametext)
    once_ids=[wordid for wordid,docfreq in name_dict.dfs.items() if docfreq==1]
    name_dict.filter_tokens(once_ids)
    name_dict.save_as_text(dictfile)
    print("Obtain Name dictionary done!")

    corpus_tf=[name_dict.doc2bow(eachappname) for eachappname in appnametext]
    #tfidf transform
    tfidf=models.TfidfModel(corpus_tf)
    corpus_tfidf=tfidf[corpus_tf]
    writecorpustofile(corpus_tfidf,namefile)






def descriptionTFIDF(textfile,dictfile,descfile):
    appdescription=[]
    htmlcompilers=re.compile(r'<[^>]+>| +|=+|\?+|!+|-+|\*+|\.+|(&gt;)+|\(+|\)+|\^+|\_+|#+|\[+|\,+|(&amp;)+|\/+|\]+|:+|(&39;t)+|(&quot;)+|(&#39;)+|~+',re.S)
    spacecompilers=re.compile(r'\s+',re.S)
    with open(textfile,'r') as fin:
        for line in fin:
            line=json.loads(line)
            tmp=htmlcompilers.sub(' ',line['description'])
            tmp=spacecompilers.sub(' ',tmp)
            appdescription.append(tmp)
    print("Preprocessing description Text done!")

    tokenizer=RegexpTokenizer(r'\w+')
    appdescriptiontext=[]
    count=0
    for descriptionstr in appdescription:
        wordlist=tokenizer.tokenize(descriptionstr)
        wordlist_rmstopword=[word for word in wordlist if word not in stopwords.words('english')]
        for i in range(len(wordlist_rmstopword)):
            wordlist_rmstopword[i]=nltk.PorterStemmer().stem_word(wordlist_rmstopword[i])
        Lnumber=[word for word in wordlist_rmstopword if re.match(r'\d+$',word)]
        Lothers=[word for word in wordlist_rmstopword if re.match(r'^_+',word)]
        wordlist_rmLnumber=[word for word in wordlist_rmstopword if word not in Lnumber]
        wordlist_rmLothers=[word for word in wordlist_rmLnumber if word not in Lothers]
        appdescriptiontext.append(wordlist_rmLothers)
        if count%1000==0:
            print count
        count=count+1        
    # appdescription.clear()
    print("Obtain Description Wordlist done!")
    
    description_dict=corpora.Dictionary(appdescriptiontext)
    once_ids=[wordid for wordid,docfreq in description_dict.dfs.items() if docfreq==1]
    description_dict.filter_tokens(once_ids)
    description_dict.save_as_text(dictfile)
    print("Obtain Description dictionary done!")

    corpus_tf=[description_dict.doc2bow(eachappdescription) for eachappdescription in appdescriptiontext]
    #tfidf transform
    tfidf=models.TfidfModel(corpus_tf)
    corpus_tfidf=tfidf[corpus_tf]
    writecorpustofile(corpus_tfidf,descfile)





def categoryOneHot(textfile,dictfile,catfile):
    appcategory=[]
    with open(textfile,'r') as fin:
        for line in fin:
            line=json.loads(line)
            appcategory.append(line['category'])
    print("Preprocessing Category Text done!")

    appcategorytext=[]
    for catstr in appcategory:
        wordlist=[]
        wordlist.append(catstr)
        appcategorytext.append(wordlist)
    # appcategory.clear()
    print("Obtain Category Wordlist done!")
    
    category_dict=corpora.Dictionary(appcategorytext)
    category_dict.save_as_text(dictfile)
    print("Obtain Category dictionary done!")

    corpus_tf=[category_dict.doc2bow(eachappcategory) for eachappcategory in appcategorytext]
    print("Obtain Category One-Shot representation done!")
    writecorpustofile(corpus_tf,catfile)

    
    
if __name__=="__main__":
    corpusfile='Corpus_small.txt'
    nameDict='TFIDF/nameDict.txt'
    nameRepresentation='TFIDF/name/nameVector'
    descDict='TFIDF/descDict.txt'
    descRepresentation='TFIDF/desc/descVector'
    # cateDict='TFIDF/cateDict.txt'
    # cateRepresentation='TFIDF/cate/cateVector'
    # nameTFIDF(corpusfile,nameDict,nameRepresentation)
    descriptionTFIDF(corpusfile,descDict,descRepresentation)
    # categoryOneHot(corpusfile,cateDict,cateRepresentation)


