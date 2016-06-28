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




def nameLDArepresentation(textfile,dictfile,representationfile):
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
    

    KList=[50,100,200,300,400,500,600,700,800,900,1000]
    for k in KList:
        #lda transform
        lda=models.LdaModel(corpus=corpus_tf,id2word=name_dict,num_topics=k,minimum_probability=0)
        # print 'lda:',type(lda)
        # lda.save()
        corpus_lda=lda[corpus_tf]
        print("Obtain Name lsi representation done!")
        featurefile=representationfile+str(k)+".txt"
        fout=open(featurefile,'w')
        for doc in corpus_lda:
            line=[]
            for i in range(k):
                line.append(0)
            for (fid,fvalue) in doc:
                line[fid]=fvalue
            for item in line:
                t=fout.write(str(item)+'\t')
            t=fout.write('\r\n')
        fout.close()






def descriptionLDArepresentation(textfile,dictfile,representationfile):
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
    

    KList=[600,700,800,900,1000]
    for k in KList:
        #lda transform
        lda=models.LdaModel(corpus=corpus_tf,id2word=description_dict,num_topics=k,minimum_probability=0)
        corpus_lda=lda[corpus_tf]
        print("Obtain Description lsi representation done!")
        featurefile=representationfile+str(k)+".txt"
        fout=open(featurefile,'w')
        for doc in corpus_lda:
            line=[]
            for i in range(k):
                line.append(0)
            for (fid,fvalue) in doc:
                line[fid]=fvalue
            for item in line:
                t=fout.write(str(item)+'\t')
            t=fout.write('\r\n')
        fout.close()





def categoryOneHotrepresentation(textfile,dictfile,representationfile):
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
    #tfidf transform
    #tfidf=models.TfidfModel(corpus_tf)
    #corpus_tfidf=tfidf[corpus_tf]
    #lsi transform
    #lsi=models.LsiModel(corpus=corpus_tfidf,id2word=description_dict,num_topics=k)
    #corpus_lsi=lsi[corpus_tfidf]
    print("Obtain Category One-Shot representation done!")

    fout=open(representationfile,'w')
    dictnum=len(category_dict)
    for doc in corpus_tf:
        line=[]
        for i in range(dictnum):
            line.append(0)
        for (fid,fvalue) in doc:
            line[fid]=fvalue
        for item in line:
            fout.write(str(item)+'\t')
        fout.write('\r\n')
    fout.close()


    
    
if __name__=="__main__":
    corpusfile='Corpus_small.txt'
    nameDict='LDA/nameDict.txt'
    nameRepresentation='LDA/name/nameVector'
    descDict='LDA/descDict.txt'
    descRepresentation='LDA/desc/descVector'
    # cateDict='LDA/cateDict.txt'
    # cateRepresentation='LDA/cate/cateVector'
    # nameLDArepresentation(corpusfile,nameDict,nameRepresentation)
    descriptionLDArepresentation(corpusfile,descDict,descRepresentation)
    # categoryOneHotrepresentation(corpusfile,cateDict,cateRepresentation)


