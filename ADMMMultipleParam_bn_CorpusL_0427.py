import numpy
import codecs
import json
import scipy
import re
from scipy.sparse import csc_matrix
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import gensim
from gensim import models,corpora,similarities
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import datetime

def LoadXY(X,Y):
    #sparse matrix X muliplies sparse matrix Y
    Z=X.transpose()*Y
    return Z


import datetime
def ComputeObjectiveFunction(A,B,U,V,W1,W2,X1,X2,lamb,gam1,gam2,gam3,gam4,gam5,gam6):
    #Sparse Matrix: A and B
    #dense Matrix: U, V1, V2, W
    f=0
    #a1=numpy.linalg.norm(A.todense()-U*V.T,'fro')
    starttime=datetime.datetime.now()
    C=A*A.transpose()
    a1=sum(C.diagonal())
    C=V.T*A.transpose()*U
    a1=a1-2*numpy.trace(C)
    C=(V.T*V)*(U.T*U)
    a1=a1+numpy.trace(C)
    endtime=datetime.datetime.now()
    interval=(endtime-starttime).seconds
    print("Computing the first loss: "+str(interval))
    #a2=numpy.linalg.norm(B.todense()-W1*X1.T-W2*X2.T,'fro')

    starttime=datetime.datetime.now()
    a3=numpy.linalg.norm(U,'fro')
    a4=numpy.linalg.norm(V,'fro')
    a5=numpy.linalg.norm(W1,'fro')
    a6=numpy.linalg.norm(W2,'fro')
    a7=numpy.linalg.norm(X1,'fro')
    a8=numpy.linalg.norm(X2,'fro')
    endtime=datetime.datetime.now()
    interval=(endtime-starttime).seconds
    print("Computing the regularization terms: "+str(interval))

    starttime=datetime.datetime.now()
    W=numpy.mat(numpy.hstack((W1.A,W2.A)))
    del W1
    del W2
    X=numpy.mat(numpy.hstack((X1.A,X2.A)))
    del X1
    del X2
    C=B*B.transpose()
    a2=sum(C.diagonal())
    C=X.T*B.transpose()*W
    a2=a2-2*numpy.trace(C)
    C=X.T*X*(W.T*W)
    a2=a2+numpy.trace(C)
    endtime=datetime.datetime.now()
    interval=(endtime-starttime).seconds
    print("Computing the second loss: "+str(interval))


    del C,A,B,U,V,W,X
    f=lamb*a1+(1-lamb)*a2+gam1*a3*a3+gam2*a4*a4+gam3*a5*a5+gam4*a6*a6+gam5*a7*a7+gam6*a8*a8
    #
    return f

def AlternatingDirectionMethodofMultipliers(E,L,kE,kL,eta,eps,maxiter,strlamb,Ufile,Wfile):
    #

    starttime=datetime.datetime.now()
    le=float(strlamb)
    gam1=1.0
    gam2=1.0
    gam3=1.0
    gam4=1.0
    gam5=1.0
    gam6=1.0
    imaxiter=int(maxiter)
    feta=float(eta)
    ikE=int(kE)
    ikL=int(kL)
    Asize=E.shape
    Bsize=L.shape
    l=Asize[0]
    m=Asize[1]
    r=Bsize[0]
    print(str(l)+"-"+str(m)+"-"+str(r))
    #a1tmp=numpy.linalg.norm(E.todense(),'fro')
    a1tmp=scipy.sparse.linalg.norm(E,'fro')
    #a2tmp=numpy.linalg.norm(L.todense(),'fro')
    a2tmp=scipy.sparse.linalg.norm(L,'fro')
    endtime=datetime.datetime.now()
    interval=(endtime-starttime).seconds
    print("initialization 1 : "+str(interval)+"s")

    starttime=datetime.datetime.now()
    U=numpy.mat(numpy.zeros([l,ikE]))#=numpy.mat(numpy.ones([l,ik]))/(1.0*l*ik)#numpy.mat(numpy.zeros([l,ik]))#numpy.mat(numpy.random.rand(l,ik))#
    V=numpy.mat(numpy.random.rand(m,ikE))#numpy.mat(numpy.ones([m,ik])/(1.0*m*ik))#numpy.mat(numpy.zeros([m,ik]))#numpy.mat(numpy.random.rand(m,ik))#
    X1=numpy.mat(numpy.random.rand(m,ikE))#numpy.mat(numpy.ones([m,ik])/(1.0*m*ik))#numpy.mat(numpy.zeros([m,ik]))#numpy.mat(numpy.random.rand(m,ik))#
    X2=numpy.mat(numpy.random.rand(m,ikL-ikE))
    W1=numpy.mat(numpy.zeros([r,ikE]))#numpy.mat(numpy.ones([r,ik])/(1.0*r*ik))#numpy.mat(numpy.zeros([r,ik]))#numpy.mat(numpy.random.rand(r,ik))#
    W2=numpy.mat(numpy.zeros([r,ikL-ikE]))
    IkE=numpy.mat(numpy.eye(ikE,ikE))
    #IkL=numpy.mat(numpy.eye(ikL,ikL))
    IkLE=numpy.mat(numpy.eye(ikL-ikE,ikL-ikE))
    Lamb=numpy.mat(numpy.zeros([m,ikE]))
    fval=100
    feps=100
    falpha=0.0002*max(a1tmp,a2tmp)*float(max(l,m,r))/float(max(ikE,ikL))#1.0*0.01*max(508.49786,100.1467)*float(max(l,m,n))/float(ik)
    endtime=datetime.datetime.now()
    interval=(endtime-starttime).seconds
    print("initialization 2:"+str(interval)+"s")

    t=0
    while t<imaxiter and feps>float(eps) and fval>float(eps):
        #(1)update U
        starttime=datetime.datetime.now()
        tmp=le*V.T*V+gam1*IkE
        U=le*E*V*tmp.I
        endtime=datetime.datetime.now()
        interval=(endtime-starttime).seconds
        print("updating U: "+str(interval))

        #(2)update V
        starttime=datetime.datetime.now()
        tmp=2*le*U.T*U+2*(falpha+gam2)*IkE
        V=(2*le*E.transpose()*U+2*falpha*X1-Lamb)*tmp.I
        endtime=datetime.datetime.now()
        interval=(endtime-starttime).seconds
        print("updating V"+str(interval))


        #(3)update W1
        starttime=datetime.datetime.now()
        tmp=(1-le)*X1.T*X1+gam3*IkE
        #W1=(1-le)*(L*X1-W2*(X2.T*X1))*tmp.I
        W1=(1-le)*(L*X1-W2*(X2.T*X1))*tmp.I
        endtime=datetime.datetime.now()
        interval=(endtime-starttime).seconds
        print("updating W1:"+str(interval))

        #(4)update W2
        starttime=datetime.datetime.now()
        tmp=(1-le)*X2.T*X2+gam4*IkLE
        W2=(1-le)*(L*X2-W1*(X1.T*X2))*tmp.I
        endtime=datetime.datetime.now()
        interval=(endtime-starttime).seconds
        print("Updating W2:"+str(interval))

        #(5)update X1
        starttime=datetime.datetime.now()
        tmp=2*(1-le)*W1.T*W1+2*(gam5+falpha)*IkE
        X1=(2*(1-le)*L.transpose()*W1-2*(1-le)*X2*(W2.T*W1)+2*falpha*V+Lamb)*tmp.I
        endtime=datetime.datetime.now()
        interval=(endtime-starttime).seconds
        print("Updating X1:"+str(interval))

        #(6)update X2
        starttime=datetime.datetime.now()
        tmp=(1-le)*W2.T*W2+gam6*IkLE
        X2=(1-le)*(L.transpose()*W2-X1*(W1.T*W2))*tmp.I
        endtime=datetime.datetime.now()
        interval=(endtime-starttime).seconds
        print("Updating X2:"+str(interval))
        del tmp

        #(7)update Lagrangian Multiplier Lambda
        Lamb=Lamb+feta*2*falpha*(V-X1)
        fold=fval
        starttime=datetime.datetime.now()
        fval=ComputeObjectiveFunction(E,L,U,V,W1,W2,X1,X2,le,gam1,gam2,gam3,gam4,gam5,gam6)
        endtime=datetime.datetime.now()
        interval=(endtime-starttime).seconds
        print("Computing objective function: "+str(interval))

        feps=abs(fold-fval)/max(1,abs(fold))
        print("iteration "+str(t)+": f="+str(fval)+", eps="+str(feps))
        t=t+1

    numpy.savetxt(Ufile,U,newline='\r\n')
    numpy.savetxt(Wfile,numpy.mat(numpy.hstack((W1.A,W2.A))),newline='\r\n')


def getappFeatureTest(featureFile):
    appdescriptiontext=[]
    fin=open(featureFile,'r')
    for line in fin:
        appdescriptiontext.append(eval(line.strip()))
    return appdescriptiontext


def getDescriptionDict(appdescriptiontext):

    description_dict=corpora.Dictionary(appdescriptiontext)
    once_ids=[wordid for wordid,docfreq in description_dict.dfs.items() if docfreq==1]
    description_dict.filter_tokens(once_ids)
    print 'ok'
    return description_dict

def getDescriptionText(textfile,outfile):
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
        # print wordlist_rmLothers
        appdescriptiontext.append(wordlist_rmLothers)
        if count%1000==0:
            print count
        count=count+1
    fout=open(outfile,'w')
    for item in appdescriptiontext:
        fout.write(' '.join(item)+'\n')
    fout.close()

def getCorpusL():
    fin=open('appDescrition_text.txt','r')
    appdescriptiontext=[]
    for line in fin:
        wordsList=line.strip().split()
        appdescriptiontext.append(wordsList)

    description_dict=corpora.Dictionary(appdescriptiontext)
    once_ids=[wordid for wordid,docfreq in description_dict.dfs.items() if docfreq==1]
    description_dict.filter_tokens(once_ids)
    # description_dict.save_as_text(dictfile)


    print("Obtain Description dictionary done!")
    print 'description_dict,L:',len(description_dict)
    corpus_tf=[description_dict.doc2bow(eachappdescription) for eachappdescription in appdescriptiontext]
    print 'Lcorpus:',len(corpus_tf)
    tfidf=models.TfidfModel(corpus_tf)
    corpus_tfidf=tfidf[corpus_tf]
    print 'tfidf,L:',len(corpus_tfidf[0])
    matatmp=gensim.matutils.corpus2csc(corpus_tfidf,num_terms=30360)
    print 'L,mat:',matatmp.shape
    matatmp=matatmp.transpose()
    # print matatmp.ndim
    return matatmp,description_dict
#docnum: 15282


def getCorpusE(textfile,description_dict):
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

    # name_dict=corpora.Dictionary(appnametext)
    # once_ids=[wordid for wordid,docfreq in name_dict.dfs.items() if docfreq==1]
    # name_dict.filter_tokens(once_ids)
    # name_dict.save_as_text(dictfile)
    print("Obtain Name dictionary done!")
    print 'description_dict,E:',len(description_dict)
    corpus_tf=[description_dict.doc2bow(eachappname) for eachappname in appnametext]
    print 'Ecorpus:',len(corpus_tf)
    tfidf=models.TfidfModel(corpus_tf)
    corpus_tfidf=tfidf[corpus_tf]
    print 'tfidf,E',len(corpus_tfidf[0])
    # print corpus_tfidf[]
    matatmp=gensim.matutils.corpus2csc(corpus_tfidf,num_terms=30360)
    print 'E,mat:',matatmp.shape
    matatmp=matatmp.transpose()
    return matatmp


# import getCorpusL
def run_bn():

    textfile='Corpus_small.txt'

    L,description_dict=getCorpusL()
    print 'L:',L.shape
    print 'L.....'
    pathf="ADMM/lowerWords/feature_final/"
    E=getCorpusE(textfile,description_dict)
    print 'E:',E.shape
    print("E.....")
    print("Loading done !")
    eps=1e-6
    maxiter=1000
    # ks=[1000]
    kE=[2000,3000]
    kL=[4000]
    # ks=[20,30,70,150]
    # alphas=[0]#[0.1]#,1,10,100,1000]#,10000]
    lambdas=[0.5]
    etas=[1.618]#,1.5,1,0.5,0.1]#0.1
    it=0;
    for k1 in kE:
        for k2 in kL:
            for lam in lambdas:
                for eta in etas:
                    umatrixfile=os.path.join(pathf,"pKE"+str(k1)+"pKL"+str(k2)+"pL"+str(lam)+"-Umatrix.txt")
                    wmatrixfile=os.path.join(pathf,"pKE"+str(k1)+"pKL"+str(k2)+"pL"+str(lam)+"-Wmatrix.txt")
                    AlternatingDirectionMethodofMultipliers(E,L,k1,k2,eta,eps,maxiter,lam,umatrixfile,wmatrixfile)
                    print("Optimizing done!")
                    print("iteration"+str(it)+": "+str(k1)+" "+str(k2)+" "+str(lam))
                    it=it+1



def run():
    textfile='Corpus_small.txt'
    outfile='appDescrition_text.txt'
    getDescriptionText(textfile,outfile)

if __name__=="__main__":
    run_bn()
    # run()

