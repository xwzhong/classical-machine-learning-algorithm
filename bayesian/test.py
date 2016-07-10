#coding: utf-8
#date: 2016-07-10
#mail: artorius.mailbox@qq.com
#author: xinwangzhong -version 0.1

from numpy import *
from nativeBayesian import *
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#1 is abusive, 0 not
    return postingList,classVec
                  
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)
 
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec
 
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
 
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
 
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
     
# def spamTest():
#     docList=[]; classList = []; fullText =[]
#     for i in range(1,26):
#         wordList = textParse(open('email/spam/%d.txt' % i).read())
#         docList.append(wordList)
#         fullText.extend(wordList)
#         classList.append(1)
#         wordList = textParse(open('email/ham/%d.txt' % i).read())
#         docList.append(wordList)
#         fullText.extend(wordList)
#         classList.append(0)
#     vocabList = createVocabList(docList)#create vocabulary
#     trainingSet = range(50); testSet=[]           #create test set
#     for i in range(10):
#         randIndex = int(random.uniform(0,len(trainingSet)))
#         testSet.append(trainingSet[randIndex])
#         del(trainingSet[randIndex])  
#     trainMat=[]; trainClasses = []
#     for docIndex in trainingSet:#train the classifier (get probs) trainNB0
#         trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
#         trainClasses.append(classList[docIndex])
#     p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
#     errorCount = 0
#     for docIndex in testSet:        #classify the remaining items
#         wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
#         if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
#             errorCount += 1
#             print ("classification error",docList[docIndex])
#     print ('the error rate is: ',float(errorCount)/len(testSet))
#     #return vocabList,fullText
     
if __name__ == "__main__":
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print (myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat, listClasses)
    testingNB()
    # spamTest()
