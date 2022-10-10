# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:39:57 2022

@author: rupak
"""

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
#The code above simply processes the text with Python. 

reload(kNN)
# reloaded the kNN.py module (name of my Python file). When you modify a module, you must reload that module or you will always use the old version.
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')

datingDataMat

 datingLabels[0:20]
 
 #normalize
 newValue = (oldValue-min)/(max-min)
 
 def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

reload(kNN)
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
normMat

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d"\
        % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
            print "the total error rate is: %f" % (errorCount/float(numTestVecs))
            
 kNN.datingClassTest()
 
 #putting all together
 def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(raw_input(\"percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-\minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ",\resultList[classifierResult - 1]
kNN.classifyPerson()]