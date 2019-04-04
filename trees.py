# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 20:09:11 2016

@file trees.py
@brief 决策树算法实现 实现西瓜案例 改进
在上一个tree.py版本中无法对连续属性进行处理，西瓜案例中的密度与含糖度两个属性是连续数据，那该如何处理呢
@version V1.1
"""

"""
@brief 计算给定数据集的信息熵
@param dataSet 数据集
@return 香农熵
"""

import operator
import copy
from math import log
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)#求取数据集的行数
    labelCounts = {}
    for featVec in dataSet:#读取数据集中的一行数据
        currentLabel = featVec[-1] #取featVec中最后一列的值
        #以一行数据中的最后一列值为键值进行统计
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries#将每一类求取概率
        shannonEnt -= prob * log(prob,2)#求取数据集的信息熵
    return shannonEnt

"""
@brief 划分数据集 按照给定的特征划分数据集
@param[in] dataSet 待划分的数据集
@param[in] axis  划分数据集的特征
@param[in] value 需要返回的特征的值
@return retDataSet 返回划分后的数据集
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []#返回的划分后的数据集
    for featVec in dataSet:
        #抽取符合划分特征的值
        if featVec[axis] == value:
            #如果符合此特征值 则存储，存储划分后的数据集时 不需要存储选为划分的特征
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1 :])
            retDataSet.append(reducedFeatVec)
    return retDataSet

"""
@brief 与上述函数类似，区别在于上述函数是用来处理离散特征值而这里是处理连续特征值 
对连续变量划分数据集，direction规定划分的方向， 
决定是划分出小于value的数据样本还是大于value的数据样本集
"""
def splitContinuousDataSet(dataSet,axis,value,direction):
    retDataSet=[]
    for featVec in dataSet:
        if direction==0:
            if featVec[axis]>value:
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        else:
            if featVec[axis]<=value:
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet

"""
@brief 针对离散属性 遍历整个数据集，循环计算香农熵和选择划分函数，找到最好的划分方式。
对于连续属性需要做处理
决策树算法中比较核心的地方，究竟是用何种方式来决定最佳划分？ 
使用信息增益作为划分标准的决策树称为ID3 
使用信息增益率比作为划分标准的决策树称为C4.5 
本程序为信息增益的ID3树 
从输入的训练样本集中，计算划分之前的熵，找到当前有多少个特征，遍历每一个特征计算信息增益，找到这些特征中能带来信息增益最大的那一个特征。 
这里用分了两种情况，离散属性和连续属性 
1、离散属性，在遍历特征时，遍历训练样本中该特征所出现过的所有离散值，假设有n种取值，那么对这n种我们分别计算每一种的熵，最后将这些熵加起来 
就是划分之后的信息熵 
2、连续属性，对于连续值就稍微麻烦一点，首先需要确定划分点，用二分的方法确定（连续值取值数-1）个切分点。遍历每种切分情况，对于每种切分， 
计算新的信息熵，从而计算增益，找到最大的增益。 
假设从所有离散和连续属性中已经找到了能带来最大增益的属性划分，这个时候是离散属性很好办，直接用原有训练集中的属性值作为划分的值就行，但是连续 
属性我们只是得到了一个切分点，这是不够的，我们还需要对数据进行二值处理。
@param[in] dataSet 整个特征集 待选择的集
@return bestFeature 划分数据集最好的划分特征列的索引值
"""
def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        # 对连续型特征进行处理 ,i代表第i个特征,featList是每次选取一个特征之后这个特征的所有样本对应的数据
        featList = [example[i] for example in dataSet]
        #因为特征分为连续值和离散值特征，对这两种特征需要分开进行处理。
        #if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':
        if isinstance(featList[0],float) == True or isinstance(featList[0],int) == True:
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)
            bestSplitEntropy = 10000
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点  找到最大信息熵的划分
            for value in splitList:
                newEntropy = 0.0
                #根据value将属性集分为两个部分
                subDataSet0 = splitContinuousDataSet(dataSet, i, value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)

                prob0 = len(subDataSet0) / float(len(dataSet))
                newEntropy += prob0 * calcShannonEnt(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newEntropy += prob1 * calcShannonEnt(subDataSet1)
                if newEntropy < bestSplitEntropy:
                    bestSplitEntropy = newEntropy
                    bestSplit = value
            # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = bestSplit
            infoGain = baseEntropy - bestSplitEntropy

        # 对离散型特征进行处理
        else:
            uniqueVals = set(featList)              #set 挑选属性中的每一个特征
            newEntropy = 0.0
            # 计算该特征下每种划分的信息熵,选取第i个特征的值为value的子集
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue,例如将密度变为密度<=0.3815
    #将属性变了之后，之前的那些float型的值也要相应变为0和1
    if type(dataSet[0][bestFeature]).__name__=='float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(len(dataSet)):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
    return bestFeature

"""
@brief 计算一个特征数据列表中 出现次数最多的特征值以及次数
@param[in] 特征值列表
@return 返回次数最多的特征值
例如：[1,1,0,1,1]数据列表 返回 1
0"""
def majorityCnt(classList):
    classCount = {}
    #统计数据列表中每个特征值出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #根据出现的次数进行排序 key=operator.itemgetter(1) 意思是按照次数进行排序
    #classCount.items() 转换为数据字典 进行排序 reverse = True 表示由大到小排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
    #返回次数最多的一项的特征值
    return sortedClassCount[0][0]

"""
@brief 主程序，递归产生决策树。 
params: 
dataSet:用于构建树的数据集,最开始就是data_full，然后随着划分的进行越来越小，第一次划分之前是17个瓜的数据在根节点，然后选择第一个bestFeat是纹理 
纹理的取值有清晰、模糊、稍糊三种，将瓜分成了清晰（9个），稍糊（5个），模糊（3个）,这个时候应该将划分的类别减少1以便于下次划分 
labels：还剩下的用于划分的类别 
data_full：全部的数据 
label_full:全部的类别 

既然是递归的构造树，当然就需要终止条件，终止条件有三个： 
1、当前节点包含的样本全部属于同一类别；-----------------注释1就是这种情形 
2、当前属性集为空，即所有可以用来划分的属性全部用完了，这个时候当前节点还存在不同的类别没有分开，这个时候我们需要将当前节点作为叶子节点， 
同时根据此时剩下的样本中的多数类（无论几类取数量最多的类）-------------------------注释2就是这种情形 
3、当前节点所包含的样本集合为空。比如在某个节点，我们还有10个西瓜，用大小作为特征来划分，分为大中小三类，10个西瓜8大2小，因为训练集生成 
树的时候不包含大小为中的样本，那么划分出来的决策树在碰到大小为中的西瓜（视为未登录的样本）就会将父节点的8大2小作为先验同时将该中西瓜的 
大小属性视作大来处理。 
构
"""
def createTree(dataSet,labels,data_full,labels_F):
    #注意label和labels_full可能是同一参数  这样就会导致删除了labels_full
    #因此在此处使用深拷贝 解决此类问题
    labels_full = copy.deepcopy(labels_F)
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):  #注释1
        return classList[0]
    if len(dataSet[0])==1:                             #注释2
        return majorityCnt(classList)
    #平凡情况，每次找到最佳划分的特征
    bestFeat=chooseBestFeatureToSplit(dataSet,labels)
    bestFeatLabel=labels[bestFeat]

    myTree={bestFeatLabel:{}}
    featValues=[example[bestFeat] for example in dataSet]
    ''''' 
    刚开始很奇怪为什么要加一个uniqueValFull，后来思考下觉得应该是在某次划分，比如在根节点划分纹理的时候，将数据分成了清晰、模糊、稍糊三块 
    ，假设之后在模糊这一子数据集中，下一划分属性是触感，而这个数据集中只有软粘属性的西瓜，这样建立的决策树在当前节点划分时就只有软粘这一属性了， 
    事实上训练样本中还有硬滑这一属性，这样就造成了树的缺失，因此用到uniqueValFull之后就能将训练样本中有的属性值都囊括。 
    如果在某个分支每找到一个属性，就在其中去掉一个，最后如果还有剩余的根据父节点投票决定。 
    但是即便这样，如果训练集中没有出现触感属性值为“一般”的西瓜，但是分类时候遇到这样的测试样本，那么应该用父节点的多数类作为预测结果输出。 
    '''
    uniqueVals=set(featValues)
    if type(dataSet[0][bestFeat]).__name__=='str':
       # currentlabel=labels_full.index(labels[bestFeat])
        #找到此标签在原始标签中的索引
        currentlabel=labels_full.index(bestFeatLabel)
        featValuesFull=[example[currentlabel] for example in data_full]
        uniqueValsFull=set(featValuesFull)
    del(labels[bestFeat])
    ''''' 
    针对bestFeat的每个取值，划分出一个子树。对于纹理，树应该是{"纹理"：{？}}，显然？处是纹理的不同取值，有清晰模糊和稍糊三种，对于每一种情况， 
    都去建立一个自己的树，大概长这样{"纹理"：{"模糊"：{0},"稍糊"：{1},"清晰":{2}}}，对于0\1\2这三棵树，每次建树的训练样本都是值为value特征数减少1 
    的子集。 
    '''
    for value in uniqueVals:
        subLabels = labels[:]
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            uniqueValsFull.remove(value)
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, data_full, labels_full)
    #完成对缺失值的处理
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value] = majorityCnt(classList)
    return myTree

"""
@brief 对未知特征在创建的决策树上进行分类
@param[in] inputTree
@param[in] featLabels
@param[in] testVec
@return classLabel 返回识别的结果
"""
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key :
            if isinstance(secondDict[key],dict) == True:
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

"""
@brief 存储构建的决策树
"""
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

"""
@brief 读取文本存储的决策树
"""
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
