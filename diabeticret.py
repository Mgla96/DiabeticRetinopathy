#!/usr/bin/python3
import numpy as np
import random

"""
w/ decision tree we want to choose features based on greatest information gain (smallest entropy)

p_dot = P/(P+N)
Entropy: Imp(p_dot) = -p_dot*log_2(p_dot) - (1-p_dot)log_2(1-p_dot)
Combine entropy of branches with weights for total entropy after split 
Lowest Entropy & lower than before split = Greatest information gain
We want the feature with greatest information gain
"""

class decisionTree(object):
    def __init__(self):
        self.feature = None
        self.entropy = None
        self.ent1 = None
        self.ent2 = None
        self.left = None
        self.right = None
        self.parent = None
        self.range = None
        self.isLeaf = False
        self.leafVal = None

    def get_range(self):
        return self.range

    def setParent(self, feature, entropy, left, right, parent):
        self.parent = decisionTree()
        self.parent.feature = feature
        self.parent.entropy = entropy
        self.parent.left = left
        self.parent.right = right
        self.parent.parent = None

    def setChildren(self, left, right):
        self.left = left
        self.right = right

    def getIsLeaf(self):
        return self.isLeaf

    def getLeafVal(self):
        return self.leafVal


def get_parents(tree):
    parents = []
    get_parent_helper(tree, parents)
    return parents


def get_parent_helper(tree, parents):
    if tree.parent is not None:
        parents.append(tree.parent)
        get_parent_helper(tree.parent, parents)
    else:
        return parents


def getEntropyBin(c, label):
    onePos = 0
    oneNeg = 0
    zeroPos = 0
    zeroNeg = 0
    size = len(label)
    for i in range(c.size):
        if(c[i] == 1):
            if(label[i] == 1):
                onePos += 1
            else:
                oneNeg += 1
        else:  # c[i]=0
            if(label[i] == 1):
                zeroPos += 1
            else:
                zeroNeg += 1
    if(onePos+oneNeg != 0):
        onepdot = onePos / float(onePos + oneNeg)
    else:
        onepdot = 0
    if(zeroPos+zeroNeg != 0):
        zeropdot = zeroPos / float(zeroPos + zeroNeg)
    else:
        zeropdot = 0
    if(onepdot == 0 or onepdot == 1):
        oneEnt = 0
    else:
        oneEnt = -onepdot*np.log2(onepdot) - (1-onepdot)*np.log2(1-onepdot)
    if(zeropdot == 0 or zeropdot == 1):
        zeroEnt = 0
    else:
        zeroEnt = -zeropdot*np.log2(zeropdot) - \
            (1-zeropdot)*np.log2(1-zeropdot)
    Entropy = oneEnt*(onePos+oneNeg)/size + zeroEnt*(zeroPos+zeroNeg)/size
    return Entropy, 0.5, zeroEnt, oneEnt, zeroPos, zeroNeg, onePos, oneNeg


def getEntropyInt(c, label, idx):
    d = np.sort(c, axis=0, kind="quicksort")
    split = []
    for splitVal in range(len(label)-1):
        split.append((d[splitVal]+d[splitVal+1])/2)
    size = len(label)
    leaf1PosFinal = 0
    leaf1NegFinal = 0
    leaf2PosFinal = 0
    leaf2NegFinal = 0
    smallestEntropy = 1
    ent1Final = 1
    ent2Final = 1
    splitRange = None
    for splitVal in split:  # iterating through trying to find best split val
        leaf1Pos = leaf1Neg = leaf2Pos = leaf2Neg = 0
        # for each split value calculating all the positive and negative labeled values for leaf1 and leaf2
        for i in range(len(label)):
            # int(c[i])  #if i replace with c again so entropy of floats too high and overall score without using float features, score higher
            if c[i] <= splitVal:
                if(label[i] == 1):  # label says true
                    leaf1Pos += 1
                else:
                    leaf1Neg += 1
            else:  # for leaf 2 when greater than the split value
                if label[i] == 1:  # label says true
                    leaf2Pos += 1
                else:
                    leaf2Neg += 1
        if (leaf1Pos+leaf1Neg) != 0:
            pdot1 = leaf1Pos/float(leaf1Pos+leaf1Neg)
        else:
            pdot1 = 0
        if (leaf2Pos+leaf2Neg) != 0:
            pdot2 = leaf2Pos/float(leaf2Pos+leaf2Neg)
        else:
            pdot2 = 0
        if(pdot1 == 0 or pdot1 == 1):
            ent1 = 0
        else:
            ent1 = -pdot1*np.log2(pdot1) - (1-pdot1)*np.log2(1-pdot1)
        if pdot2 == 0 or pdot2 == 1:
            ent2 = 0
        else:
            ent2 = -pdot2*np.log2(pdot2) - (1-pdot2)*np.log2(1-pdot2)
        Entropy = (ent1 * ((leaf1Pos+leaf1Neg)/size)) + \
            (ent2*((leaf2Pos+leaf2Neg)/size))
        if Entropy < smallestEntropy:
            smallestEntropy = Entropy
            ent1Final = ent1
            ent2Final = ent2
            splitRange = splitVal
            leaf1PosFinal = leaf1Pos
            leaf1NegFinal = leaf1Neg
            leaf2PosFinal = leaf2Pos
            leaf2NegFinal = leaf2Neg
    return smallestEntropy, splitRange, ent1Final, ent2Final, leaf1PosFinal, leaf1NegFinal, leaf2PosFinal, leaf2NegFinal


def transformMatrix(c, idx):
    if(idx == 8):
        c = c*2
    if(idx == 9):
        c = c*5
    if(idx == 10):
        c = c*3
    if(idx == 11 or idx == 12):
        c = c*13
    if(idx == 13):
        c = c*13
    if(idx == 14):
        c = c*70
    if(idx == 15):
        c = c*130
    if(idx == 16):
        c = c*3000
    if(idx == 17):
        c = c*3000
    return c


def getEntropyFloat(c, label, idx):
    '''
    sorts c in increasing order and splits inbetween each for the split values to test
    use gain ratio to remove bias for information gain to attributes with a large number of values
    '''
    d = np.sort(c, axis=0, kind="quicksort")
    split = []
    for splitVal in range(len(label)-1):
        split.append((d[splitVal]+d[splitVal+1])/2)
    size = len(label)
    smallestEntropy = 1
    ent1Final = ent2Final = splitRange = None
    leaf1PosFinal = leaf1NegFinal = leaf2PosFinal = leaf2NegFinal = 0
    for splitVal in split:  # iterating through trying to find best split val
        leaf1Pos = leaf1Neg = leaf2Pos = leaf2Neg = 0
        # for each split value calculating all the positive and negative labeled values for leaf1 and leaf2
        for i in range(len(label)):
            if c[i] <= splitVal:
                if label[i] == 1:  # label says true
                    leaf1Pos += 1
                else:
                    leaf1Neg += 1
            else:  # for leaf 2 when greater than the split value
                if label[i] == 1:  # label says true
                    leaf2Pos += 1
                else:
                    leaf2Neg += 1
        if (leaf1Pos+leaf1Neg) != 0:
            pdot1 = leaf1Pos/float(leaf1Pos+leaf1Neg)
        else:
            pdot1 = 0
        if (leaf2Pos+leaf2Neg) != 0:
            pdot2 = leaf2Pos/float(leaf2Pos+leaf2Neg)
        else:
            pdot2 = 0
        if pdot1 == 0 or pdot1 == 1:
            ent1 = 0
        else:
            ent1 = -pdot1*np.log2(pdot1) - (1-pdot1)*np.log2(1-pdot1)
        if pdot2 == 0 or pdot2 == 1:
            ent2 = 0
        else:
            ent2 = -pdot2*np.log2(pdot2) - (1-pdot2)*np.log2(1-pdot2)
        Entropy = (ent1 * ((leaf1Pos+leaf1Neg)/size)) + \
            (ent2*((leaf2Pos+leaf2Neg)/size))
        if Entropy < smallestEntropy:
            smallestEntropy = Entropy
            ent1Final = ent1
            ent2Final = ent2
            splitRange = splitVal
            leaf1PosFinal = leaf1Pos
            leaf1NegFinal = leaf1Neg
            leaf2PosFinal = leaf2Pos
            leaf2NegFinal = leaf2Neg

    return smallestEntropy, splitRange, ent1Final, ent2Final, leaf1PosFinal, leaf1NegFinal, leaf2PosFinal, leaf2NegFinal


def getNewData(train_data, train_labels, bestFeature, rnge, section):
    '''
    to get smaller matrix
    section is 0 or 1  
      if 0 means this is left tree so we are comparing <= range and removing greater than range
      if section 1 means we are comparing greater range and removing <= range
    '''
    c = train_data.T[bestFeature]  # column
    # td = transformMatrix(c,bestFeature) #column of bestfeature all transformed just incase it is float
    removeRows = []
    if(section == 0):
        for i in range(train_labels.size):
            if(c[i] > rnge):
                # because will remove these indexes to only get remaining ones on left
                removeRows.append(i)
    else:  # section1
        # so will go through every row of the training data
        for i in range(train_labels.size):
            if(c[i] <= rnge):
                removeRows.append(i)
    train_data_new = np.delete(train_data, removeRows, 0)
    train_labels_new = np.delete(train_labels, removeRows, 0)
    return train_data_new, train_labels_new


def build(tree, train_data, train_labels, flg, forestfeatures):
    '''
    flg
    -0 for being built off left attribute
    -1 for being built off right attribute
    '''
    #parents = get_parents(tree)
    #parentsList = []
    # for parent in parents:
    #    parentsList.append(parent.feature)
    if(flg == 2):
        Ent1BeforeSplit = 1
        Ent2BeforeSplit = 1
    # if(not parents):
    #    Ent1BeforeSplit = 1
    #    Ent2BeforeSplit = 1
    else:
        Ent1BeforeSplit = tree.parent.ent1
        Ent2BeforeSplit = tree.parent.ent2
    minEntropy = 1
    bestFeature = -1
    leaf1PosFinal = leaf1NegFinal = leaf2PosFinal = leaf2NegFinal = 0
    thernge = 0
    earlyStop = 20  # 4
    ent1Final = ent2Final = 1
    for i in range(train_data[0].size):  # length of a row
        # if(i not in parentsList): #save time because woudn't need feature already used by parent (maybe ignore this and allow more splits but a lot worse runtime
        c = train_data.T[i]  # a column
        if (i <= 1 or i == 18) and i in forestfeatures:
            entro, rnge, leaf1Entropy, leaf2Entropy, leaf1Pos, leaf1Neg, leaf2Pos, leaf2Neg = getEntropyBin(
                c, train_labels)
        elif (i >= 2 and i <= 7) and i in forestfeatures:
            entro, rnge, leaf1Entropy, leaf2Entropy, leaf1Pos, leaf1Neg, leaf2Pos, leaf2Neg = getEntropyInt(
                c, train_labels, i)
        elif i in forestfeatures:
            entro, rnge, leaf1Entropy, leaf2Entropy, leaf1Pos, leaf1Neg, leaf2Pos, leaf2Neg = getEntropyFloat(
                c, train_labels, i)
        else:
            # not in forestfeature list so don't use (random forests implementation)
            continue
        if entro < minEntropy:
            minEntropy = entro
            thernge = rnge
            bestFeature = i
            leaf1PosFinal = leaf1Pos
            leaf1NegFinal = leaf1Neg
            leaf2PosFinal = leaf2Pos
            leaf2NegFinal = leaf2Neg
            ent1Final = leaf1Entropy
            ent2Final = leaf2Entropy

    # left branch so compare with left entropy before split
    if(flg == 0 and minEntropy > Ent1BeforeSplit):
        tree.isLeaf = True
        if(leaf1PosFinal+leaf2PosFinal > leaf1NegFinal + leaf2NegFinal):
            tree.leafVal = 1
        else:
            tree.leafVal = 0
        return
    elif(flg == 1 and minEntropy > Ent2BeforeSplit):
        tree.isLeaf = True
        if(leaf1PosFinal+leaf2PosFinal > leaf1NegFinal + leaf2NegFinal):
            tree.leafVal = 1
        else:
            tree.leafVal = 0
        return
    else:
        tree.feature = bestFeature
        tree.entropy = minEntropy
        tree.ent1 = ent1Final
        tree.ent2 = ent2Final
        tree.range = thernge
    if(leaf1PosFinal > leaf1NegFinal):
        leaf1Prob = 1
    else:
        leaf1Prob = 0
    if(leaf2PosFinal > leaf2NegFinal):
        leaf2Prob = 1
    else:
        leaf2Prob = 0
    if(minEntropy == 0):  # both will be leaves
        #print("both leaf1 and leaf2 leaves entrop 0 early stop")
        tree.left = decisionTree()
        tree.left.parent = tree
        tree.left.isLeaf = True
        tree.left.leafVal = leaf1Prob

        tree.right = decisionTree()
        tree.right.parent = tree
        tree.right.isLeaf = True
        tree.right.leafVal = leaf2Prob
    else:
        if(leaf1PosFinal+leaf1NegFinal < earlyStop or ent1Final == 0):
            if(leaf2PosFinal+leaf2NegFinal < earlyStop or ent2Final == 0):  # both leaves
                #print("leaf1&2 early stop")
                leafLeft = decisionTree()
                leafLeft.isLeaf = True
                leafLeft.leafVal = leaf1Prob
                leafRight = decisionTree()
                leafRight.isLeaf = True
                leafRight.leafVal = leaf2Prob
                tree.left = leafLeft
                tree.right = leafRight
                leafLeft.parent = tree
                leafRight.parent = tree
            else:  # only left side leaf
                #print("only leaf1 early stop")
                leafLeft = decisionTree()
                leafLeft.isLeaf = True
                leafLeft.leafVal = leaf1Prob
                tree.left = leafLeft
                leafLeft.parent = tree
                tree.right = decisionTree()
                tree.right.parent = tree
                trainData, trainLabels = getNewData(train_data, train_labels, bestFeature, tree.range, 1)  # updates Matrix
                build(tree.right, trainData, trainLabels, 1, forestfeatures)
        else:  # first part not leaf
            if(leaf2PosFinal+leaf2NegFinal < earlyStop or ent2Final == 0):  # only right side leaf
                #print("only leaf2 early stop")
                leafRight = decisionTree()
                leafRight.isLeaf = True
                leafRight.leafVal = leaf2Prob
                tree.right = leafRight
                tree.left = decisionTree()
                tree.left.parent = tree

                trainData, trainLabels = getNewData(
                    train_data, train_labels, bestFeature, tree.range, 0)  # updates Matrix
                build(tree.left, trainData, trainLabels, 0, forestfeatures)
            else:  # both aren't leaves
                #print("no early stop for either leaves")
                tree.left = decisionTree()
                tree.left.parent = tree
                tree.right = decisionTree()
                tree.right.parent = tree
                trainDataOne, trainLabelsOne = getNewData(
                    train_data, train_labels, bestFeature, tree.range, 0)  # updates Matrix
                trainDataTwo, trainLabelsTwo = getNewData(
                    train_data, train_labels, bestFeature, tree.range, 1)
                build(tree.left, trainDataOne,
                      trainLabelsOne, 0, forestfeatures)
                build(tree.right, trainDataTwo,
                      trainLabelsTwo, 1, forestfeatures)


def solve(tree, test_data_row):  # test_data row
    if(tree.getIsLeaf() == False):
        transformed = test_data_row  # temp seeing if don't need to transform
        #transformed = transformMatrix(test_data_row,tree.feature)
        if(transformed[tree.feature] <= tree.get_range()):
            #print("transformed[tree.feature]",transformed[tree.feature],"original val",test_data_row[tree.feature],"divideval",tree.get_range())
            return solve(tree.left, test_data_row)
        else:
            #print("when feature > range transformed[tree.feature]",transformed[tree.feature],"original val",test_data_row[tree.feature],tree.get_range())
            return solve(tree.right, test_data_row)
    else:  # it is leaf so return val
        return tree.leafVal


def print_tree(tree):
    if(tree.isLeaf):
        print(tree.leafVal, "->leaf.")
    else:
        print(tree.feature, "->tree feature.")
        print_tree(tree.left)
        print_tree(tree.right)


def compareRandomForests(treeArr, trainData, trainLabels):
    ''' 
    get accuracy of each random forest and return tree with best accuracy
    '''
    accuracyFinal = 0
    treeFinal = None
    for tree in treeArr:
        prediction = []
        for row in trainData:
            prediction.append(solve(tree, row))
        accuracy = len([i for i in range(
            len(prediction)) if prediction[i] == trainLabels[i]]) / float(len(prediction))
        if(accuracy > accuracyFinal):
            accuracyFinal = accuracy
            treeFinal = tree
    #print("highest accuracy",accuracyFinal, "with",treeFinal)
    return treeFinal


def run_train_test(training_data, training_labels, testing_data):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: List[List[float]]
        training_label: List[int]
        testing_data: List[List[float]]

    Output:
        testing_prediction: List[int]
    Example:
    return [1]*len(testing_data)
    implement the decision tree and return the prediction
    """
    trainData = np.array(training_data)
    trainLabels = np.array(training_labels)
    testData = np.array(testing_data)

    #root = decisionTree()
    forestFeatures = [0, 1, 2, 3, 4, 5, 6, 7, 8,
                      9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # build(root, trainData, trainLabels,2, forestFeatures) #original tree to test against
    # treeArr=[root]
    treeArr = []
    #trainMatrix = np.insert(trainData,19,trainLabels,1)
    for _ in range(7):
        #trainSample = trainMatrix[np.random.choice(trainMatrix.shape[0], 100, replace=False)]
        random.shuffle(forestFeatures)
        anotherRoot = decisionTree()
        #trainSampleJustVals= trainSample[:,[0,19]]
        #trainSampJustLabels = trainSample[:,19]
        #trainSampleVals = trainData[:,feats]
        build(anotherRoot, trainData, trainLabels, 2, forestFeatures[0:5])
        # build(anotherRoot, trainData, trainLabels,2, forestFeatures[0:5]) #splitting numpy array back to training values and labels
        treeArr.append(anotherRoot)
    finalTree = compareRandomForests(treeArr, trainData, trainLabels)
    # print_tree(root)
    sol = []
    for row in testData:
        # sol.append(solve(root,row))
        sol.append(solve(finalTree, row))
    return sol
