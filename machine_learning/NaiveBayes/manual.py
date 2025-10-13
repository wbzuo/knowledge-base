import numpy as np
import pandas as pd
from tqdm import tqdm
import time


def loadData(fileName):

    dataArr = []
    labelArr = []
    fr = open(fileName, 'r')
    fr.readline()
    for line in tqdm(fr.readlines()):
        curLine = line.strip().split(',')     
           
        dataArr.append([int(int(x) > 128) for x in curLine[1:]])
        labelArr.append(int(curLine[0]))

    
    return dataArr, labelArr

def getAllProbability(train_data, train_label):
    
    
    featureNum = 784
    
    classes = list(range(10))
    classNum = len(classes)
    
    # 先验P(Y = y_i)
    Py = np.zeros((classNum, 1))
    
    
    
    for x in classes:
        
        Py[x] = (np.sum((np.asmatrix(train_label) == x)) + 1) / (len(train_label) +  classNum)
        
    
    Py = np.log(Py)       
    
    # 计算条件概率
    
    Px_y = np.zeros((classNum, featureNum, 2))
    for i in range(len(train_label)):
        label = train_label[i]
        x = train_data[i]
        
        for j in range(featureNum):
            Px_y[label][j][x[j]] += 1
            
        
    for label in range(classNum):
        for j in range(featureNum):
            Px_y0 = Px_y[label][j][0]
            Px_y1 = Px_y[label][j][1]
            
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))
    
    
    return Py, Px_y
        
    

def NaiveBayes(Py, Px_y, x):
    
    featrueNum = 784
    classNum = 10
    
    P = [0] * classNum
    for i in range(classNum):
        sum = 0
        for j in range(featrueNum):
            sum += Px_y[i][j][x[j]]

        P[i] = sum + Py[i]
    
    # max(P) 找到概率最大的类别
    # P.index(max(P)) 找到概率最大的下标
    return P.index(max(P))

def model_test(Py, Px_y, testDataArr, testLabelArr):
    '''
    对测试集进行测试
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param testDataArr: 测试集数据
    :param testLabelArr: 测试集标记
    :return: 准确率
    '''
    #错误值计数
    errorCnt = 0
    #循环遍历测试集中的每一个样本
    for i in range(len(testDataArr)):
        #获取预测值
        presict = NaiveBayes(Py, Px_y, testDataArr[i])
        #与答案进行比较
        if presict != testLabelArr[i]:
            #若错误  错误值计数加1
            errorCnt += 1
    #返回准确率
    return 1 - (errorCnt / len(testDataArr))

    
if __name__ == "__main__":
        
    start = time.time()
    # 获取训练集
    print('start read transSet')   
    train_data, train_label = loadData(r"C:\Users\Administrator\Desktop\251\study\datasets\mnist\mnist_train.csv")

    print('start read testSet')
    test_data, test_label = loadData(r"C:\Users\Administrator\Desktop\251\study\datasets\mnist\mnist_test.csv")

    #开始训练，学习先验概率分布和条件概率分布
    print('start to train')
    Py, Px_y = getAllProbability(train_data, train_label)
    
    # 使用先验和条件概率分布
    print('start to test')
    accuracy = model_test(Py, Px_y, test_data, test_label)




    end = time.time()

    #打印准确率
    print('the accuracy is:', accuracy)
    #打印时间
    print('time span:', time.time() -start)