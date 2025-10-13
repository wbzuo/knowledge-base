import numpy as np
import pandas as pd
from tqdm import tqdm

def loadData(fileName):
    print("start to read data")
    dataArr = []
    labelArr = []
    fr = open(fileName, 'r')
    fr.readline()
    for line in tqdm(fr.readlines()):
        curLine = line.strip().split(',')

        labelArr.append(1 if  curLine[0] >= "5" else -1)
        
        dataArr.append([int(x) for x in curLine[1:]])

    
    return dataArr, labelArr


def perceptron(dataArr, labelArr, epochs = 50):
    """_summary_

    Args:
        dataArr (_type_): _description_
        labelArr (_type_): _description_
        epochs (int, optional): _description_. Defaults to 50.
    """
    
    print("start to train")
    dataMat = np.asmatrix(dataArr)
    labelMat = np.asmatrix(labelArr).T
    print(dataMat.shape)
    print(labelMat.shape)
    
    
    
    m, n = dataMat.shape
    
    w = np.zeros((1, n))
    b = 0
    learning_rate = 0.0001
    
    for epoch in tqdm(range(epochs), desc = f"{epochs}"):
        for i in tqdm(range(m)):
            xi = dataMat[i]
            yi = labelMat[i]

            # 误分类的情况
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + learning_rate * yi * xi
                b = b + learning_rate * yi
        print('Round %d: %d training' % (epoch, epochs))
    
    return w, b
  
def model_test(dataArr, labelArr, w, b):
    dataMat = np.asmatrix(dataArr)
    labelMat = np.asmatrix(labelArr).T
    
    m, n = dataMat.shape
    errorCnt = 0
    
    for i in tqdm(range(m)):
        xi = dataMat[i]
        yi = labelMat[i]
        
        res = -1 * yi * (w * xi.T + b)
        
        if res >=  0: errorCnt += 1
    
    acc = 1 - (errorCnt / m)

    return acc
    
            
    
    
train_data, train_label = loadData(r"C:\Users\Administrator\Desktop\251\study\datasets\mnist\mnist_train.csv")
test_data, test_label = loadData(r"C:\Users\Administrator\Desktop\251\study\datasets\mnist\mnist_test.csv")


w, b = perceptron(train_data, train_label, epochs = 50)


accrurate_rate = model_test(test_data, test_label, w, b)

print(accrurate_rate)