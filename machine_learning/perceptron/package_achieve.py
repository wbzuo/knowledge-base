import numpy as np
from sklearn.linear_model import Perceptron
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


   
train_data, train_label = loadData(r"C:\Users\Administrator\Desktop\251\study\datasets\mnist\mnist_train.csv")
test_data, test_label = loadData(r"C:\Users\Administrator\Desktop\251\study\datasets\mnist\mnist_test.csv")

perceptron = Perceptron()
perceptron.fit(train_data, train_label)
print("w", perceptron.coef_, "\n b",perceptron.intercept_, "\n, n_iter",perceptron.n_iter_)



res = perceptron.score(test_data, test_label)

print(res)