import numpy as np
import time

from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB


def loadData(fileName):

    dataArr = []
    labelArr = []
    fr = open(fileName, 'r')
    fr.readline()
    for line in tqdm(fr.readlines()):
        curLine = line.strip().split(',')

        labelArr.append(curLine[0])
        
        dataArr.append([int(x) for x in curLine[1:]])

    
    return dataArr, labelArr




def model_test(model, testDataArr, testLabelArr):
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

    pred = model.predict(testDataArr)
    #与答案进行比较
    
        
    errorCnt = np.sum(pred != testLabelArr)
    #返回准确率
    return 1 - (errorCnt / len(testDataArr))




if __name__ == "__main__":
    start = time.time()
    # 获取训练集
    print('start read transSet')
    trainDataArr, trainLabelArr = loadData(r'C:\Users\Administrator\Desktop\251\study\datasets\mnist\mnist_train.csv')

    # 获取测试集
    print('start read testSet')
    testDataArr, testLabelArr = loadData(r'C:\Users\Administrator\Desktop\251\study\datasets\mnist\mnist_test.csv')

    # 使用朴素贝叶斯
    
    
    print('start to train')
    
    # scikit-learn 中的三个实现类：
    # GaussianNB：当样本的特征大部分是连续型数值（例如身高、温度、像素值等）时，合适。
    # MultinomialNB：当样本的特征大部分是多元离散值（例如文本分类中的词频计数、每篇文章中某个单词出现的次数）时，这个模型是更好的选择。
    # BernoulliNB：当样本的特征是二元离散值（例如文本分类中的“单词是否出现”，用 1 和 0 表示）或者虽然是多元离散值但非常稀疏（大部分特征值为0）时，应该使用这个模型。
    
    # naiveBayes = MultinomialNB() # the accuracy is: 0.8365 time span: 13.266
    # naiveBayes = BernoulliNB() # accuracy is: 0.8413 time span: 13.6598
    # naiveBayes = GaussianNB() # the accuracy is: 0.5558 time span: 11.104
    
    naiveBayes.fit(trainDataArr, trainLabelArr)

    #使用习得的先验概率分布和条件概率分布对测试集进行测试
    print('start to test')
    accuracy = model_test(naiveBayes, testDataArr, testLabelArr)

    #打印准确率
    print('the accuracy is:', accuracy)
    #打印时间
    print('time span:', time.time() -start)