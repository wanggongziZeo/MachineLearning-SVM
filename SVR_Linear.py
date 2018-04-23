# SVR
# 导入包
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model, cross_validation, svm

#-----------------------------------------------------------------------------------------------------#
# 在支持向量机回归问题中，使用的数据集是scikit-learn自带的一个糖尿病病人的数据集。该数据集特点如下：
# 数据集有442个样本
# 每个样本有10个特征
# 每个特征都是浮点数，数据都在-0.2~0.2之间，样本的目标在整数25~346之间
# 加载数据集
def load_data_regression():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target, test_size = 0.25, random_state = 0)

# 在支持向量机分类问题中，使用的是鸢尾花数据集。该数据集特点如下：
# 数据集有150个数据，分成3类，每类50个数据
# 每个数据包含4个属性
# 采用分层抽样
# 加载数据集
def load_data_classfication():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return cross_validation.train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)
#-----------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------#
def test_LinearSVR(*data):
    X_train, X_test, y_train, y_test = data
    regr = svm.LinearSVR()
    regr.fit(X_train, y_train)
    # print("y_test_real:{}".format(y_test[0]))
    # print("y_test_output:%s"%(regr.predict(X_test[0])))
    print("Cofficients:%s, intercept:%s"%(regr.coef_, regr.intercept_))
    print("Score: %.2f" % regr.score(X_test, y_test))

X_train, X_test, y_train, y_test = load_data_regression()
# test_LinearSVR(X_train, X_test, y_train, y_test)
#-----------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------#
# 考察损失函数类型对预测性能的影响
def test_LinearSVR_loss(*data):
    X_train, X_test, y_train, y_test = data
    losses = ['epsilon_insensitive', 'squared_epsilon_insensitive']
    for loss in losses:
        regr = svm.LinearSVR(loss=loss)
        regr.fit(X_train, y_train)
        print("Loss:", loss)
        print("Cofficients:%s, intercept:%s"%(regr.coef_, regr.intercept_))
        print("Score: %.2f" % regr.score(X_test, y_test))
        print()

# test_LinearSVR_loss(X_train, X_test, y_train, y_test)
#-----------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------#
# 考察epsilon对预测性能的影响
def test_LinearSVR_epsilon(*data):
    X_train, X_test, y_train, y_test = data
    epsilons = np.logspace(-2,2)
    train_scores = []
    test_scores = []
    for epsilon in epsilons:
        regr = svm.LinearSVR(epsilon=epsilon, loss = "squared_epsilon_insensitive")
        regr.fit(X_train, y_train)
        train_scores.append(regr.score(X_train, y_train))
        test_scores.append(regr.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(epsilons, train_scores, label = "Training Scores", marker = '+')
    ax.plot(epsilons, test_scores, label = "Testing Scores", marker = 'o')
    ax.set_xlabel(r"$\epsilon$")
    ax.set_xscale('log')
    ax.set_ylabel("scores")
    ax.set_ylim(-1,1.05)
    ax.set_title("LinearSVR_epsilon")
    ax.legend(loc = 'best', framealpha = 0.5)
    plt.show()

# test_LinearSVR_epsilon(X_train, X_test, y_train, y_test)
#-----------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------#
# 考察罚项系数C对预测性能的影响
def test_LinearSVR_C(*data):
    X_train, X_test, y_train, y_test = data
    Cs = np.logspace(-1,2)
    train_scores = []
    test_scores = []
    for c in Cs:
        regr = svm.LinearSVR(epsilon=0.1, loss = "squared_epsilon_insensitive", C=c)
        regr.fit(X_train, y_train)
        train_scores.append(regr.score(X_train, y_train))
        test_scores.append(regr.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Cs, train_scores, label = "Training Scores", marker = '+')
    ax.plot(Cs, test_scores, label = "Testing Scores", marker = 'o')
    ax.set_xlabel(r"C")
    ax.set_xscale('log')
    ax.set_ylabel("scores")
    ax.set_ylim(-1,1.05)
    ax.set_title("LinearSVR_C")
    ax.legend(loc = 'best', framealpha = 0.5)
    plt.show()

test_LinearSVR_C(X_train, X_test, y_train, y_test)