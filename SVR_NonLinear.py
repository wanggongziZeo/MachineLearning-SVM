# SVR
# 导入包
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model, cross_validation, svm


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

#-----------------------------------------------------------------------------------------------------------------------#
# 主要观察不同的核对预测性能的影响
# 首先观察最简单的线性核K(X,Z) = XZ
def test_SVR_linear(*data):
    X_train, X_test, y_train, y_test = data
    regr = svm.SVR(kernel="linear")
    regr.fit(X_train, y_train)
    print("Coffecients:%s,\n intercept:%s"%(regr.coef_, regr.intercept_))
    print("Scores: %.2f"% regr.score(X_test, y_test))

X_train, X_test, y_train, y_test = load_data_regression()
# test_SVR_linear(X_train, X_test, y_train, y_test)
#-----------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------#
# 考察多项式核
# p由degree参数决定，y由gamma参数决定，r由coef0参数决定
def test_SVR_poly(*data):
    X_train, X_test, y_train, y_test = data
    fig = plt.figure()
    ### 测试degree ###
    degrees = range(1,20)
    train_scores = []
    test_scores = []
    for degree in degrees:
        regr = svm.SVR(kernel="poly", degree=degree, coef0=1)
        regr.fit(X_train, y_train)
        train_scores.append(regr.score(X_train, y_train))
        test_scores.append(regr.score(X_test, y_test))
    ax = fig.add_subplot(1,3,1)
    ax.plot(degrees, train_scores, label = "Training Scores", marker = '+')
    ax.plot(degrees, test_scores, label = "Testing Scores", marker = 'o')
    ax.set_xlabel("p")
    ax.set_ylabel("scores")
    ax.set_ylim(-1,1.)
    ax.set_title("SVR_poly_degree")
    ax.legend(loc = 'best', framealpha = 0.5)
    
    ### 测试gamma ###
    gammas = range(1,20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        regr = svm.SVC(kernel="poly", gamma=gamma, degree=3, coef0=1)
        regr.fit(X_train, y_train)
        train_scores.append(regr.score(X_train, y_train))
        test_scores.append(regr.score(X_test, y_test))
    ax = fig.add_subplot(1,3,2)
    ax.plot(gammas, train_scores, label = "Training Scores", marker = '+')
    ax.plot(gammas, test_scores, label = "Testing Scores", marker = 'o')
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("scores")
    ax.set_ylim(-1,1.)
    ax.set_title("SVR_poly_gamma")
    ax.legend(loc = 'best', framealpha = 0.5)

    ### 测试r ###
    rs = range(0,20)
    train_scores = []
    test_scores = []
    for r in rs:
        regr = svm.SVR(kernel="poly", coef0=r, gamma = 20, degree=3)
        regr.fit(X_train, y_train)
        train_scores.append(regr.score(X_train, y_train))
        test_scores.append(regr.score(X_test, y_test))
    ax = fig.add_subplot(1,3,3)
    ax.plot(rs, train_scores, label = "Training Scores", marker = '+')
    ax.plot(rs, test_scores, label = "Testing Scores", marker = 'o')
    ax.set_xlabel(r"r")
    ax.set_ylabel("scores")
    ax.set_ylim(-1,1.)
    ax.set_title("SVR_poly_r")
    ax.legend(loc = 'best', framealpha = 0.5)
    plt.show()

# test_SVR_poly(X_train, X_test, y_train, y_test)
#-----------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------#
# 考察高斯核
# p由degree参数决定，y由gamma参数决定，r由coef0参数决定
def test_SVR_rbf(*data):
    X_train, X_test, y_train, y_test = data
    gammas = range(1,20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        regr = svm.SVR(kernel="poly", gamma=gamma)
        regr.fit(X_train, y_train)
        train_scores.append(regr.score(X_train, y_train))
        test_scores.append(regr.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(gammas, train_scores, label = "Training Scores", marker = '+')
    ax.plot(gammas, test_scores, label = "Testing Scores", marker = 'o')
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("scores")
    ax.set_ylim(-1,1.05)
    ax.set_title("SVR_rbf")
    ax.legend(loc = 'best', framealpha = 0.5)
    plt.show()
    
# test_SVR_rbf(X_train, X_test, y_train, y_test)
#-----------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------#
# 考察sigmoid核
# p由degree参数决定，y由gamma参数决定，r由coef0参数决定
def test_SVR_sigmoid(*data):
    X_train, X_test, y_train, y_test = data
    fig = plt.figure()

    ### 测试gamma ###
    gammas = np.logspace(-1,3)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        regr = svm.SVC(kernel="sigmoid", gamma=gamma, coef0=0.01)
        regr.fit(X_train, y_train)
        train_scores.append(regr.score(X_train, y_train))
        test_scores.append(regr.score(X_test, y_test))
    ax = fig.add_subplot(1,2,1)
    ax.plot(gammas, train_scores, label = "Training Scores", marker = '+')
    ax.plot(gammas, test_scores, label = "Testing Scores", marker = 'o')
    ax.set_xlabel(r"$\gamma$")
    ax.set_xscale('log')
    ax.set_ylabel("scores")
    ax.set_ylim(-1,1.05)
    ax.set_title("SVR_sigmoid_gamma r=0.01")
    ax.legend(loc = 'best', framealpha = 0.5)

    ### 测试r ###
    rs = np.linspace(0,5)
    train_scores = []
    test_scores = []
    for r in rs:
        regr = svm.SVC(kernel="sigmoid", coef0=r, gamma = 10)
        regr.fit(X_train, y_train)
        train_scores.append(regr.score(X_train, y_train))
        test_scores.append(regr.score(X_test, y_test))
    ax = fig.add_subplot(1,2,2)
    ax.plot(rs, train_scores, label = "Training Scores", marker = '+')
    ax.plot(rs, test_scores, label = "Testing Scores", marker = 'o')
    ax.set_xlabel(r"r")
    ax.set_ylabel("scores")
    ax.set_ylim(-1,1.05)
    ax.set_title("SVR_sigmoid_r gamma=10")
    ax.legend(loc = 'best', framealpha = 0.5)
    plt.show()

test_SVR_sigmoid(X_train, X_test, y_train, y_test)
#-----------------------------------------------------------------------------------------------------------------------#
