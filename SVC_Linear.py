# SVC
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
# 使用LinearSVC类考察线性分类支持向量机的预测能力
def test_LinearSVC(*data):
    X_train, X_test, y_train, y_test = data
    cls = svm.LinearSVC()
    cls.fit(X_train, y_train)
    print("Coffecients:%s,\n intercept:%s"%(cls.coef_, cls.intercept_))
    print("Scores: %.2f"% cls.score(X_test, y_test))

X_train, X_test, y_train, y_test = load_data_classfication()
test_LinearSVC(X_train, X_test, y_train, y_test)
#-----------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------#
# 考察损失函数的影响
def test_LinearSVC_loss(*data):
    X_train, X_test, y_train, y_test = data
    losses = ['hinge', 'squared_hinge']
    for loss in losses:
        cls = svm.LinearSVC(loss=loss)
        cls.fit(X_train, y_train)
        print("Loss:%s"%loss)
        print("Coffecients:%s,\n intercept:%s"%(cls.coef_, cls.intercept_))
        print("Scores(Train): %.2f"% cls.score(X_train, y_train))
        print("Scores: %.2f"% cls.score(X_test, y_test))

# test_LinearSVC_loss(X_train, X_test, y_train, y_test)
# 结果不是很好，在训练集上的准确率只有0.95（0.98），测试集上的准确率为0.84（0.92），与例中给出的结果不一样
#-----------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------#
# 考察罚项形式的影响
def test_LinearSVC_L12(*data):
    X_train, X_test, y_train, y_test = data
    L12 = ['l1', 'l2']
    for p in L12:
        cls = svm.LinearSVC(penalty=p,dual=False)
        cls.fit(X_train, y_train)
        print("Loss:%s"%p)
        print("Coffecients:%s,\n intercept:%s"%(cls.coef_, cls.intercept_))
        print("Scores(Train): %.2f"% cls.score(X_train, y_train))
        print("Scores: %.2f"% cls.score(X_test, y_test))

# test_LinearSVC_L12(X_train, X_test, y_train, y_test)
# 这里dual=False是因为当dual=True，penalty=l2的情况不支持
#-----------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------#
# 考察罚项系数C的影响
# C衡量了误分点的重要性，C越大则误分点越重要
def test_LinearSVC_C(*data):
    X_train, X_test, y_train, y_test = data
    Cs = np.logspace(-2,1)
    train_scores = []
    test_scores = []
    for c in Cs:
        cls = svm.LinearSVC(C=c)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    # 绘图
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Cs, train_scores, label = "Training Scores")
    ax.plot(Cs, test_scores, label = "Testing Scores")
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"scores")
    ax.set_xscale("log")
    ax.set_title("LinearSVC")
    ax.legend(loc = 'best')
    plt.show()

# test_LinearSVC_C(X_train, X_test, y_train, y_test)
#-----------------------------------------------------------------------------------------------------------------------#