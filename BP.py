import numpy as np
import pandas as pd


class neuralNetworks:
    def __init__(self, d, q, l):
        self.d = d
        self.q = q
        self.l = l
        self.v = (np.random.rand(d, q) + 0.001) % 1
        self.w = (np.random.rand(q, l) + 0.001) % 1
        self.gamma = (np.random.rand(q) + 0.001) % 1
        self.theta = (np.random.rand(l) + 0.001) % 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x_train, y_train, lr=0.1, tol_iter=1000):
        for x in range(tol_iter):
            for i in range(x_train.shape[0]):
                alpha = np.dot(x_train[i], self.v)
                b = self.sigmoid(alpha - self.gamma)
                beta = np.dot(b, self.w)
                yhatk = self.sigmoid(beta - self.theta)
                g = yhatk*(1-yhatk)*(y_train[i]-yhatk)
                e = b*(1-b)*np.dot(self.w, g)
                dw = lr*np.dot(b.reshape([b.shape[0], 1]),
                               g.reshape(g.shape[0], 1).T)
                dt = -lr*g
                dv = lr * \
                    np.dot(x_train[i].reshape(
                        [x_train[i].shape[0], 1]), e.reshape([e.shape[0], 1]).T)
                dg = -lr*e
                self.v += dv
                self.w += dw
                self.gamma += dg
                self.theta += dt

    def predict(self, x_test):
        yhat = list()
        for i in range(x_test.shape[0]):
            alpha = np.dot(x_train[i], self.v)
            b = self.sigmoid(alpha - self.gamma)
            beta = np.dot(b, self.w)
            yhat.append(np.round(self.sigmoid(beta - self.theta)))
        yhat = np.vstack(yhat)
        return yhat


if __name__ == '__main__':
    # 读取鸢尾花数据集
    data = pd.read_csv('Iris.csv')
    data = data.drop('Id', axis=1)

    # 分割鸢尾花数据集
    train_df = data.groupby('Species', group_keys=False).apply(
        lambda x: x.sample(frac=0.7))
    test_df = data[~data.index.isin(train_df.index)]
    x = data.drop('Species', axis=1).values
    y = data['Species'].values
    x_train = train_df.drop('Species', axis=1).values
    y_train = train_df['Species'].values
    x_test = test_df.drop('Species', axis=1).values
    y_test = test_df['Species'].values

    # 预处理鸢尾花数据集
    for i in range(y.shape[0]):
        if y[i] == 'Iris-setosa':
            y[i] = np.array([1, 0, 0])
        elif y[i] == 'Iris-versicolor':
            y[i] = np.array([0, 1, 0])
        else:
            y[i] = np.array([0, 0, 1])
    for i in range(y_train.shape[0]):
        if y_train[i] == 'Iris-setosa':
            y_train[i] = np.array([1, 0, 0])
        elif y_train[i] == 'Iris-versicolor':
            y_train[i] = np.array([0, 1, 0])
        else:
            y_train[i] = np.array([0, 0, 1])
    for i in range(y_test.shape[0]):
        if y_test[i] == 'Iris-setosa':
            y_test[i] = np.array([1, 0, 0])
        elif y_test[i] == 'Iris-versicolor':
            y_test[i] = np.array([0, 1, 0])
        else:
            y_test[i] = np.array([0, 0, 1])
    y = np.vstack(y)
    y_train = np.vstack(y_train)
    y_test = np.vstack(y_test)
    # 训练模型
    model = neuralNetworks(4, 6, 3)
    model.fit(x_train, y_train)
    # 评估模型
    y_pred = model.predict(x_train)
    print('模型在鸢尾花数据集的训练集上的结果如下：')
    print(y_pred)
    y_pred = model.predict(x_test)
    print('模型在鸢尾花数据集的测试集上的结果如下：')
    print(y_pred)

    # 读取西瓜数据集
    data = pd.read_csv('melon.csv')
    data = data.drop('编号', axis=1)

    # 分割西瓜数据集
    train_df = data.groupby('好瓜', group_keys=False).apply(
        lambda x: x.sample(frac=0.7))
    test_df = data[~data.index.isin(train_df.index)]
    x = data.drop('好瓜', axis=1).values
    y_in = data['好瓜'].values
    x_train = train_df.drop('好瓜', axis=1).values
    y_train_in = train_df['好瓜'].values
    x_test = test_df.drop('好瓜', axis=1).values
    y_test_in = test_df['好瓜'].values

    # 预处理西瓜数据集
    y = np.zeros((y_in.shape[0], 2))
    for i in range(y_in.shape[0]):
        if y_in[i] == 1:
            y[i, 0] = 1
        else:
            y[i, 1] = 1

    y_train = np.zeros((y_train_in.shape[0], 2))
    for i in range(y_train_in.shape[0]):
        if y_train_in[i] == 1:
            y_train[i, 0] = 1
        else:
            y_train[i, 1] = 1

    y_test = np.zeros((y_test_in.shape[0], 2))
    for i in range(y_test_in.shape[0]):
        if y_test_in[i] == 1:
            y_test[i, 0] = 1
        else:
            y_test[i, 1] = 1

    y = np.vstack(y)
    y_train = np.vstack(y_train)
    y_test = np.vstack(y_test)
    # 训练模型
    model = neuralNetworks(2, 4, 2)
    model.fit(x_train, y_train)
    # 评估模型
    y_pred = model.predict(x_train)
    print('模型在西瓜数据集的训练集上的结果如下：')
    print(y_pred)
    y_pred = model.predict(x_test)
    print('模型在西瓜数据集的测试集上的结果如下：')
    print(y_pred)
    pass
