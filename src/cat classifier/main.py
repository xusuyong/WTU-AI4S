# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from lr_utils import load_dataset

# 导入数据库
import h5py
from PIL import Image

# import imageio
# from skimage.transform import resize

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# 将维度降低并转置。flatten是变平的意思
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# 将数据归一化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(z):  # 参数z ：任何大小的标量或numpy数组。
    A = 1 / (1 + np.exp(-z))
    return A


def initialize_parameters(dim):  # 这里的dim=12288
    W = np.zeros(shape=(1, dim))
    b = 0
    # assert(W.shape == (1,dim))
    # assert(isinstance(b, float) or isinstance(b, int)) #b的类型是float或者是int
    parameters = {"W": W, "b": b}
    return parameters


def forward_propagation(X_train, parameters):
    W = parameters["W"]
    b = parameters["b"]
    Z = np.dot(W, X_train) + b

    A = sigmoid(Z)
    #     assert(A.shape == (1,X_train.shape[1]))
    cache = {"Z": Z, "A": A}
    return (A, cache)


def compute_cost(A, Y_train):
    m = Y_train.shape[1]
    cost = (-1 / m) * np.sum(Y_train * np.log(A) + (1 - Y_train) * (np.log(1 - A)))
    # cost = np.squeeze(cost)
    #     assert(isinstance(cost,float))
    return cost


def backward_propagation(cache, X_train, Y_train):
    m = X_train.shape[1]
    A = cache["A"]
    dW = (1 / m) * np.dot((A - Y_train), X_train.T)
    db = (1 / m) * np.sum(A - Y_train)
    grads = {"dW": dW, "db": db}
    return grads


def update_parameters(parameters, grads, learning_rate):
    W = parameters["W"]
    b = parameters["b"]
    dW = grads["dW"]
    db = grads["db"]
    W = W - learning_rate * dW
    b = b - learning_rate * db
    parameters = {"W": W, "b": b}
    return parameters


def model(X_train, Y_train, learning_rate, num_iterations, print_cost=False):
    parameters = initialize_parameters(dim=12288)  # 初始化w和b
    costs = []
    for i in range(num_iterations):
        A, cache = forward_propagation(
            X_train, parameters
        )  # 传进去的是初始化的parameters，一个字典
        cost = compute_cost(A, Y_train)
        grads = backward_propagation(cache, X_train, Y_train)
        parameters = update_parameters(parameters, grads, learning_rate)
        if i % 100 == 0:
            costs.append(cost)
        if print_cost:
            if i % 100 == 0:
                print("第 ", i, " 次循环，成本函数值为：" + str(cost))

    d = {"parameters": parameters, "costs": costs, "learning_rate": learning_rate}
    return d


d = model(
    train_set_x, train_set_y, learning_rate=0.005, num_iterations=2001, print_cost=True
)


def predict(parameters, X):
    A, cache = forward_propagation(X, parameters)  # 这次用的是优化完成的parameters
    predictions = np.round(A)  # round是四舍五入取整函数
    return predictions


predictions_train = predict(d["parameters"], train_set_x)  # 训练集
predictions_test = predict(d["parameters"], test_set_x)  # 测试集
# 打印训练后的准确性
# print(type(train_set_y),predictions_train)
print(
    "训练集准确性：",
    format(100 - np.mean(np.abs(predictions_train - train_set_y)) * 100),
    "%",
)
print(
    "测试集准确性：",
    format(100 - np.mean(np.abs(predictions_test - test_set_y)) * 100),
    "%",
)

# 绘制图
costs = np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iterations (per hundreds)")
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


print("\n" + "-------------------------------------------------------" + "\n")
learning_rates = [0.01, 0.001, 0.005, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(
        train_set_x, train_set_y, num_iterations=1500, learning_rate=i, print_cost=False
    )
    predictions_train = predict(models[str(i)]["parameters"], train_set_x)  # 训练集
    predictions_test = predict(models[str(i)]["parameters"], test_set_x)  # 测试集
    # 打印训练后的准确性
    # print(type(train_set_y),predictions_train)
    print(
        "训练集准确性：",
        format(100 - np.mean(np.abs(predictions_train - train_set_y)) * 100),
        "%",
    )
    print(
        "测试集准确性：",
        format(100 - np.mean(np.abs(predictions_test - test_set_y)) * 100),
        "%",
    )

    print("\n" + "-------------------------------------------------------" + "\n")

for i in learning_rates:
    plt.plot(
        np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"])
    )

plt.ylabel("cost")
plt.xlabel("iterations")

legend = plt.legend(loc="upper center", shadow=True)
frame = legend.get_frame()
frame.set_facecolor("0.90")
plt.show()

# #用自己的照片测试模型
# my_image = "456.jpg"  # 修改你图像的名字
# ## END CODE HERE ##
# # We preprocess the image to fit your algorithm.
# fname = "images/" + my_image  # 图片位置
# image = np.array(imageio.imread(fname))  # 读入图片为矩阵
# print(image.shape)
#
# # print(num_px)
# # 先把图片放缩到 64x64
# # 转置图片为 (num_px*num_px*3, 1)向量
# my_image = resize(image, output_shape=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
# print(my_image)
# my_predicted_image = predict(d["w"], d["b"], my_image)  # 用训练好的参数来预测图像
# plt.imshow(image)
# # print(classes)
# print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
#     int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
