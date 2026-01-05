import torch
import matplotlib.pyplot as plt
import numpy as np
import random

def create_data(w, b, data_num):# 创建数据
    x =torch.normal(0, 1, (data_num, len(w)))# normal函数的参数（均值，标准差，size） len（）函数在对矩阵使用的时候得到的是行数
    y = torch.matmul(x,w) + b  # matmul是矩阵相乘函数  只能是tensor相乘

    noise = torch.normal(0, 0.01, y.shape)# shape属性以元组的形式输出矩阵的形状如二行三列（2，3）
    y += noise# 将噪声加到y上
    return x, y

num = 500
true_w = torch.tensor([8.1, 2, 2, 4])
true_b = torch.tensor(1.1)

X, Y = create_data(true_w, true_b, num)

plt.scatter(X[:, 3], Y, 30)
plt.show()

def data_provider(data, label, batchsize): # 每次访问这个函数就能提供一批数据 其中形参data为500*4的tensor  label是500*1的tensor  batchsize是每一批数据的个数
    length = len(label)
    indices = list(range(length))
    random.shuffle(indices)
    for each in range(0, length, batchsize):
        get_indices = indices[each:each + batchsize]
        get_data = data[get_indices] # tensor的切片 在tensor的方括号中写一个list就能获得所有索引为list元素的切片
        get_label = label[get_indices]

        yield get_data, get_label
# random.shuffle（） 函数直接对列表进行修改，将列表随机打乱
batchsize = 16
for batch_x, batch_y in data_provider(X, Y, batchsize):
    print(batch_x, batch_y)
    break
def fun(x, w , b): # 预测函数
    pred_y = torch.matmul(x, w) + b
    return pred_y

def maeLoss(pred_y, y): # 平均损失
    return torch.sum(abs(pred_y - y)) / len(y)

def sgd(paras, lr):# paras：参数 ，lr:学习率
    with torch.no_grad():
        for para in paras:
            para -= para.grad * lr # 必须写-=而不能写para = para - para。grad * lr        *****
            para.grad.zero_()


# 开始训练
lr = 0.01
w_0 = torch.normal(0, 0.01, true_w.shape, requires_grad=True) #requires_grad=True代表这个w_0需要计算梯度
b_0 = torch.tensor(0.01,requires_grad=True)
print(w_0, b_0)


epochs = 50

for epoch in range(epochs):
    data_loss = 0
    for batch_x, batch_y in data_provider(X, Y, batchsize):
        pred_y = fun(batch_x, w_0, b_0)
        loss = maeLoss(pred_y,batch_y)
        loss.backward()
        sgd([w_0, b_0], lr)
        data_loss += loss
    print(f"epoch:{epoch},loss:{data_loss}")

print("the real paras is ", true_w , true_b)
print("the paras we get via training is ", w_0, b_0)


idx = int(input("第几列（0-3）"))
plt.plot(X[:, idx].detach().numpy(),X[:,idx].detach().numpy()*w_0[idx].detach().numpy() + b_0.detach().numpy())# 当数据在张量网上时不能画图，要用.detach().numpy()将数据从张量网上取下来
plt.scatter(X[:, idx], Y, 20)
plt.show()