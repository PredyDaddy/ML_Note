#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 设置随机种子  保证每次初始化化值相同
np.random.seed(4)

class LR:
    def __init__(self,data=None,learning_rate=0.01,iter_max=200,batch_size=10):
        # 初始化模型超参数
        # learning_rate=0.001,学习率一般在0.001-0.1
        # iter_max=30,设置最大迭代次数 控制模型训练停止条件
        # batch_size=1   控制梯度下降法(随机梯度下降法，批量梯度下降法小批量梯度下降法)
        self.data = data
        self.learning_rate = learning_rate
        self.iter_max = iter_max
        self.batch_size = batch_size
    
    def data_prosses(self):
        # 数据预处理
        # 扩充常数项维度
        data = np.array(self.data)
        # 数据标准化
        # 计算均值和标准差
        mean = np.mean(data[:,:-1],axis=0)
        std = np.std(data[:,:-1],axis=0)
        data[:,:-1] = (data[:,:-1]-mean)/std
        
        one = np.ones((data.shape[0],1))
        self.data = np.hstack((one,data))
        # 训练总样本数
        self.m = self.data.shape[0]
        # 特征数 包含偏置项
        self.n = self.data.shape[1]-1  
    
    def model(self,train_data):
        # 定义算法模型
        y  = np.dot(train_data,self.theta)
        return y
    
    def tarin(self):
        # 模型训练
        n = 0   # 记录参数更新次数
        all_loss = [] # 记录每一次更新后的损失值 
        # 1、数据预处理
        self.data_prosses()
        # 2、初始化模型参数theta
        self.theta = np.random.rand(self.n,1)
        # 3、前向传播 不断训练模型
        b_n = self.m//self.batch_size   # //向下取整
        while True:
            # 每训练完一轮数据 打乱数据
            np.random.shuffle(self.data)
            for i in range(b_n):
                train_data = self.data[i*self.batch_size:(i+1)*self.batch_size]
                # 调用算法模型得到预测结果
                predict_y = self.model(train_data[:,:-1])
                # 计算损失 调用损失函数
                loss = self.loss(predict_y,train_data[:,-1:])   # train_data[:,-1:]样本真实值
                all_loss.append(loss)
                # 计算梯度  
                self.cal_grad(predict_y,train_data[:,-1:],train_data)
                # 更新参数theta的值
                self.theta = self.theta - self.learning_rate*self.grad
                n += 1 
                print(f'迭代次数:{n}\t当前误差值:{loss}\t模型参数:{self.theta}')
            # 控制模型训练停止条件
            if self.iter_max<n:
                break
        self.draw(all_loss)
        # 保存模型参数
        with open('model_lr.pt','wb') as f:
            pickle.dump(self.theta,f)
    
    def loss(self,y_predict,y_true):
        # 定义损失函数  回归模型  使用平方损失函数
        return np.sum((y_predict-y_true)**2)
        
    def cal_grad(self,y_predict,y_true,train_data):
        # 计算梯度
        self.grad = np.zeros((self.n,1))
        for i in range(self.n):
            self.grad[i,0] = np.mean((y_predict-y_true)*train_data[:,i:i+1])
        return self.grad
            
    @staticmethod
    def draw(all_loss):
        plt.plot(all_loss)
        plt.show()
        

if __name__ == "__main__":
    # 读取训练数据
    data = pd.read_excel(r'lr.xlsx')
    lr = LR(data)
    lr.tarin()
    
