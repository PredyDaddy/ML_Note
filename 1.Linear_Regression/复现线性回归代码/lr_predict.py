#coding=utf-8
from lr import LR
import pickle
import numpy as np

if __name__ == "__main__":
    data = np.array([10,100,2,100])
    with open('model_lr.pt','rb') as f:
        model_theta = pickle.load(f)
    model_lr = LR()
    model_lr.theta = model_theta
    result = model_lr.model(data).item()
    print(result)
    

