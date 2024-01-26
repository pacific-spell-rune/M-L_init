import numpy as np
import torch as t
# import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_spilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from random import random
import math


def NN(node_in, node_h, node_out):
    init = []
    lr_h = []
    lr_out =[]

    for i in range(lr_h):
        l =[] , bi_as = []
        for j in range(node_in): 
            l.append(randoma())
            bi_as.append(randoma())
        dict_weight = {"w":l}
        dict_bias = {"b":bi_as}
        lr_h.append(dict_weight)
        lr_h.append(bi_as)
        
    for i in range(node_out): 
        l=[]
        for 