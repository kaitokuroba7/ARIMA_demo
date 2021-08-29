#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: demo.py
@time: 2021/8/29 15:45
"""
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import numpy


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return numpy.array(diff)


# load dataset
series = read_csv('dataset.csv', header=0)
# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7, 0, 1))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())



if __name__ == "__main__":
    pass
