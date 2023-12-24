from Flask import Flask, render_template, request, send_file
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from io import BytesIO

def getStockData(ticker, period = "1y"):
    if(ticker.cont)
    data = yf.download(ticker, period=period)
    return data
def prepareData(data):
    data['Signal'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    