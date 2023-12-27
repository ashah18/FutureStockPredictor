from flask import Flask, render_template, request, send_file
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
    #if(yf.Ticker(ticker)==1):
    data = yf.download(ticker, period=period)
    return data
    #return None
def prepareData(data):
    data['Signal'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    targets = data['Signal']
    return features, targets

def train_model(features, targets):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def plot_stock_performance(stock_data):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.title('Stock Performance in the Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.show()

def plot_price_projection(model, features):
    future_days = 30  # Number of days for price projection
    future_dates = pd.date_range(start=features.index[-1], periods=future_days+1, freq='B')[1:]
    
    projected_prices = []

    for _ in range(future_days):
        prediction = model.predict(features.iloc[-1].values.reshape(1, -1))[0]
        projected_prices.append(features.iloc[-1]['Close'])

        # Assuming 1 as an increase, 0 as no change, -1 as a decrease in price
        if prediction == 1:
            features.iloc[-1]['Close'] *= 1.01  # Increase by 1%
        elif prediction == 0:
            features.iloc[-1]['Close'] *= 1.001  # Increase by 0.1%
        else:
            features.iloc[-1]['Close'] *= 0.99  # Decrease by 1%

    plt.figure(figsize=(10, 6))
    plt.plot(features['Close'], label='Historical Close Price')
    plt.plot(future_dates, projected_prices, label='Projected Close Price', linestyle='dashed')
    plt.title('Stock Price Projection for the Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.show()

app = Flask(__name__, template_folder='template')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    graph_url = None

    if request.method == 'POST':
        stock_input = request.form['search'].upper()

        try:
            stock_data = getStockData(stock_input, period="1y")
            features, _ = prepareData(stock_data)
            model, _, _ = train_model(features, stock_data['Signal'])

            # Make stock price projection for the next month
            projected_prices = []
            for _ in range(30):
                prediction = model.predict(features.iloc[-1].values.reshape(1, -1))[0]
                projected_prices.append(features.iloc[-1]['Close'])

                if prediction == 1:
                    features.iloc[-1]['Close'] *= 1.01
                elif prediction == 0:
                    features.iloc[-1]['Close'] *= 1.001
                else:
                    features.iloc[-1]['Close'] *= 0.99

            # Plot the stock price projection
            plt.plot(features['Close'], label='Historical Close Price')
            plt.plot(projected_prices, label='Projected Close Price', linestyle='dashed')
            plt.title('Stock Price Projection for the Next 30 Days')
            plt.xlabel('Days')
            plt.ylabel('Stock Price (USD)')
            plt.legend()
            plt.close()  # Close the plot to avoid displaying it on the web page

            # Save the plot to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            graph_url = f"data:image/png;base64,{buffer.getvalue().decode('utf-8')}"

            prediction = "Prediction: Buy" if prediction == 1 else "Prediction: Hold" if prediction == 0 else "Prediction: Sell"
            return render_template('mainPage.html', prediction=prediction, graph_url=graph_url)
        
        except Exception as e:
            print(f"Error fetching stock data: {e}")

    return render_template("mainPage.html")

if __name__ == "__main__":
    app.run(debug=True, port = 5500)