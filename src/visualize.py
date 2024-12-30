import matplotlib.pyplot as plt

def visualize_results(actual_stock_price, predicted_stock_price, stock_symbol):
    """
    Visualize the predicted vs. actual stock prices.
    :param actual_stock_price: Actual stock prices.
    :param predicted_stock_price: Predicted stock prices.
    :param stock_symbol: The stock symbol being predicted (e.g., 'AAPL').
    """
    plt.figure(figsize=(14, 7))
    plt.plot(actual_stock_price, color='blue', label=f'{stock_symbol} Actual Price')
    plt.plot(predicted_stock_price, color='red', label=f'{stock_symbol} Predicted Price')
    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.show()
