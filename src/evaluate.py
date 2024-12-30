from sklearn.metrics import mean_squared_error

def evaluate_model(actual_stock_price, predicted_stock_price):
    """
    Evaluate the model using Mean Squared Error.
    :param actual_stock_price: Actual stock prices (scaled back to original values).
    :param predicted_stock_price: Predicted stock prices (scaled back to original values).
    :return: Mean Squared Error (MSE).
    """
    mse = mean_squared_error(actual_stock_price, predicted_stock_price)
    return mse
