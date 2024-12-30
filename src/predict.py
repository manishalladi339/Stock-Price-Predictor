def make_predictions(model, X_test, scaler):
    """
    Make predictions using the trained model.
    :param model: Trained LSTM model.
    :param X_test: Test data for making predictions.
    :param scaler: The MinMaxScaler used for scaling the data.
    :return: Predicted stock prices.
    """
    predictions = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predictions)
    return predicted_stock_price
