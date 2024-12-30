from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(stock_data, time_step=60):
    """
    Preprocess the stock data: scaling and sequence creation for LSTM.
    :param stock_data: Stock data (closing prices).
    :param time_step: Number of previous days to predict the next day's price.
    :return: Preprocessed data, including training and test sets.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))
    
    def create_sequences(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split data into training and test sets (80% training, 20% testing)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler
