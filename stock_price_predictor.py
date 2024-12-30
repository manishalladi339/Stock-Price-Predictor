from src.fetch_data import fetch_data
from src.preprocess import preprocess_data
from src.model import build_lstm_model
from src.predict import make_predictions
from src.visualize import visualize_results
from src.evaluate import evaluate_model

def main():
    stock_symbol = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2024-01-01'
    
    # Step 1: Fetch data
    stock_data = fetch_data(stock_symbol, start_date, end_date)
    
    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data)
    
    # Step 3: Build and train the model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Step 4: Make predictions
    predictions = make_predictions(model, X_test, scaler)
    
    # Step 5: Visualize the results
    visualize_results(y_test, predictions, stock_symbol)
    
    # Step 6: Evaluate the model
    mse = evaluate_model(y_test, predictions)
    print(f'Mean Squared Error (MSE): {mse}')

if __name__ == "__main__":
    main()
