import yfinance as yf

def fetch_data(stock_symbol, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    :param stock_symbol: The stock ticker symbol (e.g., 'AAPL').
    :param start_date: The start date for the data (YYYY-MM-DD).
    :param end_date: The end date for the data (YYYY-MM-DD).
    :return: Pandas Series with the 'Close' prices.
    """
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data['Close']
