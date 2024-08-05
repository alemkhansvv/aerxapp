import yfinance as yf

def get_risk_free_rate():
    try:
        # Используем тикер ^TNX для 10-Year Treasury Note Yield
        stock = yf.Ticker("^TNX")
        data = stock.history(period="1d")  # Запрашиваем данные за последний день
        if data.empty:
            return "No data available"  # В случае отсутствия данных, возвращаем соответствующий текст
        latest_data = data.iloc[-1]
        print("Latest data:", latest_data)  # Выводим последнюю строку данных для диагностики
        risk_free_rate = latest_data['Close'] / 100  # Преобразуем значение в процент
        return risk_free_rate
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return "Error"  # В случае ошибки, возвращаем текст "Error"

rate = get_risk_free_rate()
print("Risk-free rate:", rate)
