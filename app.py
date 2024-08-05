import locale
from flask import Flask, render_template, request
from api_helpers import get_stock_data, get_news, get_financials, forecast_arima, forecast_prophet, analyze_news, fetch_full_text, get_investment_opinion
import openai
from dotenv import load_dotenv
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import scipy.stats as stats
import yfinance as yf
import os

# Установка локали для форматирования чисел
locale.setlocale(locale.LC_ALL, '')

# Загрузка переменных из .env файла
load_dotenv()

app = Flask(__name__)

# Ваши API-ключи для Finnhub
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Установка API ключа OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


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


def get_stock_volatility(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        log_returns = np.log(1 + hist['Close'].pct_change())
        volatility = np.std(log_returns) * np.sqrt(252)  # годовая волатильность
        return volatility
    except Exception as e:
        print(f"Error calculating volatility: {e}")
        return None


def plot_historical_prices(historical_data):
    df = pd.DataFrame(historical_data)
    df['Date'] = pd.to_datetime(df['Date'])

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Prices'))

    fig.update_layout(
        title='Historical Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    graph_html = fig.to_html(full_html=False)
    return graph_html


def process_financial_data(financial_data):
    processed_data = {}
    for key, value in financial_data.items():
        df = pd.DataFrame(value).T
        df.index = pd.to_datetime(df.index).date
        df = df.applymap(lambda x: locale.format_string("%d", x, grouping=True) if pd.notnull(x) and isinstance(x, (
            int, float)) else x)
        df = df[df.index > pd.to_datetime('2019-12-31').date()]  # Удаление данных за 2019 год
        processed_data[key] = df.T
    return processed_data


def analyze_financials(financial_data):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"""
    Analyze the following financial data and provide a structured summary with bold headers for each section. The summary should cover:

    - Key trends over the past four fiscal years.
    - Balance sheet analysis.
    - Cash flow statement analysis.
    - Margins and ratios.
    - Concluding statement on financial health.

    Use the following format as an example for the response, ensuring each section is clearly separated by paragraphs and new lines. IT IS ONLY THE EXAMPLE, DON'T USE IT COPY PASTED ONE. I JUST GAVE YOU TO UNDERSTAND THE STRUCTURE HOW IT SHOULD BE LOOKED:

    **Key Trends Over the Past Four Fiscal Years:**
    - Total Revenue has shown consistent growth from X billion in 2021 to Y billion in 2024. 
    - Net Income has also increased significantly from A billion in 2021 to B billion in 2024. 
    - Operating Income has shown a positive trend, reaching C billion in 2024. 
    - Free Cash Flow has experienced substantial growth, reaching D billion in 2024. 

    **Balance Sheet Analysis:**
    - Net Debt has fluctuated over the years, with a decrease from E billion in 2022 to F billion in 2024. 
    - Cash and Cash Equivalents have increased steadily, indicating improved liquidity. 
    - Total Debt has shown fluctuations but remained manageable over the years. 

    **Cash Flow Statement Analysis:**
    - Operating Cash Flow has shown positive growth, reaching G billion in 2024. 
    - Investing Cash Flow has fluctuated over the years, indicating varying investment activities.
    - Financing Cash Flow experienced notable fluctuations, with significant changes in debt issuance and stock repurchases. 

    **Margins and Ratios:**
    - Net Income Margin has shown improvement over the years, reflecting better profitability. 
    - Gross Margin has remained relatively stable, indicating efficient cost management. 
    - ROA (Return on Assets) and ROE (Return on Equity) have shown positive trends, showcasing healthy asset utilization and shareholder returns. 
    - Current Ratio and Quick Ratio have not been provided in the data but would be essential for evaluating liquidity. 

    **Concluding Statement on Financial Health:**
    The company has displayed strong financial performance over the past four years, characterized by revenue growth, increased profitability, and improving cash flow. While there have been fluctuations in debt levels and investment activities, the overall financial health appears robust with a focus on generating free cash flow and enhancing shareholder value. Investors should monitor liquidity ratios and debt management practices for a comprehensive assessment of the company's financial stability.

    Here is the financial data for analysis:
    {financial_data}
    """
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()



def calculate_volatility(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        log_returns = np.log(1 + hist['Close'].pct_change())
        hist_volatility = np.std(log_returns) * np.sqrt(252)  # годовая волатильность
        return hist_volatility
    except Exception as e:
        print(f"Error calculating volatility: {e}")
        return None


def analyze_volatility(symbol, hist_volatility):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Analyze the following volatility data for company {symbol}. Historical volatility: {hist_volatility:.2f}. Interpret this data and explain what it means for investors. (Historical Volatility is measured annually, not daily. just FYI. Just remember. For the promt: Remember, You are the greatest financial, statistic, econometric, economics and maths guy!!!!!. But You don't have to give any thanks. The text is purposed for the readers. Make it accuracy."
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def calculate_var(symbol, confidence_level=0.95):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        log_returns = np.log(1 + hist['Close'].pct_change())
        mean = np.mean(log_returns)
        std_dev = np.std(log_returns)
        var_value = stats.norm.ppf(1 - confidence_level, mean, std_dev) * np.sqrt(252)
        return var_value
    except Exception as e:
        print(f"Error calculating VaR: {e}")
        return None


def analyze_risk(symbol, var_value):
    if var_value is None:
        return "VaR calculation failed, unable to provide risk analysis."

    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Analyze the following risk data for company {symbol}. Value at Risk (VaR): {var_value:.2f}. Please provide an interpretation and explain what it means for investors."
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price


@app.route('/historical-prices')
def historical_prices():
    symbol = request.args.get('symbol')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    stock_data = get_stock_data(symbol, start_date, end_date)
    if not stock_data:
        return "Error fetching stock data"

    historical_data = stock_data['historical_prices']
    price_chart = plot_historical_prices(historical_data)

    return render_template('company.html',
                           symbol=symbol,
                           historical_data=historical_data,
                           price_chart=price_chart,
                           active_tab='historical-prices')


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/termsofuse')
def terms_of_use():
    return render_template('termsofuse.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/company')
def company():
    symbol = request.args.get('symbol')
    stock_data = get_stock_data(symbol)
    if not stock_data:
        return "Error fetching stock data"

    historical_data = stock_data['historical_prices']
    news = get_news(symbol)

    print("Fetched news:", news)  # Debugging line

    financials = get_financials(symbol)
    processed_financials = process_financial_data(financials)
    financial_analysis = analyze_financials(processed_financials)
    price_chart = plot_historical_prices(historical_data)
    hist_volatility = calculate_volatility(symbol)
    formatted_hist_volatility = f"{hist_volatility:.2f}" if hist_volatility is not None else "N/A"
    volatility_analysis = analyze_volatility(symbol, hist_volatility)
    risk_free_rate = get_risk_free_rate()
    volatility = get_stock_volatility(symbol)
    var_value = calculate_var(symbol)
    risk_analysis = analyze_risk(symbol, var_value)

    news_analysis = analyze_news(news)
    investment_opinion = get_investment_opinion(financial_analysis, volatility_analysis, risk_analysis, news_analysis)

    print("News analysis result:", news_analysis)  # Debugging line

    return render_template('company.html',
                           company_info=stock_data['company_info'],
                           historical_data=historical_data,
                           news=news,
                           financials=processed_financials,
                           financial_analysis=financial_analysis,
                           price_chart=price_chart,
                           symbol=symbol,
                           risk_free_rate=risk_free_rate,
                           volatility=volatility,
                           hist_volatility=formatted_hist_volatility,
                           volatility_analysis=volatility_analysis,
                           risk={'var': var_value},
                           risk_analysis=risk_analysis,
                           news_analysis=news_analysis,
                           investment_opinion=investment_opinion,
                           call_price=None,
                           forecast=None,
                           active_tab="ai-analysis")


@app.route('/calculate_black_scholes', methods=['POST'])
def calculate_black_scholes():
    S = float(request.form['S'])
    K = float(request.form['K'])
    T = float(request.form['T'])
    r = get_risk_free_rate()  # Используем значение из функции
    if r == "Error":
        return "Error fetching risk-free rate"
    sigma = float(request.form['sigma'])
    symbol = request.form['symbol']  # Получение символа акции из скрытого поля

    call_price = black_scholes_call(S, K, T, r, sigma)
    stock_data = get_stock_data(symbol)
    if not stock_data:
        return "Error fetching stock data"

    historical_data = stock_data['historical_prices']
    news = get_news(symbol)
    financials = get_financials(symbol)  # Получение финансовых данных

    processed_financials = process_financial_data(financials)  # Обработка финансовых данных
    financial_analysis = analyze_financials(processed_financials)
    price_chart = plot_historical_prices(historical_data)
    hist_volatility = calculate_volatility(symbol)
    formatted_hist_volatility = f"{hist_volatility:.2f}" if hist_volatility is not None else "N/A"
    volatility_analysis = analyze_volatility(symbol, hist_volatility)
    var_value = calculate_var(symbol)
    risk_analysis = analyze_risk(symbol, var_value)
    news_analysis = analyze_news(news)
    investment_opinion = get_investment_opinion(financial_analysis, volatility_analysis, risk_analysis, news_analysis)

    return render_template('company.html',
                           company_info=stock_data['company_info'],
                           historical_data=historical_data,
                           news=news,
                           financials=processed_financials,  # Передача обработанных финансовых данных в шаблон
                           call_price=call_price,
                           price_chart=price_chart,
                           symbol=symbol,
                           risk_free_rate=r,
                           volatility=sigma,
                           forecast=None,
                           financial_analysis=financial_analysis,
                           hist_volatility=formatted_hist_volatility,
                           volatility_analysis=volatility_analysis,
                           risk={'var': var_value},
                           risk_analysis=risk_analysis,
                           news_analysis=news_analysis,
                           investment_opinion=investment_opinion,
                           active_tab="mathematical-analysis")


@app.route('/forecast_arima', methods=['POST'])
def forecast_arima_route():
    symbol = request.form['symbol']
    periods = int(request.form['periods'])
    forecast = forecast_arima(symbol, periods)

    stock_data = get_stock_data(symbol)
    if not stock_data:
        return "Error fetching stock data"

    historical_data = stock_data['historical_prices']
    news = get_news(symbol)
    financials = get_financials(symbol)  # Получение финансовых данных

    processed_financials = process_financial_data(financials)  # Обработка финансовых данных

    price_chart = plot_historical_prices(historical_data)

    risk_free_rate = get_risk_free_rate()
    volatility = get_stock_volatility(symbol)

    return render_template('company.html',
                           company_info=stock_data['company_info'],
                           historical_data=historical_data,
                           news=news,
                           financials=processed_financials,  # Передача обработанных финансовых данных в шаблон
                           price_chart=price_chart,
                           forecast=forecast,
                           symbol=symbol,
                           risk_free_rate=risk_free_rate,
                           volatility=volatility,
                           active_tab="mathematical-analysis")


@app.route('/forecast_prophet', methods=['POST'])
def forecast_prophet_route():
    symbol = request.form['symbol']
    periods = int(request.form['periods'])
    forecast = forecast_prophet(symbol, periods)

    stock_data = get_stock_data(symbol)
    if not stock_data:
        return "Error fetching stock data"

    historical_data = stock_data['historical_prices']
    news = get_news(symbol)
    financials = get_financials(symbol)  # Получение финансовых данных

    processed_financials = process_financial_data(financials)  # Обработка финансовых данных

    price_chart = plot_historical_prices(historical_data)

    risk_free_rate = get_risk_free_rate()
    volatility = get_stock_volatility(symbol)

    return render_template('company.html',
                           company_info=stock_data['company_info'],
                           historical_data=historical_data,
                           news=news,
                           financials=processed_financials,  # Передача обработанных финансовых данных в шаблон
                           price_chart=price_chart,
                           forecast=forecast,
                           symbol=symbol,
                           risk_free_rate=risk_free_rate,
                           volatility=volatility,
                           active_tab="mathematical-analysis")


@app.route('/ai_analysis')
def ai_analysis():
    symbol = request.args.get('symbol')
    stock_data = get_stock_data(symbol)
    if not stock_data:
        return "Error fetching stock data"

    # Dummy data for the example
    financials = {
        'revenue': 274.5,
        'net_income': 57.4,
        'gross_margin': 40,
        'operating_margin': 30,
        'roe': 80,
        'roa': 20
    }
    volatility = {
        'historical': 20,
        'forecast': 18
    }
    risk = {
        'var': 5,
        'beta': 1.2,
        'recession': 20,
        'market_shock': 15
    }
    forecast = {
        'prophet_3m': 150
    }
    news = [
        {'title': 'Apple announces new iPhone', 'date': '2024-08-01',
         'description': 'Apple announced the release of its new iPhone model...'},
        {'title': 'Apple expands AI initiatives', 'date': '2024-07-28',
         'description': 'Apple is investing heavily in AI research...'}
    ]
    sentiment = {
        'overall': 'Позитивный'
    }
    valuation = {
        'dcf': 160,
        'pe': 30
    }
    recommendation = {
        'action': 'Покупка',
        'target_price': 155,
        'strategy': 'Долгосрочная'
    }
    conclusion = "Apple Inc. демонстрирует сильные финансовые показатели и стабильный рост. Рекомендуется покупать акции Apple Inc. с целевой ценой $155."

    return render_template('company.html',
                           company_info=stock_data['company_info'],
                           financials=financials,
                           volatility=volatility,
                           risk=risk,
                           forecast=forecast,
                           news=news,
                           sentiment=sentiment,
                           valuation=valuation,
                           recommendation=recommendation,
                           conclusion=conclusion,
                           symbol=symbol,
                           active_tab="ai-analysis")


if __name__ == '__main__':
    app.run(debug=True)
