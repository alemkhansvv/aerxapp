import yfinance as yf
import requests as req
from datetime import datetime, timedelta
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import openai
from bs4 import BeautifulSoup
import os
import time  # Добавил импорт для модуля time

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="max")
        info = stock.info
        if data.empty:
            return None
        latest_data = data.iloc[-1]
        response = {
            'symbol': symbol,
            'date': latest_data.name.strftime('%Y-%m-%d'),
            'open': latest_data['Open'],
            'high': latest_data['High'],
            'low': latest_data['Low'],
            'close': latest_data['Close'],
            'volume': latest_data['Volume'],
            'company_info': {
                'name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary')
            },
            'historical_prices': data.reset_index().to_dict('records')
        }
        return response
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def get_news(symbol):
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        news_list = []
        for article in news:
            title = article['title']
            summary = article.get('summary', '')
            link = article['link']
            published_at = datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')
            news_list.append({
                'title': title,
                'description': summary,
                'published_at': published_at,
                'url': link
            })
        return news_list
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def get_financials(symbol):
    try:
        stock = yf.Ticker(symbol)
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        income_statement = stock.financials
        return {
            'balance_sheet': balance_sheet.to_dict(),
            'cashflow': cashflow.to_dict(),
            'income_statement': income_statement.to_dict()
        }
    except Exception as e:
        print(f"Error fetching financials: {e}")
        return None

def forecast_arima(symbol, periods):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5y")
        if hist.empty:
            return None

        hist = hist['Close']
        model = ARIMA(hist, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)

        last_date = hist.index[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
        forecast_dict = {date.strftime('%Y-%m-%d'): value for date, value in zip(forecast_dates, forecast)}

        return forecast_dict
    except Exception as e:
        print(f"Error forecasting ARIMA: {e}")
        return None



def fetch_full_text(url):
    try:
        response = req.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        full_text = ' '.join([para.get_text() for para in paragraphs])
        return full_text
    except Exception as e:
        print(f"Error fetching text from {url}: {e}")
        return ""

def analyze_news(news_list):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    articles_text = "\n".join([f"Title: {article['title']}\nDescription: {article['description']}\nPublished Date: {article['published_at']}\n" for article in news_list])
    prompt = f"""
    Please analyze the following news articles related to the specified company and provide a comprehensive summary in one paragraph. The summary should cover:

    - Key points from the articles.
    - Overall impact on the company's stock or business.
    - General recommendations or insights for investors based on the news.

    Here are the news articles:

    {articles_text}
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ]
    )

    analysis = response.choices[0].message.content.strip()
    return analysis

def forecast_prophet(symbol, periods):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5y")
        if hist.empty:
            return None

        hist = hist.reset_index()[['Date', 'Close']]
        hist.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

        model = Prophet()
        model.fit(hist)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        forecast_dates = forecast['ds'].tail(periods).dt.strftime('%Y-%m-%d').tolist()
        forecast_values = forecast['yhat'].tail(periods).tolist()
        forecast_dict = dict(zip(forecast_dates, forecast_values))

        return forecast_dict
    except Exception as e:
        print(f"Error forecasting Prophet: {e}")
        return None

def get_investment_opinion(financial_analysis, volatility_analysis, risk_analysis, news_analysis):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Based on the following analyses, provide an investment opinion (Buy, Sell, Hold) and a brief reason why.\n\nFinancial Analysis:\n{financial_analysis}\n\nVolatility Analysis:\n{volatility_analysis}\n\nRisk Analysis:\n{risk_analysis}\n\nNews Analysis:\n{news_analysis}\n\nYour response should be structured as follows:\n\nAI Opinion: [Buy/Sell/Hold]\nReason: [Your brief reason]"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ]
    )

    result = response.choices[0].message.content.strip()
    print("AI Response:", result)  # Добавим логирование для отладки

    try:
        opinion_start = result.index("AI Opinion: ") + len("AI Opinion: ")
        reason_start = result.index("Reason: ") + len("Reason: ")
        opinion = result[opinion_start:result.index('\n', opinion_start)].strip()
        reason = result[reason_start:].strip()
    except (ValueError, IndexError) as e:
        print(f"Error parsing AI response: {e}")
        opinion = "No Opinion"
        reason = "The AI did not return a valid opinion and reason."

    return {'opinion': opinion, 'reason': reason}