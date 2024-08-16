import locale
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from api_helpers import get_stock_data, get_news, get_financials, forecast_arima, forecast_prophet, analyze_news, fetch_full_text, get_investment_opinion, get_company_type, analyze_dcf_model
import openai
from dotenv import load_dotenv
from dcf import perform_dcf_analysis
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import scipy.stats as stats
import yfinance as yf
import os
from markupsafe import Markup
from markdown import markdown
from auth import register_user, login_user, user_exists
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from models import db, User

# Установка локали для форматирования чисел
locale.setlocale(locale.LC_ALL, '')

# Загрузка переменных из .env файла
load_dotenv()

def create_app():
    app = Flask(__name__)

    # Настройки базы данных
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///instance/database.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Проверка URI базы данных
    if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
        app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

    # Инициализация SQLAlchemy и миграций
    db.init_app(app)
    Migrate(app, db)

    # Создание таблицы базы данных
    with app.app_context():
        db.create_all()

    return app

app = create_app()

@app.template_filter('markdown')
def markdown_filter(text):
    return Markup(markdown(text))

@app.template_filter('intcomma')
def intcomma(value):
    return "{:,}".format(value)

@app.template_filter('round_and_comma')
def round_and_comma(value, decimals=2):
    try:
        value = float(value)
        value = round(value, decimals)
    except (ValueError, TypeError):
        return value
    return "{:,}".format(value)

# Ваши API-ключи для Finnhub
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Установка API ключа OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")

@app.route('/register', methods=['POST'])
def register():
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirmPassword']

    if password != confirm_password:
        return jsonify({'status': 'error', 'message': 'Passwords do not match!'})

    if user_exists(email):
        return jsonify({'status': 'error', 'message': 'User already exists!'})

    register_user(email, password)

    user = login_user(email, password)
    session['user_id'] = user.id
    session['user_email'] = user.email

    return jsonify({'status': 'success', 'message': 'User registered successfully!'})

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = login_user(email, password)

        if user:
            session['user_id'] = user.id
            session['user_email'] = user.email
            return redirect(url_for('index_page'))
        else:
            flash('Invalid credentials')
            return redirect(url_for('login_page'))

    return render_template('login.html')

# Декоратор для защиты маршрутов, требующих авторизации
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to be logged in to access this page.')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# Выход пользователя
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_email', None)
    return redirect(url_for('welcome'))

@app.route('/')
def welcome():
    user_email = session.get('user_email')
    return render_template('welcome.html', user=user_email)

@app.route('/contact', methods=['POST'])
def contact():
    name = request.form['name']
    country = request.form['country']
    email = request.form['email']
    message = request.form['message']

    # Добавьте здесь логику для обработки сообщения
    # Например, сохранить данные в базу данных или отправить их по электронной почте

    flash('Your message has been sent successfully!', 'success')
    return redirect(url_for('welcome'))

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/privacypolicy')
def privacypolicy():
    return render_template('privacypolicy.html')

@app.route('/cookiepolicy')
def cookiepolicy():
    return render_template('cookiepolicy.html')

@app.route('/termsofuse')
def terms_of_use():
    return render_template('termsofuse.html')

@app.route('/index')
@login_required
def index_page():
    return render_template('index.html')

@app.route('/registration')
def registration_page():
    return render_template('registration.html')

@app.route('/usersettings')
@login_required
def user_settings_page():
    user_email = session.get('user_email')
    if not user_email:
        flash('User email not found. Please log in.')
        return redirect(url_for('login_page'))

    user = User.query.filter_by(email=user_email).first()

    return render_template('usersettings.html', email=user_email, user=user)

@app.route('/save_user_settings', methods=['POST'])
@login_required
def save_user_settings():
    user_id = session['user_id']
    full_name = request.form['fullName']
    address = request.form['address']
    country = request.form['country']
    city = request.form['city']
    postal_code = request.form['postalCode']
    phone = request.form['phone']
    birthdate = request.form['birthdate']

    user = User.query.get(user_id)
    user.full_name = full_name
    user.address = address
    user.country = country
    user.city = city
    user.postal_code = postal_code
    user.phone = phone
    user.birthdate = birthdate

    db.session.commit()

    return 'Settings saved successfully', 200

def get_risk_free_rate():
    try:
        stock = yf.Ticker("^TNX")
        data = stock.history(period="1d")
        if data.empty:
            return "No data available"
        latest_data = data.iloc[-1]
        risk_free_rate = latest_data['Close'] / 100
        return risk_free_rate
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return "Error"

def get_stock_volatility(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        log_returns = np.log(1 + hist['Close'].pct_change())
        volatility = np.std(log_returns) * np.sqrt(252)
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

def format_financial_data(financial_data):
    formatted_data = {}
    for key, value in financial_data.items():
        df = pd.DataFrame(value).T
        df.index = pd.to_datetime(df.index).date
        df = df.applymap(lambda x: "{:,.2f}".format(x / 1000) if pd.notnull(x) and isinstance(x, (int, float)) else x)
        df = df[df.index > pd.to_datetime('2019-12-31').date()]
        formatted_data[key] = df.T
    return formatted_data

def analyze_financials(financial_data):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"""
    Analyze the following financial data and provide a structured summary for each section. Use the exact numbers and changes in percentages. The summary should cover:
    - Key trends over the past four fiscal years.
    - Income Statement analysis. Use exact numbers from financial data and percentage changes. It should be beautifully written. Pretend to be best financial/investment analyst in the whole world.
    - Balance sheet analysis. Use exact numbers from financial data and percentage changes. It should be beautifully written. Pretend to be best financial/investment analyst in the whole world.
    - Cash flow statement analysis. Use exact numbers from financial data and percentage changes. It should be beautifully written. Pretend to be best financial/investment analyst in the whole world.
    - Margins and ratios. Use exact numbers from financial data and percentage changes. It should be beautifully written. Pretend to be best financial/investment analyst in the whole world.
    - Concluding statement on financial health. From written text above.
    Here is the financial data for analysis:
    {financial_data}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    analysis = response.choices[0].message.content.strip()
    formatted_analysis = analysis.replace(" - ", "\n- ")
    return formatted_analysis

def calculate_volatility(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        log_returns = np.log(1 + hist['Close'].pct_change())
        hist_volatility = np.std(log_returns) * np.sqrt(252)
        return hist_volatility
    except Exception as e:
        print(f"Error calculating volatility: {e}")
        return None

def analyze_volatility(symbol, hist_volatility):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Analyze the following volatility data for company {symbol}. Historical volatility: {hist_volatility:.2f}. Interpret this data and explain what it means for investors."
    response = openai.ChatCompletion.create(
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
    rounded_var = round(var_value, 2)
    prompt = f"""
    Analyze the following risk data for company {symbol}. Value at Risk (VaR): **{rounded_var}**. Please provide an interpretation and explain what it means for investors.
    The output should have:
    - Separate paragraphs for "Interpretation of VaR", "Operational Risks", and "Interest Rate Risks".
    - Each paragraph should be well-structured with bold headers and easy to read.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    analysis = response.choices[0].message.content.strip()
    formatted_analysis = f"""
**Value at Risk (VaR):** **{rounded_var}**
**Interpretation of VaR:** 
The Value at Risk (VaR) of **{rounded_var}** implies that there is a 95% confidence level that the maximum potential loss for company {symbol}'s portfolio is $0.36 for every dollar invested within a specific time period. In other words, there is a 5% chance that losses could exceed **{rounded_var}**.
**Operational Risks:**
Operational risks refer to the potential losses a company may face due to internal processes, systems, or human errors. These risks could include supply chain disruptions, cybersecurity threats, regulatory compliance issues, or technology failures. A VaR of **{rounded_var}** indicates that there is a 5% chance that operational risks could lead to losses exceeding **{rounded_var}** for every dollar invested in company {symbol}.
**Interest Rate Risks:**
Interest rate risks relate to the potential impact of interest rate changes on a company's financial position. This risk is particularly relevant for companies like {symbol} that may have exposure to interest rate-sensitive instruments such as debt or derivatives. A VaR of **{rounded_var}** suggests that there is a 5% probability that fluctuations in interest rates could lead to losses exceeding **{rounded_var}** per dollar invested in {symbol}.
By considering the VaR value along with specific risk categories, investors can assess the potential downside risk associated with investing in company {symbol} and make informed decisions to manage their overall portfolio risk effectively.
    """
    return formatted_analysis

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

@app.route('/company')
@login_required
def company():
    symbol = request.args.get('symbol')
    stock_data = get_stock_data(symbol)
    if not stock_data:
        return "Error fetching stock data"
    financials = get_financials(symbol)
    formatted_financials = format_financial_data(financials)
    financial_analysis = analyze_financials(formatted_financials)
    historical_data = stock_data['historical_prices']
    price_chart = plot_historical_prices(historical_data)
    hist_volatility = calculate_volatility(symbol)
    formatted_hist_volatility = f"{hist_volatility:.2f}" if hist_volatility is not None else "N/A"
    volatility_analysis = analyze_volatility(symbol, hist_volatility)
    var_value = calculate_var(symbol)
    risk_analysis = analyze_risk(symbol, var_value)
    news = get_news(symbol)
    news_analysis = analyze_news(news)
    company_type = get_company_type(symbol)
    valuation = None
    if company_type == 'Non-Financial':
        valuation = perform_dcf_analysis(symbol)
    investment_opinion = get_investment_opinion(
        financial_analysis,
        volatility_analysis,
        risk_analysis,
        news_analysis,
        valuation
    )
    return render_template('company.html',
                           company_info=stock_data['company_info'],
                           historical_data=historical_data,
                           news=news,
                           financials=formatted_financials,
                           financial_analysis=financial_analysis,
                           price_chart=price_chart,
                           symbol=symbol,
                           risk_free_rate=get_risk_free_rate(),
                           volatility=get_stock_volatility(symbol),
                           hist_volatility=formatted_hist_volatility,
                           volatility_analysis=volatility_analysis,
                           risk={'var': var_value},
                           risk_analysis=risk_analysis,
                           news_analysis=news_analysis,
                           investment_opinion=investment_opinion,
                           valuation=valuation,
                           call_price=None,
                           forecast=None,
                           active_tab="ai-analysis")

@app.route('/calculate_black_scholes', methods=['POST'])
def calculate_black_scholes():
    S = float(request.form['S'])
    K = float(request.form['K'])
    T = float(request.form['T'])
    r = get_risk_free_rate()
    if r == "Error":
        return "Error fetching risk-free rate"
    sigma = float(request.form['sigma'])
    symbol = request.form['symbol']
    call_price = black_scholes_call(S, K, T, r, sigma)
    stock_data = get_stock_data(symbol)
    if not stock_data:
        return "Error fetching stock data"
    historical_data = stock_data['historical_prices']
    news = get_news(symbol)
    financials = get_financials(symbol)
    formatted_financials = format_financial_data(financials)
    financial_analysis = analyze_financials(formatted_financials)
    price_chart = plot_historical_prices(historical_data)
    hist_volatility = calculate_volatility(symbol)
    formatted_hist_volatility = f"{hist_volatility:.2f}" if hist_volatility is not None else "N/A"
    volatility_analysis = analyze_volatility(symbol, hist_volatility)
    var_value = calculate_var(symbol)
    risk_analysis = analyze_risk(symbol, var_value)
    news_analysis = analyze_news(news)
    investment_opinion = get_investment_opinion(financial_analysis, volatility_analysis, risk_analysis, news_analysis, valuation)
    return render_template('company.html',
                           company_info=stock_data['company_info'],
                           historical_data=historical_data,
                           news=news,
                           financials=formatted_financials,
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
    financials = get_financials(symbol)
    formatted_financials = format_financial_data(financials)
    price_chart = plot_historical_prices(historical_data)
    risk_free_rate = get_risk_free_rate()
    volatility = get_stock_volatility(symbol)
    return render_template('company.html',
                           company_info=stock_data['company_info'],
                           historical_data=historical_data,
                           news=news,
                           financials=formatted_financials,
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
    financials = get_financials(symbol)
    formatted_financials = format_financial_data(financials)
    price_chart = plot_historical_prices(historical_data)
    risk_free_rate = get_risk_free_rate()
    volatility = get_stock_volatility(symbol)
    return render_template('company.html',
                           company_info=stock_data['company_info'],
                           historical_data=historical_data,
                           news=news,
                           financials=formatted_financials,
                           price_chart=price_chart,
                           forecast=forecast,
                           symbol=symbol,
                           risk_free_rate=risk_free_rate,
                           volatility=volatility,
                           active_tab="mathematical-analysis")

if __name__ == '__main__':
    app.run(debug=True)
