import yfinance as yf
import numpy as np
from datetime import datetime

def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    financials = stock.financials.T
    cashflow = stock.cashflow.T
    quarterly_financials = stock.quarterly_financials.T
    quarterly_cashflow = stock.quarterly_cashflow.T
    balance_sheet = stock.balance_sheet.T
    info = stock.info

    return financials, cashflow, quarterly_financials, quarterly_cashflow, balance_sheet, info

def calculate_cagr(start_value, end_value, periods):
    cagr = (end_value / start_value) ** (1 / periods) - 1
    return cagr

def calculate_average_margin(fcf_data, revenue_data):
    margins = [fcf / revenue if revenue != 0 else 0 for fcf, revenue in zip(fcf_data, revenue_data)]
    average_margin = np.mean(margins)
    return average_margin

def calculate_ttm(quarterly_data):
    return quarterly_data.iloc[:4].sum()

def forecast_revenue(revenue_data, periods=5):
    start_value = revenue_data[-1]
    end_value = revenue_data[0]
    cagr = (end_value / start_value) ** (1 / periods) - 1
    forecasted_revenue = [end_value * (1 + cagr) ** i for i in range(1, periods + 1)]
    return forecasted_revenue, cagr

def forecast_fcf(forecasted_revenue, average_margin):
    forecasted_fcf = [revenue * average_margin for revenue in forecasted_revenue]
    return forecasted_fcf

def get_wacc(balance_sheet, financials, beta, risk_free_rate, market_return, tax_rate):
    total_debt = balance_sheet['Total Debt'].iloc[0]
    possible_equity_names = ['Total Stockholder Equity', 'Stockholders Equity', 'Total Equity']
    total_equity = None
    for name in possible_equity_names:
        if name in balance_sheet.columns:
            total_equity = balance_sheet[name].iloc[0]
            break

    if total_equity is None:
        raise ValueError("Could not find the 'Total Equity' column in the balance sheet.")

    total_value = total_debt + total_equity
    Wd = total_debt / total_value
    We = total_equity / total_value

    interest_expense = financials['Interest Expense'].iloc[0]
    Kd = interest_expense / total_debt
    Ke = risk_free_rate + beta * (market_return - risk_free_rate)

    WACC = Wd * Kd * (1 - tax_rate) + We * Ke
    return WACC

def calculate_pv_of_fcf(forecasted_fcf, WACC):
    pv_of_fcf = [fcf / (1 + WACC) ** i for i, fcf in enumerate(forecasted_fcf, start=1)]
    return pv_of_fcf

def calculate_terminal_value(fcf, WACC, g=0.03):
    terminal_value = fcf * (1 + g) / (WACC - g)
    return terminal_value

def perform_dcf_analysis(ticker):
    financials, cashflow, quarterly_financials, quarterly_cashflow, balance_sheet, info = get_financial_data(ticker)

    revenue_data = financials['Total Revenue'].values[-4:].tolist()
    revenue_data = [float(revenue) for revenue in revenue_data]
    dates = list(financials.index[-4:])

    fcf_data = cashflow['Free Cash Flow'].sort_index(ascending=False).iloc[:4].tolist()
    fcf_data = [float(fcf) for fcf in fcf_data]

    ttm_revenue = calculate_ttm(quarterly_financials['Total Revenue'])
    ttm_fcf = calculate_ttm(quarterly_cashflow['Free Cash Flow'])

    revenue_data.insert(0, ttm_revenue)
    fcf_data.insert(0, ttm_fcf)
    dates.insert(0, 'TTM')

    average_margin = calculate_average_margin(fcf_data, revenue_data)

    forecasted_revenue, cagr = forecast_revenue(revenue_data)

    forecasted_fcf = forecast_fcf(forecasted_revenue, average_margin)

    beta = info['beta']
    risk_free_rate = yf.Ticker('^TNX').history(period='1d')['Close'].iloc[0] / 100
    current_year = datetime.now().year
    previous_year = current_year - 1
    sp500 = yf.Ticker('^GSPC')
    market_return = sp500.history(start=f'{previous_year}-01-01', end=f'{previous_year}-12-31')['Close'].pct_change().mean() * 252
    tax_rate = financials['Tax Provision'].iloc[-1] / financials['Pretax Income'].iloc[-1]

    WACC = get_wacc(balance_sheet, financials, beta, risk_free_rate, market_return, tax_rate)

    pv_of_fcf = calculate_pv_of_fcf(forecasted_fcf, WACC)

    terminal_value = calculate_terminal_value(forecasted_fcf[-1], WACC)
    pv_of_terminal_value = terminal_value / (1 + WACC) ** len(forecasted_fcf)

    total_value = sum(pv_of_fcf) + pv_of_terminal_value

    shares_outstanding = info['sharesOutstanding']
    fair_value_per_share = total_value / shares_outstanding

    possible_price_keys = ['regularMarketPrice', 'currentPrice', 'previousClose']
    current_price = None
    for key in possible_price_keys:
        if key in info:
            current_price = info[key]
            break

    if current_price is None:
        raise ValueError("Could not find the current price in the info data.")

    overvaluation = ((current_price - fair_value_per_share) / fair_value_per_share) * 100

    valuation_data = {
        "forecasted_revenue": forecasted_revenue,
        "forecasted_fcf": forecasted_fcf,
        "pv_of_fcf": pv_of_fcf,
        "terminal_value": terminal_value,
        "pv_of_terminal_value": pv_of_terminal_value,
        "total_value": total_value,
        "shares_outstanding": shares_outstanding,
        "fair_value_per_share": fair_value_per_share,
        "current_price": current_price,
        "overvaluation": overvaluation,
        "wacc": WACC
    }

    return valuation_data
