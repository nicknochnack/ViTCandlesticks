import yfinance as yf 
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from colorama import Fore 
from dash import Dash, html, dcc, Input, Output, callback
import time 
import uuid

def plot_candles(stock_ticker:str) -> None: 
    # Specify stock ticker 
    dat = yf.Ticker(stock_ticker)
    # Get historical prices
    historical_prices = dat.history(period='1d',interval='1m')
    return historical_prices

if __name__ == "__main__": 
    df = plot_candles("AMZN").reset_index()

    fig = go.Figure(data=[go.Candlestick(x=df['Datetime'],
        open=[0],
        high=[0],
        low=[0],
        close=[0])])
    fig.update_layout(height=960, margin_l=200, margin_r=200, margin_b=200, margin_t=200)


    app = Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig, id='candlestick-graph'), 
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )
    ])

    @callback(Output('candlestick-graph', 'figure'), Input('interval-component','n_intervals'))
    def update_candles(n): 
        if n > len(df):
            data = df.iloc[len(df)-20:len(df)] 
        elif n > 20: 
            data = df.iloc[n-20:n] 
        else: 
            data = df.iloc[:n]

        fig = go.Figure(data=[go.Candlestick(x=data['Datetime'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'])])
        fig.update_layout(height=960, margin_l=200, margin_r=200, margin_b=200, margin_t=200)

        return fig 

    app.run(debug=True, use_reloader=False)


    
