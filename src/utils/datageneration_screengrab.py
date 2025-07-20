import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import uuid
from datetime import datetime, timedelta
import csv   
from dash import Dash, html, dcc, Input, Output, callback
from mss import mss
import uuid 
from PIL import Image 
from colorama import Fore 
import sys 


def generate_data(stick:int) -> None: 
    if stick == 0: 
        base_price = np.random.randint(20, high=500)
        prices = []
        for x in range(20): 
            open_price = np.random.normal(1, 0.05)*base_price
            close_price = np.random.normal(1, 0.05)*base_price
            high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
            low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
            
            prices.append([open_price, high_price,low_price, close_price]) 
            print(open_price, high_price,low_price, close_price)
            base_price = round(np.random.normal(close_price,0.00025))

    # Doji
    elif stick == 1: 
        base_price = np.random.randint(20, high=500)
        prices = []
        for x in range(19): 
            open_price = np.random.normal(1, 0.05)*base_price
            close_price = np.random.normal(1, 0.05)*base_price
            high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
            low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
            
            prices.append([open_price, high_price,low_price, close_price]) 
            print(open_price, high_price,low_price, close_price)
            base_price = round(np.random.normal(close_price,0.00025))
        # Open and close ultra close 
        high_price = round(max(np.random.randint(100,high=105)*open_price/100, np.random.randint(100,high=105)*close_price/100),2)
        low_price = round(min(np.random.randint(95,high=100)*open_price/100, np.random.randint(95,high=100)*close_price/100), 2)
        open_price = np.random.normal(1, 0.00005)*base_price
        close_price = np.random.normal(1, 0.00005)*base_price
        prices.append([open_price, high_price,low_price, close_price])
        
    # Bullish engulfing - DONE
    elif stick == 2: 
        base_price = np.random.randint(20, high=500)
        prices = []
        for x in range(18): 
            open_price = np.random.normal(1, 0.05)*base_price
            close_price = np.random.normal(1, 0.05)*base_price
            high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
            low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
            
            prices.append([open_price, high_price,low_price, close_price]) 
            print(open_price, high_price,low_price, close_price)
            base_price = round(np.random.normal(close_price,0.00025))

        # Stick 1
        open_price = np.random.normal(1.0, 0.00025)*base_price
        close_price = np.random.normal(.95, 0.00005)*base_price
        high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
        low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
        prices.append([open_price, high_price,low_price, close_price])

        # Stick 2 
        open_price = min(open_price, close_price, high_price, low_price) 
        close_price = np.random.normal(1.05, 0.005)*base_price
        
        high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
        low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
        prices.append([open_price, high_price,low_price, close_price])


    # Bearing engulfing - DONE 
    elif stick == 3: 
        base_price = np.random.randint(20, high=500)
        prices = []
        for x in range(18): 
            open_price = np.random.normal(1, 0.05)*base_price
            close_price = np.random.normal(1, 0.05)*base_price
            high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
            low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
            
            prices.append([open_price, high_price,low_price, close_price]) 
            print(open_price, high_price,low_price, close_price)
            base_price = round(np.random.normal(close_price,0.00025))

        # Stick 1
        open_price = np.random.normal(1.00, 0.00025)*base_price
        close_price = np.random.normal(1.05, 0.00005)*base_price
        high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
        low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
        prices.append([open_price, high_price,low_price, close_price])

        # Stick 2 
        open_price = max(open_price, close_price, high_price, low_price) 
        close_price = np.random.normal(.95, 0.005)*base_price
        
        high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
        low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
        prices.append([open_price, high_price,low_price, close_price])

    # Morning star
    elif stick == 4: 
        base_price = np.random.randint(20, high=500)
        prices = []
        for x in range(18): 
            open_price = np.random.normal(1, 0.05)*base_price
            close_price = np.random.normal(1, 0.05)*base_price
            high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
            low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
            
            prices.append([open_price, high_price,low_price, close_price]) 
            print(open_price, high_price,low_price, close_price)
            base_price = round(np.random.normal(close_price,0.00025))

        # Stick 1
        open_price = close_price * np.random.normal(.95, 0.005)
        close_price = np.random.normal(.95, 0.00005)*open_price
        high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
        low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
        prices.append([open_price, high_price,low_price, close_price])
        
        # Stick 2 
        open_price = min(open_price, close_price, high_price, low_price) * np.random.normal(.995, 0.005)
        close_price = max(close_price * np.random.normal(1.00, 0.005), open_price+0.0001) 
        high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
        low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
        prices.append([open_price, high_price,low_price, close_price])

        # Stick 3
        open_price = max(np.random.normal(1.00, 0.05)*close_price, open_price*1.01)
        close_price = max(np.random.normal(1.05, 0.00005)*open_price, close_price) 
        high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
        low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
        prices.append([open_price, high_price,low_price, close_price])


    # Evening star
    elif stick == 5: 
        base_price = np.random.randint(20, high=500)
        prices = []
        for x in range(18): 
            open_price = np.random.normal(1, 0.05)*base_price
            close_price = np.random.normal(1, 0.05)*base_price
            high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
            low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
            
            prices.append([open_price, high_price,low_price, close_price]) 
            print(open_price, high_price,low_price, close_price)
            base_price = round(np.random.normal(close_price,0.00025))

        # Stick 1
        open_price = close_price * np.random.normal(.95, 0.005)
        close_price = np.random.normal(1.05, 0.00005)*open_price
        high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
        low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
        prices.append([open_price, high_price,low_price, close_price])
        
        # Stick 2 
        open_price = max(open_price, close_price, high_price, low_price) * np.random.normal(1.005, 0.005)
        close_price = min(close_price * np.random.normal(1.00, 0.005), open_price-0.0001)
        high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
        low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
        prices.append([open_price, high_price,low_price, close_price])

        # Stick 3
        open_price = min(np.random.normal(1.00, 0.05)*close_price, open_price*.99) 
        close_price = min(np.random.normal(.95, 0.00005)*open_price, close_price) 
        high_price = round(max(np.random.randint(100,high=103)*open_price/100, np.random.randint(100,high=103)*close_price/100),2)
        low_price = round(min(np.random.randint(97,high=100)*open_price/100, np.random.randint(97,high=100)*close_price/100), 2)
        prices.append([open_price, high_price,low_price, close_price])

    dates = [datetime.today() - timedelta(minutes=x) for x in range(20)]
    prices.reverse()
    prices = np.array(prices)

    return dates, prices 
    
if __name__ == "__main__": 
    stick = 1
    dates, data = generate_data(stick)

    app = Dash()
    fig = go.Figure(data=[go.Candlestick(x=dates,
        open=data[:,0],
        high=data[:,1],
        low=data[:,2],
        close=data[:,3])])
    fig.update_layout(height=960, margin_l=200, margin_r=200, margin_b=200, margin_t=200)

    app.layout = html.Div([
        dcc.Graph(figure=fig, id='candlestick-graph'), 
        dcc.Interval(
            id='interval-component',
            interval=0.5*1000, # in milliseconds
            n_intervals=0
        )
    ])

    @callback(Output('candlestick-graph', 'figure'), Input('interval-component','n_intervals'))
    def update_candles(n): 
        global stick
        if stick == 6: 
            stick = 1
        print(Fore.LIGHTYELLOW_EX + str(stick) + Fore.RESET)
        dates, data = generate_data(stick)

        fig = go.Figure(data=[go.Candlestick(x=dates,
            open=data[:,0],
            high=data[:,1],
            low=data[:,2],
            close=data[:,3])])
        fig.update_layout(height=960, margin_l=200, margin_r=200, margin_b=200, margin_t=200)
            # Take screenshot BEFORE generating new data (captures previous state correctly)
        if n > 0:  # Skip first iteration
            try: 
                with mss() as sct: 
                    sct_image = sct.grab(sct.monitors[0])
                    raw_image = Image.frombytes("RGB", sct_image.size, sct_image.rgb)
                    raw_image = raw_image.crop((0,170,3840, 2160))
                    raw_image = raw_image.resize((700,500))
                    image_name = f'{uuid.uuid1()}.png'
                    raw_image.save(f'test_cap_train/{image_name}') 
                    with open('test_cap_train/labels.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([image_name, 5 if stick == 1 else stick-1, 'screengrabs'])  # Previous stick
            except Exception as e: 
                print(e) 

        stick += 1 

        return fig 

    app.run(debug=True, use_reloader=False)