import os
import pickle
import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta
import concurrent.futures

# Python libraries for yahoo
import yfinance as yf
from yahoo_fin import stock_info as si

# Import libraries for stock info
import ta
from ta import add_all_ta_features
from ta.utils import dropna
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, ADXIndicator

# Python library for fetching news
from newsapi import NewsApiClient

# Python dash
import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

os.environ["YFINANCE_CACHE_ENABLED"] = "False"

# Setting the style color
colors = {"background": "#000000", "text": "#ffFFFF"}

external_stylesheets = [dbc.themes.SLATE]

# Cache file for S&P500 and NASDAQ
CACHE_SP500 = 'sp500_cache.pkl'
CACHE_NASDAQ = 'nasdaq_cache.pkl'
CACHE_DOWJONES = 'dowjones_cache.pkl'
CACHE_EXPIRATION = timedelta(hours=24)

# News API key
newsapi = NewsApiClient(api_key='37de6347c7274cafa5687a539fae4950')

# Importing the csv file and getting all the symbols
df = pd.read_csv('./data/NYSE.csv')
symbols = df['Symbol']

# Removing the exponent symbol from the element and replacing it with a hyphen
symbols = symbols.tolist()
for num in range(len(symbols)):
    symbols[num] = str(symbols[num]).replace('^', '-').replace('/', '-')

def fetch_stock_data_yq(symbol, start_date, end_date):
    symbol = symbol.replace('/','-')
    ticker = yf.Ticker(symbol)
    info = ticker.history(start=start_date, end=end_date)
    info.reset_index(inplace=True)
    info.rename(columns={'Date': 'date', 'Close': 'close', 'Open': 'Open', 'High': 'High', 'Low': 'Low'}, inplace=True)
    return info

def generate_trends(data, window_ratio=1.0/3.0, should_plot=True):

    x = np.array(data)
    x_length = len(x)
    window = int(window_ratio * x_length)

    max_idx = np.argmax(x)
    min_idx = np.argmin(x)

    if max_idx + window > x_length:
        second_max = max(x[0:(max_idx - window)])
    else:
        second_max = max(x[(max_idx + window):])

    if min_idx - window < 0:
        second_min = min(x[(min_idx + window):])
    else:
        second_min = min(x[0:(min_idx - window)])

    second_max_idx = np.where(x == second_max)[0][0]
    second_min_idx = np.where(x == second_min)[0][0]

    max_slope = (x[max_idx] - x[second_max_idx]) / (max_idx - second_max_idx)
    min_slope = (x[min_idx] - x[second_min_idx]) / (min_idx - second_min_idx)
    max_intercept = x[max_idx] - (max_slope * max_idx)
    min_intercept = x[min_idx] - (min_slope * min_idx)
    max_end = x[max_idx] + (max_slope * (x_length - max_idx))
    min_end = x[min_idx] + (min_slope * (x_length - min_idx))
    max_line = np.linspace(max_intercept, max_end, x_length)
    min_line = np.linspace(min_intercept, min_end, x_length)

    trend_data = np.transpose(np.array((x, max_line, min_line)))

    trend_data = pd.DataFrame(trend_data, index=np.arange(0, len(x)),
                              columns=['Data', 'Resistance', 'Support'])

    return trend_data, max_slope, min_slope


def identify_extrema(data, window_ratio=1.0/3):

    x = np.array(data)
    x_length = len(x)

    if window_ratio < 1:
        window = int(window_ratio * x_length)

    signals = np.zeros(x_length, dtype=float)

    i = window

    while i != x_length:
        if x[i] > max(x[i-window:i]): signals[i] = 1
        elif x[i] < min(x[i-window:i]): signals[i] = -1
        i += 1

    return signals

def segment_trends(data, num_segments=2):
    y = np.array(data)

    num_segments = int(num_segments)
    max_values = np.ones(num_segments)
    min_values = np.ones(num_segments)
    segment_size = int(len(y)/num_segments)
    for i in range(1, num_segments+1):
        idx2 = i*segment_size
        idx1 = idx2 - segment_size
        max_values[i-1] = max(y[idx1:idx2])
        min_values[i-1] = min(y[idx1:idx2])

    x_max_values = np.ones(num_segments)
    x_min_values = np.ones(num_segments)
    
    for i in range(0, num_segments):
        x_max_values[i] = np.where(y == max_values[i])[0][0]
        x_min_values[i] = np.where(y == min_values[i])[0][0]

    return x_max_values, max_values, x_min_values, min_values

def get_sector(symbol):
    try:
        stk = yf.Ticker(symbol)
        stk_info = stk.info
        if 'sector' in stk_info:
            return symbol
    except Exception as err:
        print(err)
        return None

def process_symbol(symbol):
    try:
        # Get stock data
        stk = yf.Ticker(symbol)
        stk_info = stk.info

        # Retrieve sector information
        sector = stk_info['sector']

        # Get historical stock prices for 2 days
        stock_history = stk.history('2d')

        # Compute the stock price change
        price_change = (stock_history['Close'][1] - stock_history['Close'][0]) / stock_history['Close'][0]

        # Calculate market capitalization
        market_capitalization = stk_info['sharesOutstanding'] * stk_info['previousClose']

        return {
            "ticker": symbol,
            "sector": sector,
            "delta": price_change,
            "market_cap": market_capitalization,
        }
    except Exception as err:
        print(err)
        return None

def add_all_ta_features_with_custom_adx_window(df, high, low, close, volume):
    
    # Momentum indicators
    df = ta.add_momentum_ta(df, high=high, low=low, close=close, volume=volume)
    
    # Volume indicators
    df = ta.add_volume_ta(df, high=high, low=low, close=close, volume=volume)
    
    # Volatility indicators
    df = ta.add_volatility_ta(df, high=high, low=low, close=close)
    
    # Trend indicators except for ADX
    df = ta.add_trend_ta(df, high=high, low=low, close=close)

    # Calculate the custom ADX window size based on the number of data points
    adx_window_size = min(14, len(df) - 1)
    
    # Recalculate the ADX with the custom window size
    adx_indicator = ADXIndicator(df['High'], df['Low'], df['close'], window=adx_window_size)
    df['trend_adx'] = adx_indicator.adx()
    
    return df

# Adding a function for chunk
def chunks(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# Adding function for CACHE
def load_cache(file_name):
    if not os.path.exists(file_name):
        return None

    with open(file_name, 'rb') as file:
        cache = pickle.load(file)
    
    if datetime.now() - cache['timestamp'] > CACHE_EXPIRATION:
        return None

    return cache['data']

# Saving CACHE file
def save_cache(file_name, data):
    with open(file_name, 'wb') as file:
        pickle.dump({'timestamp': datetime.now(), 'data': data}, file)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": colors["background"], "color": colors["text"]},
    children=[   
    html.Br(),
    html.Br(),
    html.H1('NYSE Stock Visualizations', id = 'main_title', style={'textAlign': 'center'}),
    html.Br(),

    html.Div([
        dbc.Col(
            dcc.Dropdown(
                id='stock_dropdown',
                options=[{'label': s, 'value': s} for s in symbols],
                value='AAPL',
                style={
                    "marginRight": "150px",
                    "color": "black"
                },
                multi=False
            ),
            width = {"size": 3}
        ),

        dcc.DatePickerRange(
            id='date_picker_range',
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
            style={
                "marginLeft": "150px",
                "color": colors["text"],
                "backgroundColor": colors["background"],
            }
        ),
    ], style={
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        "color": colors["text"],
        "backgroundColor": colors["background"],
    }),

    html.Br(),

    dcc.Tabs(
    children=[
        dcc.Tab(label="Technicals", style={"backgroundColor": colors["background"], "color": colors["text"]}, 
                selected_style={"backgroundColor": "#1D7874", "color": "#F4C095"},
                children=[
                        dcc.Graph(
                            id='stock_graph',
                            style={
                                "color": colors["text"],
                                "backgroundColor": colors["background"],
                                "marginRight": "30px",
                                "marginLeft": "30px",
                            }
                        ),

                        html.Br(),

                        html.Div(
                            id='output_container',
                            style={
                                "color": colors["text"],
                                "backgroundColor": colors["background"],
                            }
                        ),

                        html.Br(),

                        html.Div([
                            html.Button("Toggle Analysis", id="toggle-analysis", n_clicks=0, style={"marginRight": "10px", "backgroundColor": "#000000", "color": "#FFFFFF"}),
                            html.Button("Moving Average", id="moving-average", n_clicks=0, style={"marginLeft": "10px", "backgroundColor": "#000000", "color": "#FFFFFF"}),
                        ], style={
                            'display': 'flex',
                            'justifyContent': 'center',
                            'alignItems': 'center',
                            
                        }),

                        html.Br(),
                        html.Br(),
                        html.Br(),

                        html.H2('Adanced Technical Graph with news update', style={'textAlign': 'center'}),
                        html.Br(),
                        html.Div([  
                            dcc.Graph(id='advanced_technical_graph', style={"marginLeft": "25px", "marginRight": "25px"}),
                            html.Div(id='news_feed', style={"marginLeft": "25px", "marginRight": "25px"}),
                        ], style={
                            'display': 'flex',
                            'justifyContent': 'center',
                            'alignItems': 'center',
                        }),

                        html.Br(),
                        html.Br(),
                        html.Br(),

                        html.H2('Dividends', style={'textAlign': 'center'}),
                        html.Br(),
                        html.Div(id='stock_dividends',
                                style={
                                    "marginRight": "30px",
                                    "marginLeft": "30px",
                                }),

                        html.Br(),
                        html.Br(),
        ]),
        dcc.Tab(label="Heatmap", style={"backgroundColor": colors["background"], "color": colors["text"]}, 
                selected_style={"backgroundColor": "#1D7874", "color": "#F4C095"},
                children=[
                        html.H2('S&P500 Heatmap', style={'textAlign': 'center'}),
                        html.Br(),
                        dcc.Loading(
                            id="loading_1",
                            type="cube",
                            children=[
                                dcc.Graph(
                                    id='sp500_heatmap',
                                    style={
                                        "marginRight": "30px",
                                        "marginLeft": "30px",
                                    }
                                )
                            ],
                        ),

                        html.Br(),
                        html.Br(),

                        html.H2('NASDAQ Heatmap', style={'textAlign': 'center'}),
                        html.Br(),
                        dcc.Loading(
                            id="loading_2",
                            type="cube",
                            children=[
                                dcc.Graph(
                                    id='nasdaq_heatmap',
                                    style={
                                        "marginRight": "30px",
                                        "marginLeft": "30px",
                                    }
                                )
                            ],
                        ),

                        html.Br(),
                        html.Br(),

                        html.H2('Dow-Jones Heatmap', style={'textAlign': 'center'}),
                        html.Br(),
                        dcc.Loading(
                            id="loading_3",
                            type="cube",
                            children=[
                                dcc.Graph(
                                    id='dowjones_heatmap',
                                    style={
                                        "marginRight": "30px",
                                        "marginLeft": "30px",
                                    }
                                )
                            ],
                        ),

                        html.Br(),
                        html.Br(),
        ]),

    ],
    style={
        "width": "50%",
        "margin": "0 auto",
    }
    ),
    
]
)

@app.callback(
    dash.dependencies.Output(component_id="stock_graph", component_property="figure"),
    dash.dependencies.Input(component_id="stock_dropdown", component_property="value"),
    dash.dependencies.Input(component_id="date_picker_range", component_property="start_date"),
    dash.dependencies.Input(component_id="date_picker_range", component_property="end_date"),
    dash.dependencies.Input(component_id="toggle-analysis", component_property="n_clicks"),
    dash.dependencies.Input(component_id="moving-average", component_property="n_clicks"),
)
def update_graph(stock, start_date, end_date, toggle_analysis_n_clicks, moving_average_n_clicks):

    if not stock:
        return "No stock selected"
    
    company_name = yf.Ticker(stock).info["longName"] if stock else "No stock selected"
    
    date_range = pd.to_datetime([start_date, end_date])
    selected_range = date_range[1] - date_range[0]
    num_days = selected_range.days

    if num_days <= 30:
        tick_format = "%d %b"
        tick_interval_price = "linear"
    elif num_days <= 180:
        tick_format = "%b %Y"
        tick_interval_price = "linear"
    else:
        tick_format = "%Y"
        tick_interval_price = "log"

    if not stock:
        return {}

    def parse_date(date_string):
        try:
            return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            try:
                return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                return datetime.strptime(date_string, '%Y-%m-%d')

    start_date_obj = parse_date(start_date)
    end_date_obj = parse_date(end_date)

    data = []
    df = fetch_stock_data_yq(stock, start_date_obj.strftime('%Y-%m-%d'), end_date_obj.strftime('%Y-%m-%d'))

    df = df.dropna(subset=['Open', 'High', 'Low', 'close', 'Volume'], how='all')

    if not df.empty:
        data.append(
            go.Candlestick(
                x=df['date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['close'],
                name=f'{stock} Price'
            )
        )

        data.append(
            go.Bar(
                x=df['date'],
                y=df['Volume'],
                name=f'{stock} Volume',
                marker_color='#B24D82',
                yaxis='y2'
            )
        )

        if toggle_analysis_n_clicks % 2 != 0:
            trends, slopeMax, slopeMin = generate_trends(df['close'])
            sigs = identify_extrema(df['close'])
            x_maxima, maxima, x_minima, minima = segment_trends(df['close'])

            data.append(
                go.Scatter(
                    x=df['date'],
                    y=trends['Resistance'],
                    name='Resistance',
                    mode='lines',
                    line=dict(color='red', width=1)
                )
            )

            data.append(
                go.Scatter(
                    x=df['date'],
                    y=trends['Support'],
                    name='Support',
                    mode='lines',
                    line=dict(color='#00FF73', width=1)
                )
            )

            for i, sig in enumerate(sigs):
                if sig == 1:
                    data.append(
                        go.Scatter(
                            x=[df.loc[i, 'date']],
                            y=[df.loc[i, 'close']],
                            mode='markers',
                            marker=dict(symbol='triangle-down', size=8, color='#FFC800'),
                            name='Sell'
                        )
                    )
                elif sig == -1:
                        data.append(
                            go.Scatter(
                            x=[df.loc[i, 'date']],
                            y=[df.loc[i, 'close']],
                            mode='markers',
                            marker=dict(symbol='triangle-up', size=8, color='#00FFFF'),
                            name='Buy'
                        )
                    )
            
            for i in range(len(x_maxima)):
                data.append(
                    go.Scatter(
                        x=[df.loc[int(x_maxima[i]), 'date']],
                        y=[maxima[i]],
                        mode='markers',
                        marker=dict(symbol='circle', size=8, color='blue'),
                        name='Maxima'
                    )
                )

            for i in range(len(x_minima)):
                data.append(
                    go.Scatter(
                        x=[df.loc[int(x_minima[i]), 'date']],
                        y=[minima[i]],
                        mode='markers',
                        marker=dict(symbol='circle', size=8, color='yellow'),
                        name='Minima'
                    )
                )
        
        if moving_average_n_clicks % 2 != 0:

            data.append(
                go.Scatter(
                    x=df['date'],
                    y=df['close'].rolling(window=50).mean(),
                    name='Moving Average 50',
                    mode='lines',
                    line=dict(color='#F00FAE', width=1)
                )
            )

            data.append(
                go.Scatter(
                    x=df['date'],
                    y=df['close'].rolling(window=100).mean(),
                    name='Moving Average 100',
                    mode='lines',
                    line=dict(color='#F0C10F', width=1)
                )
            )

            data.append(
                go.Scatter(
                    x=df['date'],
                    y=df['close'].rolling(window=200).mean(),
                    name='Moving Average 200',
                    mode='lines',
                    line=dict(color='#0FF051', width=1)
                )
            )
    
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    for trace in data:
        if isinstance(trace, go.Candlestick):
            fig.add_trace(trace, secondary_y=True)
        elif isinstance(trace, go.Bar):
            fig.add_trace(trace, secondary_y=False)
        else:
            fig.add_trace(trace, secondary_y=True)

    fig.update_layout(
        title='Stock Prices and Volume for '+company_name,
        xaxis={'title': 'Date',
           'rangeslider': {'visible': False},
           'rangeselector': {
               'buttons': [
                    {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                    {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                    {'count': 1, 'label': 'YTD', 'step': 'year', 'stepmode': 'backward'}
                ],
                'bgcolor': '#000000',
                'font': {'color': '#FFFFFF'},
            },
            'tickformat': "%b %Y",
            'showgrid': False
        },
        yaxis2={'title': 'Price', 'side': 'left', 'autorange': True, 'showgrid': False, 'zeroline': False},
        yaxis={'title': 'Volume', 'side': 'right', 'type': 'log', 'showgrid': False, 'autorange': True, 'zeroline': False},
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font=dict(color=colors["text"]),
        legend=dict(orientation='h', y=-0.2),
        height = 600,
    )

    fig.update_yaxes(tickmode="auto", nticks=10, type=tick_interval_price, secondary_y=True)

    return fig

@app.callback(
    dash.dependencies.Output(component_id='advanced_technical_graph', component_property='figure'),
    dash.dependencies.Input(component_id='stock_dropdown', component_property='value'),
    dash.dependencies.Input(component_id='date_picker_range', component_property='start_date'),
    dash.dependencies.Input(component_id='date_picker_range', component_property='end_date'),
)
def update_advanced_technical_graph(stock, start_date, end_date):
    
    if not stock:
        return "No stock selected"

    def parse_date(date_string):
        try:
            return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            try:
                return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                return datetime.strptime(date_string, '%Y-%m-%d')
            
    company_name = yf.Ticker(stock).info["longName"]

    start_date_obj = parse_date(start_date)
    end_date_obj = parse_date(end_date)

    df = fetch_stock_data_yq(stock, start_date_obj.strftime('%Y-%m-%d'), end_date_obj.strftime('%Y-%m-%d'))

    df = df.dropna(subset=['Open', 'High', 'Low', 'close', 'Volume'], how='all')

    # Calculate technical indicators
    df = add_all_ta_features_with_custom_adx_window(df, high="High", low="Low", close="close", volume="Volume")

    # Create traces for each indicator
    ema_trace = go.Scatter(x=df['date'], y=df['trend_ema_fast'], name="EMA")
    rsi_trace = go.Scatter(x=df['date'], y=df['momentum_rsi'], name="RSI", yaxis='y2')
    macd_trace = go.Scatter(x=df['date'], y=df['trend_macd'], name="MACD", yaxis='y3')
    atr_trace = go.Scatter(x=df['date'], y=df['volatility_atr'], name="ATR", yaxis='y4')
    cci_trace = go.Scatter(x=df['date'], y=df['trend_cci'], name="CCI", yaxis='y5')
    cmf_trace = go.Scatter(x=df['date'], y=df['volume_cmf'], name="CMF", yaxis='y6')

    # Create the figure and add traces
    fig = go.Figure()
    fig.add_trace(ema_trace)
    fig.add_trace(rsi_trace)
    fig.add_trace(macd_trace)
    fig.add_trace(atr_trace)
    fig.add_trace(cci_trace)
    fig.add_trace(cmf_trace)

    # Update the layout
    fig.update_layout(
        title='Advanced Technical Graph for '+company_name,
        xaxis=dict(domain=[0, 1], title="Date", showgrid=False, zeroline=False),
        yaxis=dict(title="EMA", showgrid=False, zeroline=False),
        yaxis2=dict(title="RSI", anchor="x", overlaying="y", side="right", showgrid=False, zeroline=False),
        yaxis3=dict(title="MACD", anchor="free", overlaying="y", side="right", position=0.85, showgrid=False, zeroline=False),
        yaxis4=dict(title="ATR", anchor="free", overlaying="y", side="right", position=0.75, showgrid=False, zeroline=False),
        yaxis5=dict(title="CCI", anchor="free", overlaying="y", side="right", position=0.65, showgrid=False, zeroline=False),
        yaxis6=dict(title="CMF", anchor="free", overlaying="y", side="right", position=0.55, showgrid=False, zeroline=False),
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font=dict(color=colors["text"]),
        legend=dict(orientation='h', y=-0.2),
        height = 600,
    )

    return fig

@app.callback(
    dash.dependencies.Output(component_id='news_feed', component_property='children'),
    dash.dependencies.Input(component_id='stock_dropdown', component_property='value'))
def news(stock):
    
    if not stock:
        return "No stock selected"
    
    news_articles = []

    news = newsapi.get_everything(q=stock, language='en', sort_by='relevancy', page_size=5)
    news_articles.extend(news['articles'])

    # Generating a simple list of news articles
    news_feed = html.Div([
        html.Div([
            html.A(article['title'], href=article['url'], target='_blank'),
            html.P(article['description'])
        ]) for article in news_articles
    ])

    return news_feed
     

@app.callback(
    dash.dependencies.Output(component_id='stock_dividends', component_property='children'),
    [dash.dependencies.Input(component_id='stock_dropdown', component_property='value')])
def update_stock_dividends(stock):
    if not stock:
        return "No stock selected"

    data = []
    no_dividends = True

    ticker = yf.Ticker(stock)
    company_name = ticker.info["longName"]
    dividends = ticker.dividends

    if not dividends.empty:
        no_dividends = False
        data.append(
            go.Scatter(
                x=dividends.index,
                y=dividends.values,
                name=stock
            )
        )

    if no_dividends:
        return "This stock has no dividends"

    fig = go.Figure(data=data)

    fig.update_layout(
        title='Dividends for '+ company_name,
        xaxis=dict(title="Date", showgrid=False, zeroline=False),
        yaxis=dict(title="Dividend", showgrid=False, zeroline=False),
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font=dict(color=colors["text"]),
    )

    return dcc.Graph(figure=fig)

@app.callback(
    dash.dependencies.Output(component_id='sp500_heatmap', component_property='figure'),
    dash.dependencies.Input(component_id='main_title', component_property='id'),
    )
def sp_500(val):

    data_frame = load_cache(CACHE_SP500)

    if data_frame is None:

        # Scrape S&P 500 tickers from Wikipedia
        sp500_symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

        # Define the batch size
        batch_size = 100

        # Initialize empty lists
        company_tickers = []
        price_changes = []
        industry_sectors = []
        mkt_caps = []

        # Process stock symbols in batches
        for batch in chunks(sp500_symbols, batch_size):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(process_symbol, batch)

                for result in results:
                    if result:
                        company_tickers.append(result['ticker'])
                        industry_sectors.append(result['sector'])
                        price_changes.append(result['delta'])
                        mkt_caps.append(result['market_cap'])

        # Create a DataFrame
        data_frame = pd.DataFrame({'ticker': company_tickers,
                                'sector': industry_sectors,
                                'delta': price_changes,
                                'market_cap': mkt_caps,
                                })
        
        # Save the processed data to the cache
        save_cache(CACHE_SP500, data_frame)

    # Define color bins for the treemap
    color_bins = [-1, -0.02, -0.01, 0, 0.01, 0.02, 1]
    data_frame['colors'] = pd.cut(data_frame['delta'], bins=color_bins, labels=['red', 'indianred', 'lightpink', 'lightgreen', 'lime', 'green'])

    fig = px.treemap(data_frame, path=[px.Constant("all"), 'sector', 'ticker'], values='market_cap', color='colors',
                           color_discrete_map={'(?)': '#262931', 'red': 'red', 'indianred': 'indianred', 'lightpink': 'lightpink', 'lightgreen': 'lightgreen', 'lime': 'lime', 'green': 'green'},
                           hover_data={'delta': ':.2p'}
                           )
    
    fig.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
    )

    return fig

@app.callback(
    dash.dependencies.Output(component_id='nasdaq_heatmap', component_property='figure'),
    dash.dependencies.Input(component_id='main_title', component_property='id'),
    )
def nasdaq(val):

    data_frame = load_cache(CACHE_NASDAQ)

    if data_frame is None:
        # Get the list of Dow Jones Industrial Average tickers
        nasdaq_tickers = si.tickers_nasdaq()

        # Convert the list of tickers to a DataFrame
        valid_tickers = [ticker for ticker in nasdaq_tickers if ticker.isalnum()] # type: ignore
        nasdaq_symbols = pd.DataFrame(valid_tickers, columns=['Symbol'])
        nasdaq_symbols = nasdaq_symbols[:2000]

        # Define the batch size
        batch_size = 100

        # Initialize empty lists
        company_tickers = []
        price_changes = []
        industry_sectors = []
        mkt_caps = []

        # Filter symbols that have sector information
        filtered_symbols = []
        for batch in chunks(nasdaq_symbols['Symbol'], batch_size):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(get_sector, batch)

                for result in results:
                    if result:
                        filtered_symbols.append(result)

        nasdaq_symbols_filtered = pd.DataFrame(filtered_symbols, columns=['Symbol'])

        # Process stock symbols in batches
        for batch in chunks(nasdaq_symbols_filtered['Symbol'], batch_size):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(process_symbol, batch)

                for result in results:
                    if result:
                        company_tickers.append(result['ticker'])
                        industry_sectors.append(result['sector'])
                        price_changes.append(result['delta'])
                        mkt_caps.append(result['market_cap'])

        # Create a DataFrame
        data_frame = pd.DataFrame({'ticker': company_tickers,
                                'sector': industry_sectors,
                                'delta': price_changes,
                                'market_cap': mkt_caps,
                                })
        
        # Save the processed data to cache file
        save_cache(CACHE_NASDAQ, data_frame)

    # Define color bins for the treemap
    color_bins = [-1, -0.02, -0.01, 0, 0.01, 0.02, 1]
    data_frame['colors'] = pd.cut(data_frame['delta'], bins=color_bins, labels=['red', 'indianred', 'lightpink', 'lightgreen', 'lime', 'green'])

    fig = px.treemap(data_frame, path=[px.Constant("all"), 'sector', 'ticker'], values='market_cap', color='colors',
                           color_discrete_map={'(?)': '#262931', 'red': 'red', 'indianred': 'indianred', 'lightpink': 'lightpink', 'lightgreen': 'lightgreen', 'lime': 'lime', 'green': 'green'},
                           hover_data={'delta': ':.2p'}
                           )
    
    fig.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
    )

    return fig

@app.callback(
    dash.dependencies.Output(component_id='dowjones_heatmap', component_property='figure'),
    dash.dependencies.Input(component_id='main_title', component_property='id'),
    )
def dowjones(val):

    data_frame = load_cache(CACHE_DOWJONES)

    if data_frame is None:
        # Get the list of Dow Jones Industrial Average tickers
        dowjones_tickers = si.tickers_dow()

        # Convert the list of tickers to a DataFrame
        valid_tickers = [ticker for ticker in dowjones_tickers if ticker.isalnum()] # type: ignore
        dowjones_symbols = pd.DataFrame(valid_tickers, columns=['Symbol'])
        dowjones_symbols = dowjones_symbols[:2000]

        # Define the batch size
        batch_size = 100

        # Initialize empty lists
        company_tickers = []
        price_changes = []
        industry_sectors = []
        mkt_caps = []

        # Filter symbols that have sector information
        filtered_symbols = []
        for batch in chunks(dowjones_symbols['Symbol'], batch_size):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(get_sector, batch)

                for result in results:
                    if result:
                        filtered_symbols.append(result)

        dowjones_symbols_filtered = pd.DataFrame(filtered_symbols, columns=['Symbol'])

        # Process stock symbols in batches
        for batch in chunks(dowjones_symbols_filtered['Symbol'], batch_size):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(process_symbol, batch)

                for result in results:
                    if result:
                        company_tickers.append(result['ticker'])
                        industry_sectors.append(result['sector'])
                        price_changes.append(result['delta'])
                        mkt_caps.append(result['market_cap'])

        # Create a DataFrame
        data_frame = pd.DataFrame({'ticker': company_tickers,
                                'sector': industry_sectors,
                                'delta': price_changes,
                                'market_cap': mkt_caps,
                                })
        
        # Save the processed data to cache file
        save_cache(CACHE_DOWJONES, data_frame)

    # Define color bins for the treemap
    color_bins = [-1, -0.02, -0.01, 0, 0.01, 0.02, 1]
    data_frame['colors'] = pd.cut(data_frame['delta'], bins=color_bins, labels=['red', 'indianred', 'lightpink', 'lightgreen', 'lime', 'green'])

    fig = px.treemap(data_frame, path=[px.Constant("all"), 'sector', 'ticker'], values='market_cap', color='colors',
                           color_discrete_map={'(?)': '#262931', 'red': 'red', 'indianred': 'indianred', 'lightpink': 'lightpink', 'lightgreen': 'lightgreen', 'lime': 'lime', 'green': 'green'},
                           hover_data={'delta': ':.2p'}
                           )
    
    fig.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)