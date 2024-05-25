from datetime import datetime
import numpy as np
import torch
import re
import torch.nn.functional as F
from datetime import timedelta
from math import pi
from typing import Callable, List, Tuple, Union, Any
import pandas as pd
from datatable import f, dt
from bokeh.plotting import figure, show


def clean_text(text: str) -> str:
    text = re.sub(r'\<.*?\>', '', text)
    text = re.sub(r'\r', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\t', '', text)
    return text


def get_class_weights(df: Union[pd.DataFrame, torch.Tensor], num_classes: int,
                      max_samples: int = 10000) -> torch.FloatTensor:
    if isinstance(df, pd.DataFrame):
        df = torch.Tensor(pd.DataFrame)
    totals = torch.zeros(num_classes)
    for i in range(max_samples):
        totals += F.one_hot(df[i], num_classes).sum(dim=0)
        i += 1
    return (1 - totals / totals.sum()) / (num_classes - 1)  # type: ignore


def utc_now():
    return datetime.utcnow()


def get_timestamps_range(start_utc_timestamp: datetime, end_utc_timestamp: datetime, step: timedelta) -> List[datetime]:
    ts = start_utc_timestamp
    timestamps = []
    while ts <= end_utc_timestamp:
        timestamps.append(ts)
        ts += step

    return timestamps


def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)

def load_mt5_ticks_csv(filename: str):
    df = pd.read_csv(filename, sep='\t', engine="pyarrow")
    ask = df['<ASK>']
    bid = df['<BID>']
    timestamps = pd.to_datetime(
        df['<DATE>'] + ' ' + df['<TIME>'])
    volume = df['<VOLUME>']
    df = pd.concat([timestamps, ask, bid, volume], axis=1)
    df.columns = ['timestamp', 'ask', 'bid', 'volume']
    df.ffill(inplace=True)
    df['timestamp'] = pd.to_datetime(df.timestamp, utc=True)
        
    df['index'] = df.timestamp.dt.floor('1s')
    df.set_index('index', inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    
    return df

def load_mt5_bars_csv(filename: str):
    
    df = pd.read_csv(filename, sep='\t', engine="pyarrow")
    
    df.rename(columns={
        '<DATE>': 'date', 
        '<TIME>': 'time',
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'volume',
        '<SPREAD>': 'spread',
    }, inplace=True)
    
    df['timestamp'] = pd.to_datetime(df.date.astype(str) + ' ' + df.time.astype(str), utc=True)
    df['index'] = df['timestamp']
    df.set_index('index', inplace=True)
    
    df['bid'] = df['close']
    df['ask'] = df['close'] + 0.00001 * df['spread']
    
    df = df[['timestamp', 'ask', 'bid', 'volume']]
    
    return df

def dt_load_mt5_bars_csv(filename: str):
    
    df = dt.Frame(load_mt5_bars_csv(filename))
    
    return df

def _fn_timestamp_aggregate(timestamp: datetime, period: str = 'M15') -> datetime:
    if period == 'D1':
        return timestamp.replace(minute=0, hour=0, second=0, microsecond=0)
    if period == 'H1':
        return timestamp.replace(minute=0, second=0, microsecond=0)
    if period == 'M1':
        return timestamp.replace(second=0, microsecond=0)
    if period == 'M5':
        return timestamp.replace(minute=timestamp.minute - timestamp.minute % 5, second=0, microsecond=0)
    if period == 'M15':
        return timestamp.replace(minute=timestamp.minute - timestamp.minute % 15, second=0, microsecond=0)
    if period == 'M20':
        return timestamp.replace(minute=timestamp.minute - timestamp.minute % 20, second=0, microsecond=0)
    if period == 'M30':
        return timestamp.replace(minute=timestamp.minute - timestamp.minute % 30, second=0, microsecond=0)
    return timestamp

def _group(df: pd.DataFrame, period: str) -> 'pd.DataFrameGroupBy':
    df['timestamp'] = df.timestamp.apply(lambda t: _fn_timestamp_aggregate(t, period))
    return df.groupby('timestamp')

def aggregate_prices(df: pd.DataFrame, period: str, price_col_open='bid', price_col_close='bid', price_col_low='bid', price_col_high='bid') -> pd.DataFrame:
    
    if price_col_open not in df.columns:
        price_col_open = 'open'
    if price_col_close not in df.columns:
        price_col_close = 'close'
    if price_col_low not in df.columns:
        price_col_low = 'low'
    if price_col_high not in df.columns:
        price_col_high = 'high'
    
    grouped = _group(df, period)

    result = pd.concat(
        [grouped.timestamp.first(), grouped[price_col_open].first(), grouped[price_col_close].last(), grouped[price_col_low].min(), grouped[price_col_high].max(),
            grouped.volume.sum()], axis=1)
    result.columns = ['timestamp', 'open', 'close', 'low', 'high', 'volume']
    return result


def plot_candlesticks(df: pd.DataFrame, p: figure, timeframe: str = 'M1'):
    inc = df.close > df.open
    dec = df.open > df.close

    w = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 24 * 60}[timeframe] * 60 * 1000

    p.xaxis.major_label_orientation = pi / 4
    p.grid.grid_line_alpha = 0.3

    p.segment(df.timestamp, df.high, df.timestamp, df.low, color="black")
    p.vbar(df.index[inc], w, df.open[inc], df.close[inc], fill_color="#00FF00", line_color="black")
    p.vbar(df.index[dec], w, df.open[dec], df.close[dec], fill_color="#FF0000", line_color="black")


def plot(df: pd.DataFrame, charts: List[Callable], tools="pan,wheel_zoom,box_zoom,reset", x_axis_type="datetime") -> Union[Tuple[Any, figure], None]:
    if not charts:
        return
    p = figure(title="Plot", width=800, tools=tools, x_axis_label="Timestamp", y_axis_label="Price",
               x_axis_type=x_axis_type)
    for chart in charts:
        chart(df, p)
    t = show(p, notebook_handle=True)
    return t, p


def plot_wrap_fn(fn: Callable, *args, **kwargs) -> Callable:
    def plot_fn(df: pd.DataFrame, p: figure):
        fn(df, p, *args, **kwargs)

    return plot_fn