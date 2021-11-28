import os
import sys
import time
import datetime
import requests
import json
import hashlib
import numpy as np
import pandas as pd
import talib as ta

import winsound

import threading

import asyncio
import websockets

import mplfinance as mpf

#import trading_calendars as tcal
#import pandas_market_calendars as mcal
import cn_stock_holidays.data as ccal

#import akshare
from pytdx.hq import TdxHq_API
from pytdx.config.hosts import hq_hosts

class quote_base:
    _decl = None
    _resample_rules = {
            '1m':   '1T',
            '2m':   '2T',
            '3m':   '3T',
            '5m':   '5T',
            '10m':  '10T',
            '15m':  '15T',
            '30m':  '30T',
            '1h':   '1H',
            '2h':   '2H',
            '1d':   '1D',
            '2d':   '2D',
            '3d':   '3D',
            '1w':   '1W',
            '2w':   '2W',
            '3w':   '3W',
            '1M':   '1M',
            '2M':   '2M',
            '45d':  '45D',
            '1q':   '1Q',
            '2q':   '2Q',
            '1y':   '1Y',
            '2y':   '2Y',
            '3y':   '3Y'
            }

    _session = None
    
    def __init__(self):
        pass

    def sleep(self):
        pass
    
    def get_decl(self):
        return self._decl
    
    def get_first(self):
        return next(iter(self._decl))
    
    def get_resample_rules(self):
        return self._resample_rules
    
    def get_symbol(self, symbol):
        vs = symbol.split('.')
        return vs[1], vs[0]
        
    def during(self):
        return True
    
    def time_between(self, _time, time_ranges):
        for time_range in time_ranges:
            if time_range[1] < time_range[0]:
                if time_range[1] <= _time <= time_range[0]:
                    return True
    
            if time_range[0] <= _time <= time_range[1]:
                return True

    def get_digits(self, symbol):
        return 5
    
class quote_china(quote_base):
    _time_ranges = [(datetime.time(9, 10), datetime.time(11, 35)),  (datetime.time(12, 55), datetime.time(15, 5))]
    
    def during(self, force=0):
        #v = tcal.get_calendar('XSHG').is_session(pd.Timestamp.today()) and self.time_between(pd.Timestamp.now().time(), self._time_ranges)
        #v = mcal.get_calendar('SSE').open_at_time(schedule, pd.Timestamp.now())
        v = ccal.is_trading_day(pd.Timestamp.today()) and self.time_between(pd.Timestamp.now().time(), self._time_ranges)
        return force or v
        
    def get_market(self, symbol):
        if symbol[:1] in ('0', '3'):
            return '0'
        elif symbol[:1] in ('5', '6'):
            return '1'
        elif symbol[:2] in ('15', '16'):
            return '0'
        elif symbol[:3] in ('123', '127', '128'):
            return '0'
        elif symbol[:3] in ('110', '111', '113', '118', '132'):
            return '1'
        
    def get_digits(self, symbol):
        if symbol[:1] in ('0', '3', '5', '6'):
            return 2
        elif symbol[:2] in ('15', '16'):
            return 2
        elif symbol[:3] in ('110', '111', '113', '118', '132'):
            return 2
        elif symbol[:3] in ('123', '127', '128'):
            return 3
        
    def resample_Candle(self, df, interval):
        resample_rules = self.get_resample_rules()
        if interval in resample_rules:
            rule = resample_rules[interval]
            if rule == '1H':
                bins = np.digitize(df.index.hour.values, bins=[12])
                df.index += (1 - bins) * pd.Timedelta('30 min')
            elif rule == '2H':
                rule = '4H'

            #print(df, interval)
            
            ohlcv = { 'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min',  'volume': 'sum' }
            return df.resample(rule, closed='right', label='right').agg(ohlcv).dropna()
'''
class quote_AkShare(quote_china):
    _real = { '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '1d': 'daily', '1w': 'weekly', '1M': 'monthly' }
    _decl = {
            '5m':   [None, None],
            '10m':  ['5m', '5m'],
            '15m':  [None, '5m'],
            '30m':  [None, '15m'],
            '1h':   [None, '30m'],
            '2h':   ['1h', '1h'],
            '1d':   [None, '1h'],
            '1w':   [None, '1d'],
            '1M':   [None, '1d'],
            }
    
    def get_symbols(self):
        df = akshare.stock_zh_a_spot_em()
        
        vs = {}
        for row in df.itertuples():
            _symbol = row[2]
            _market = self.get_market(_symbol)
            vs[f'{_symbol}.{_market}'] = row[3]

        return vs
        
    def get_candle(self, symbol, interval=None, flag=0):
        if interval:
            if interval in self._real:
                _market, _symbol = self.get_symbol(symbol)
                _interval = self._real[interval]
                if _interval in ['daily', 'weekly', 'monthly']:
                    df = akshare.stock_zh_a_hist(symbol=_symbol, period=_interval, adjust='qfq')
                    df.rename(columns={'日期': 'time', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}, inplace=True)
                else:
                    df = akshare.stock_zh_a_hist_min_em(symbol=_symbol, period=_interval, adjust='qfq')
                    df.rename(columns={'时间': 'time', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}, inplace=True)
                    
                df.drop(columns=['涨跌幅', '涨跌额', '成交额', '振幅', '换手率'], inplace=True)
                df.time = pd.to_datetime(df.time)
                df.set_index(['time'], inplace=True)
                
                return df
            
        else:
            return self.get_candle(symbol, self.get_first(), 1)
'''
class quote_PyTdx(quote_china):
    _real = { '1m': 8, '5m': 0, '15m': 1, '30m': 2, '1h': 3, '1d': 4, '1w': 5, '1M': 6, '1q': 10, '1y': 11 }
    _decl = {
            '5m':   [None, None],
            '10m':  ['5m', '5m'],
            '15m':  [None, '5m'],
            '30m':  [None, '15m'],
            '1h':   [None, '30m'],
            '2h':   ['1h', '1h'],
            '1d':   [None, '1h'],
            '1w':   [None, '1d'],
            '1M':   [None, '1d'],
            }
    
    def get_session(self):
        if not self._session:
            self._session = TdxHq_API(heartbeat=True)
            if not self._session.connect('119.147.212.81', 7709):
                return
            
        return self._session

    def get_symbols(self):
        _session = self.get_session()
        if _session:
            vs = {}
            for _market in (0, 1):
                _count = 0
                while 1:
                    try:
                        data = _session.get_security_list(_market, _count)
                        if not data:
                            break

                        for v in data:
                            _symbol = v['code']
                            vs[f'{_symbol}.{_market}'] = v['name']

                        _count += len(data)
                        
                    except Exception as e:
                        pass
                        
            return vs
        
    def get_symbol(self, symbol):
        vs = symbol.split('.')
        return 0 if vs[1] == '0' else 1, vs[0]
        
    def get_candle(self, symbol, interval=None, flag=0):
        _session = self.get_session()
        if _session:
            if interval in self._real:
                _market, _symbol = self.get_symbol(symbol)
                _interval = self._real[interval]
                _count = 100 if flag else 800
                try:
                    data = _session.get_security_bars(_interval, _market, _symbol, 0, _count)
                    df = _session.to_df( data )
                    df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'amount'], inplace=True)
                    df.rename(columns={'datetime': 'time', 'vol': 'volume'}, inplace=True)
                    df.time = pd.to_datetime(df.time)
                    df.set_index(['time'], inplace=True)
                    
                    return df
                
                except Exception as e:
                    pass
        
class strategy_base:
    pass
    
class strategy_MA(strategy_base):
    def execute(self, symbol, cache):
        def df_compute(symbol, interval, df):
            param = [5, 10, 20, 30, 60, 120, 240]

            l_values = df['low'].values
            c_values = df['close'].values
            c = len(c_values)
            
            k = -1
            dt = df.index[k].strftime('%Y-%m-%d %H:%M:%S')
            price = l_values[k]
            
            vs = []
            for time_period in param:
                if time_period < c:
                    ma = ta.MA(c_values, timeperiod=time_period, matype=0)
                    value = ma[k]
                    if price > value and price < value*1.01 and ma[k] > ma[k-1] and ma[k-1] > ma[k-2]:
                        flag = 0
                        for i in range(k, k - time_period//2):
                            if l_values[i] > ma[i]:
                                flag = 1
                                break

                        if not flag:
                            diff = np.around(price - value, digits)
                            pt = np.around((price / value - 1) * 100, 2)
                            value = np.around(value, digits)
                            v = '{:>8.2f} {:>8.2f} {:>8.2f}%'.format(value, diff, pt)
                            vs.append({'time': dt, 'interval': interval, 'function': f'ma{time_period}', 'price': price, 'value': v})

            return vs

        digits = market.get_quote(symbol).get_digits(symbol)
        
        vs = []
        for i in cache:
            v = df_compute(symbol, i, cache[i])
            if v:
                vs.extend(v)
        
        return vs
    
class strategy_MACD(strategy_base):
    def execute(self, symbol, cache):
        def df_compute(symbol, interval, df):
            param = (12, 26, 9)
            
            h_values = df['high'].values
            l_values = df['low'].values
            c_values = df['close'].values
            c = len(c_values)
            
            k = -1
            dt = df.index[k].strftime('%Y-%m-%d %H:%M:%S')
            price = c_values[k]
            
            vs = []
            macd = ta.MACD(c_values, fastperiod=param[0], slowperiod=param[1], signalperiod=param[2])
            #print(macd)

            hist = macd[2]
            #for k in range(k, -c, -1):
            if 1:
                #print(symbol, interval, k)
                if k>-c and hist[k]>0 and hist[k-1]<0:
                    vsz = []
                    flag = False
                    i = k
                    while i > -c:
                        if (flag and hist[i]>0 and hist[i-1]<0) or (not flag and hist[i]<0 and hist[i-1]>0):
                            flag = not flag
                            vsz.append(i)

                        if len(vsz)>2:
                            break
                
                        i -= 1

                    #print(symbol, interval, k, df.index[k], vsz)
                    if len(vsz)>2:
                        if hist[k]>0 and hist[k-1]<0:
                            vx0 = min(macd[0][vsz[0]:k])
                            vx1 = min(macd[0][vsz[2]:vsz[1]])
                            vy0 = min(l_values[vsz[0]:k])
                            vy1 = min(l_values[vsz[2]:vsz[1]])
                            #vy = max(h_values[vsz[1]:vsz[0]])

                            #print(vx0, vx1, vy0, vy1)

                            if vx0>vx1 and vy0<vy1:
                                vs.append({'time': dt, 'interval': interval, 'function': f'macd-pd', 'price': price})
                                #print(vs)

            if 1:
                #print(symbol, interval, k)
                if k>-c and hist[k]<0 and hist[k-1]>0:
                    vsz = []
                    flag = False
                    i = k
                    while i > -c:
                        if (flag and hist[i]<0 and hist[i-1]>0) or (not flag and hist[i]>0 and hist[i-1]<0):
                            flag = not flag
                            vsz.append(i)

                        if len(vsz)>2:
                            break
                
                        i -= 1

                    #print(symbol, interval, k, df.index[k], vsz)
                    if len(vsz)>2:
                        if hist[k]<0 and hist[k-1]>0:
                            vx0 = max(macd[0][vsz[0]:k])
                            vx1 = max(macd[0][vsz[2]:vsz[1]])
                            vy0 = max(h_values[vsz[0]:k])
                            vy1 = max(h_values[vsz[2]:vsz[1]])
                            #vy = mix(l_values[vsz[1]:vsz[0]])

                            #print(vx0, vx1, vy0, vy1)

                            if vx0<vx1 and vy0>vy1:
                                vs.append({'time': dt, 'interval': interval, 'function': f'macd-nd', 'price': price})
                                #print(vs)

            return vs

        vs = []
        for i in cache:
            v = df_compute(symbol, i, cache[i])
            if v:
                vs.extend(v)
        
        return vs
    
class web_Client:
    _url = None
    _greet = None
    _callback = None
    _websocket = None
    def __init__(self, url=f'ws://localhost:5050', greet=None, callback=None):
        self._url = url
        self._greet = greet
        self._callback = callback

    def run(self):
        asyncio.run(self.open())

    def recv(self):
        asyncio.run(self.recv_msg())

    def send(self, msg):
        asyncio.run(self.send_msg(msg))
        
    async def open(self):
        async with websockets.connect(self._url) as websocket:
            self._websocket = websocket
            if self._greet:
                await self.send_msg(self._greet)
                
            while True:
                await self.recv_msg()

    async def recv_msg(self):
        msg = await self._websocket.recv()
        print(f"{msg}")
        if self._callback:
            self._callback(msg)

    async def send_msg(self, msg):
        await self._websocket.send(msg)
        
class market:
    _running = False
    _filename = 'quant.cfg'
    _config = None
    _selection = {}
    _symbols = {}
    _quotes = {}
    _strategies = {}
    _unique = []
    _checksum = ''
    
    def read_config():
        print('read_config')
        if os.path.exists(market._filename):
            with open(market._filename, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
        
    def write_config(data):
        with open(market._filename, 'w', encoding='utf-8-sig') as f:
            f.write(json.dumps(data, indent=1, ensure_ascii=False))
        
    def set_selection(data):
        root = market.read_config()
        root['selection'] = data
        with open(market._filename, 'w') as f:
            f.write(json.dumps(root, ensure_ascii=False))        
    
    def load_config():
        print('load_config')
        if not market._config:
            market._config = market.read_config()

        if market._config:
            market._selection = market._config.get('selection')
        
    def stop():
        print('Stopping...')
        market._running = False
        
    def sleep(v=0):
        if v:
            time.sleep(v)
    
    def get_quote(symbol):
        v = 'AkShare'
        v = 'PyTdx'
        if not v in market._quotes:
            market._quotes[v] = getattr(sys.modules[__name__], f'quote_{v}')()
        
        return market._quotes[v]
    
    def get_symbols(symbol):
        if not symbol in market._symbols:
            quote = market.get_quote(symbol)
            data = quote.get_symbols()
            market._symbols = {**market._symbols, **data}
        
        return market._symbols
    
    def get_cache(symbol):
        #print('get cache:', symbol)
        
        quote = market.get_quote(symbol)
        decl = quote.get_decl()
        
        cache = {}
        for i in decl:
            df = quote.get_candle(symbol, i)
            if df is None:
                v = decl[i][1]
                df = quote.resample_Candle(cache[v], i)
            elif i == '1h':
                df = quote.resample_Candle(df, i)
            else:
                quote.sleep()

            cache[i] = df

        #print(cache)
        
        return cache


    def update_cache(symbol, cache):
        #print('update_cache:', symbol)

        quote = market.get_quote(symbol)
        decl = quote.get_decl()
        
        df_sample = quote.get_candle(symbol)
        if not df_sample is None:
            for i in decl:
                df = cache[i]

                df0 = df.copy()
                
                if i == quote.get_first():
                    df_tmp = df_sample[-100:]
                else:
                    v = decl[i][1]
                    df_tmp = cache[v].iloc[-100:]
                    df_tmp = quote.resample_Candle(df_tmp, i)

                df_tmp = df_tmp.iloc[2:]
                df = df.combine_first(df_tmp)
                df.update(df_tmp)
                
                cache[i] = df

                df0.to_csv(f'r:/{i}_0.txt')
                df.to_csv(f'r:/{i}_1.txt')
                df_tmp.to_csv(f'r:/{i}_tmp.txt')
                #df0.compare(df).to_csv(f'r:/{i}_comp.txt')
                
        #print(cache)
        
    def get_strategy():
        #vs = ['MA', 'MACD']
        vs = ['MA']
        for v in vs:
            if not v in market._strategies:
                market._strategies[v] = getattr(sys.modules[__name__], f'strategy_{v}')()
        
        return market._strategies
        
    def run_single(cache, force=0):
        print(pd.Timestamp.now())

        result = False
        
        for symbol in market._selection:
            #print(symbol)
            
            if not 'title' in market._selection[symbol]:
                _symbols = market.get_symbols(symbol)
                market._selection[symbol]['title'] = _symbols[symbol]
                                
            quote = market.get_quote(symbol)
            digits = quote.get_digits(symbol)
            flag = quote.during(force)
            if flag:
                if not(symbol in cache and cache[symbol]):
                    cache[symbol] = market.get_cache(symbol)
            
                if cache[symbol]:
                    market.update_cache(symbol, cache[symbol])
                    strategies = market.get_strategy()
                    for strategy in strategies:
                        root = strategies[strategy].execute(symbol, cache[symbol])
                        vs = []
                        for v in root:
                            u = symbol + v['interval'] + v['function'] + v['time']
                            u = hashlib.md5(u.encode()).hexdigest()
                            if not u in market._unique:
                                vs.append(v)
                                market._unique.append(u)
                                
                        if vs:
                            title = market._selection[symbol]['title']
                            if 'url' in market._config and market._config['url']:
                                market.web_client.send(json.dumps({'symbol': symbol, 'title': title, 'data': vs}))
                            else:
                                for v in vs:
                                    print('{:<10} {:<10} {:>10.3f} {:>3} {:<10} {} {}'.format(symbol, title, v['price'], v['interval'], v['function'], v['v'] if 'v' in v else '', v['time']))
                            
                                winsound.PlaySound('*', winsound.SND_ALIAS)

            result = result or flag
            #break

        return result

    def show_mpf(df):
        vma5 = df['volume'].rolling(5).mean()
        vma10 = df['volume'].rolling(10).mean()
        
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        histogram[histogram < 0] = None  
        histogram_positive = histogram
        histogram = macd - signal  
        histogram[histogram >= 0] = None  
        histogram_negative = histogram
        print(macd, signal, histogram)
        add_plot = [
            mpf.make_addplot(vma5, panel=1, color='fuchsia', secondary_y='auto'),
            mpf.make_addplot(vma10, panel=1, color='b', secondary_y='auto'),

            #mpf.make_addplot(exp12, type='line', color='y'),
            #mpf.make_addplot(exp26, type='line', color='r'),
            #mpf.make_addplot(histogram, type='bar', width=0.7, panel=2, color='dimgray', alpha=1, secondary_y=False),
            mpf.make_addplot(histogram_positive, type='bar', width=0.7, panel=2, color=(.5, 1, .5)),
            mpf.make_addplot(histogram_negative, type='bar', width=0.7, panel=2, color=(1, .5, .5)),
            mpf.make_addplot(macd, panel=2, color='fuchsia', secondary_y=True),
            mpf.make_addplot(signal, panel=2, color='b', secondary_y=True),
        ]
        
        mpf.plot(df, type='candle', mav=(5, 10, 20, 30, 60, 120, 240), volume=True, addplot=add_plot, )
        
    def write_cache(cache):
        for symbol in cache:
            for interval in cache[symbol]:
                cache[symbol][interval].to_csv(f'r:/{symbol}.{interval}.txt')
                
    def run_target():
        print('Start...')
        market.load_config()
        cache = {}
        
        threading.Thread(target=market.run_client).start()

        market.run_single(cache, 1)
        #market.write_cache(cache)
        
        while market._running:
            full_during = market.run_single(cache)

            market.sleep(0 if full_during else 30)
                
        print('Stopped.')

    def run():
        if not market._running:
            market._running = True
            threading.Thread(target=market.run_target).start()
            
    def run_client():
        market.web_client = web_Client(url=f'ws://localhost:5050', greet=json.dumps({'key': '1'}), callback=market.client_callback)
        market.web_client.run()
        
    def client_callback(data):
        print(data)
        
if __name__ == '__main__':
    #print(quote_PyTdx().get_symbols())
    #print(quote_PyTdx().get_candle('000001.0', '1d'))
    #market.show_mpf(quote_PyTdx().get_candle('000001.0', '1d'))
    #web_client.run()
    #market.run_target()
    market.run()
