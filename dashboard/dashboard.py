import lzma
import os

import dill as pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-dark-palette")

import threading
import sys
sys.path.insert(1, '/Users/oskarfransson/vs_code/trading')

from datetime import date,timedelta
from backtest_engine.database.borsdata.borsdata_api import BorsdataAPI
from backtest_engine.database.borsdata.constants import API_KEY


class Dashboard:
    def __init__(self) -> None:
        self.api = BorsdataAPI(API_KEY)
        self.path = '/Users/oskarfransson/vs_code/trading/backtest_engine/database/borsdata/'
        self.information = pd.read_csv(self.path+'instrument_with_meta_data.csv')
        end_date = date.today()
        self.end_date = end_date.strftime('%Y-%m-%d')
        start_date = end_date - timedelta(565)
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.date_range = pd.date_range(start_date,end_date)
        names = self.information['name'].unique().tolist()
        self.sector_map = {name:self.information[self.information['name']==name]['sector'].values[0] for name in names}
        
    def get_dfs(self,id_map):
        dfs = {}
        def _helper(idx):
            try:
                # 66 is Ordinarie utdelning
                val0 = self.api.get_kpi_data_instrument(ins_id=idx,kpi_id=66,calc_group='last',calc='latest')
                val1 = self.api.get_kpi_data_instrument(ins_id=idx,kpi_id=61,calc_group='last',calc='latest')
                df0 = pd.DataFrame(data=[[val0.values[0][0] for _ in range(len(self.date_range))],
                                            [val1.values[0][0] for _ in range(len(self.date_range))],
                                            [self.sector_map[id_map[idx]] for _ in range(len(self.date_range))],
                                            self.date_range
                                            ]
                                            ).T
                df0.columns = ['div', 'aktier', 'sector', 'date']
                prices = self.api.get_instrument_stock_prices(ins_id=idx,
                                                                from_date=self.start_date,
                                                                to_date=self.end_date,
                                                                max_count=300
                                                                ).sort_index(ascending=True)
                dfs[id_map[idx]] = df0.join(prices,how='left', on='date').fillna(method='ffill').set_index('date')
            except:
                 print(f'Skips {id_map[idx]}')
        
        threads = [threading.Thread(target=_helper, args=(idx,)) for idx in list(id_map.keys())]
        _ = [thread.start() for thread in threads]
        _ = [thread.join() for thread in threads]
        return list(dfs.keys()),dfs

    def get_dfs_sets(self,countries):
        tickers = []
        dfs = {}
        for country,markets in countries.items():
            for market in markets:
                mask_filter = (self.information['country'] == country) & (self.information['market'] == market)
                ids = self.information[mask_filter]['ins_id'].values.tolist()
                id_map = {}
                for idx in ids:
                    id_map[idx] = self.information[self.information['ins_id']==idx]['name'].values[0]

                tickers0, dfs0 = self.get_dfs(id_map)
                tickers += tickers0
                dfs |= dfs0
                del tickers0
                del dfs0
        return tickers,dfs
    
    def get_vix_data(self):
        def load_daily_data(ticker, start=None):
            from yfinance import download
            df = download(ticker, start=start)[['Open','High','Low','Close','Volume','Adj Close']]
            df['Ratio'] = df['Adj Close'] / df['Close']
            df['Open'] = df['Open'] * df['Ratio']
            df['High'] = df['High'] * df['Ratio']
            df['Low'] = df['Low'] * df['Ratio']
            df['Close'] = df['Close'] * df['Ratio']
            df = df.drop(['Ratio', 'Adj Close'], axis = 1)
            df.columns = ['open','high','low','close','volume']
            return df
        
        dfs = {ticker:load_daily_data(ticker=ticker)['close'] for ticker in ['^VIX','^VIX3M']}
        return dfs
        
    def load_pickle(self,path):
        with lzma.open(path,"rb") as fp:
            file = pickle.load(fp)
        return file

    def save_pickle(self,path,obj):
        with lzma.open(path,"wb") as fp:
            pickle.dump(obj,fp)

    def load_data(self,name,countries,remove=False):
        path = '/Users/oskarfransson/vs_code/trading/dashboard/'
        if remove:
            os.remove(path+name+'.obj')
        try:
            tickers,dfs = self.load_pickle(path+name+'.obj')
        except:
            tickers,dfs = self.get_dfs_sets(countries)
            self.save_pickle(path+name+'.obj',(tickers,dfs))
        return tickers,dfs
    
    def study_sectors(self,dfs0:dict,sector_cluster:list, cluster_name=0):
        dfs = {name:df.copy() for name,df in dfs0.items() if df['sector'].iloc[-1] in sector_cluster}
        insts = list(dfs.keys())
        temp_closes = {}
        temp_scaled = {}
        for inst in insts:
            try:
                temp_closes[inst] = dfs[inst]['close'].copy()
                rets = dfs[inst]['close'].pct_change().copy()
                scaled_rets = rets#/(rets.ewm(0.94).std() * np.sqrt(253))
                temp_scaled[inst] = scaled_rets
            except KeyError:
                pass
        
        closes = pd.concat(temp_closes,axis=1)
        scaled = pd.concat(temp_scaled,axis=1)
        sector_momentum = (closes / closes.rolling(253).mean() - 1).mean(axis=1)
        stats_df = pd.DataFrame(index=[f'sector cluster {cluster_name}'], 
                                data=sector_momentum.iloc[-1],
                                columns=['Momentum']
                            )
        index_series = scaled.mean(axis=1)
        return index_series,stats_df

    def calculate_filter1(self,tickers,dfs):
        insts = tickers
        output_df = pd.DataFrame(index=insts, columns=['Date','Momentum filter','Dividend yield', 'Market share', 'Sector'])
        for inst in insts:
            try:
                df = dfs[inst]
                momentum = df['close']/df['close'].rolling(253).mean() - 1
                output_df.loc[inst,'Momentum filter'] = momentum.iloc[-1]
                output_df.loc[inst,'Dividend yield'] = df['div'].fillna(value=np.nan).iloc[-1]/df['close'].iloc[-1]
                output_df.loc[inst,'Market share'] = int(df['aktier'].iloc[-1]*df['close'].iloc[-1])
                output_df.loc[inst,'Date'] = df.index[-1]
                output_df.loc[inst,'Sector'] = self.sector_map[inst]
            except KeyError:
                pass

        output_df = output_df.sort_values(by='Momentum filter',ascending=False)
        return output_df[(output_df['Dividend yield'] > 0.05) & (output_df['Market share'] > 1000)]
    
    def calculate_filter2(self,dfs0:dict,sector_cluster:list):
        dfs = {name:df.copy() for name,df in dfs0.items() if df['sector'].iloc[-1] in sector_cluster}
        insts = list(dfs.keys())
        zscore = lambda x: (x - np.nanmean(x))/np.nanstd(x)
        temp = {}
        for inst in insts:
            try:
                op1 = dfs[inst].volume
                op2 = (dfs[inst].close - dfs[inst].low) - (dfs[inst].high - dfs[inst].close)
                op3 = dfs[inst].high - dfs[inst].low
                op4 = op1 * op2 / op3
                temp[inst] = op4
            except:
                print(f'Passes {inst}')
        
        temp_df = pd.concat(temp,axis=1)
        temp_df = temp_df.replace(np.inf, np.nan).replace(-np.inf, np.nan)
        zscore = lambda x: (x - np.nanmean(x))/np.nanstd(x)
        cszcre_df = temp_df.fillna(method="ffill").apply(zscore, axis=1, raw=True)

        output_df0 = pd.DataFrame(index=insts, columns=['Date','Alpha','Momentum filter', 'Market share', 'Sector'])
        for inst in cszcre_df.columns:
            df = dfs[inst].copy()
            output_df0.loc[inst,'Momentum filter'] = (df['close'] - df['close'].shift(253)).iloc[-1]
            output_df0.loc[inst,'Alpha'] = (cszcre_df[inst].rolling(35).mean() * -1).iloc[-1]
            output_df0.loc[inst,'Market share'] = int(df['close'].iloc[-1]*df['aktier'].iloc[-1])
            output_df0.loc[inst,'Date'] = df.index[-1]
            output_df0.loc[inst,'Sector'] = self.sector_map[inst]

        output_df1 = output_df0[output_df0['Momentum filter'] > 0]
        temp_ser = output_df1['Alpha'].rank(method="average",na_option="keep",ascending=True)
        output_df = pd.concat([output_df1.drop(columns='Alpha'),temp_ser],axis=1)
        return output_df.dropna().sort_values(by='Alpha',ascending=False)
    
    def calculate_vix_basis(self):
        dfs = self.get_vix_data()
        v1,v3 = dfs['^VIX'],dfs['^VIX3M']
        v1.name,v3.name = 'VIX', 'VIX3M'
        basis = v3/v1
        basis.name = 'basis'
        average = basis.rolling(23).mean()
        average.name = 'average'
        df = pd.concat([v1,v3,basis,average],axis=1)
        df.index = v1.index
        return df


if __name__ == '__main__':
    countries = {
        'Sverige':['Large Cap', 'Mid Cap'],
        'Norge':['Oslo Bors', 'Oslo Expand'],
        'Danmark':['Large Cap', 'Mid Cap']
    }
    dashboard = Dashboard()
    tickers,dfs = dashboard.load_data('data',countries)

    import streamlit as st

    st.set_page_config(layout="wide")
    st.header('Dashboard')
    st.markdown("## VIX strategy")
    cols = st.columns([1.5,1])

    output_df3 = dashboard.calculate_vix_basis().dropna().tail(252)
    fig2, axs = plt.subplots(2,figsize = (14,10))
    axs[0].plot(output_df3[['basis','average']],label=['basis','average'])
    axs[0].legend()
    axs[1].plot(output_df3[['VIX','VIX3M']],label=['VIX','VIX3M'])
    axs[1].legend()
    fig2.tight_layout()
    cols[0].pyplot(fig2)
    output_df3['alpha'] = output_df3['basis'] / output_df3['average']
    def _styler(x):
        color = 'red' if x < 1. else 'green'
        return 'color: % s' % color
    cols[1].dataframe(output_df3.sort_index(ascending=False).head(10).style.applymap(_styler,subset=['alpha','basis']))
    
    st.markdown("## Sector strategy")
    button = st.button('Press to update data')
    if button:
        tickers,dfs = dashboard.load_data('data',countries,remove=True)
    else:
        pass
    
    cols = st.columns([2,1])
    cl1 = cols[1].multiselect('Cluster 1',
                             dashboard.information['sector'].dropna().unique().tolist(),[
                                 'Industri',
                                 'Finans & Fastighet',
                                 'Sällanköpsvaror', 
                                 'Material',
                                 'Dagligvaror'
                                ],
                            key="cl1"
                        )
    cl2 = cols[1].multiselect('Cluster 2',
                             dashboard.information['sector'].dropna().unique().tolist(),[
                                 'Energi', 
                                 'Kraftförsörjning'
                                 ],
                            key="cl2"
                        )
    cl3 = cols[1].multiselect('Cluster 3',
                             dashboard.information['sector'].dropna().unique().tolist(),[
                                 'Telekommunikation'
                             ],
                             key="cl3"
                        )
    cl4 = cols[1].multiselect('Cluster 4',
                             dashboard.information['sector'].dropna().unique().tolist(),[
                                 'Hälsovård',
                                 'Informationsteknik'
                             ],
                             key="cl4"
                        )

    cluster_list = [cl1,cl2,cl3,cl4]
    
    series = {}
    tables = []
    for selected_cluster,i in zip(cluster_list,range(len(cluster_list))):
        serie,stats = dashboard.study_sectors(dfs,selected_cluster,i+1)
        series['cluster'+str(i)] = (1+serie).cumprod()
        tables.append(stats)
    
    plot_table = pd.concat(series,axis=1)
    fig1, ax = plt.subplots(1,figsize = (14,10))
    ax.plot(plot_table,label=[f'Cluster {i+1}' for i in range(4)])
    ax.legend()
    fig1.tight_layout()
    stats_table = pd.concat(tables,axis=0)
    cols[0].pyplot(fig=fig1)
    cols[1].dataframe(stats_table)
    
    cluster = cols[0].multiselect('Choose sector cluster',
                             dashboard.information['sector'].dropna().unique().tolist(),
                             dashboard.information['sector'].dropna().unique().tolist()[0])
    
    output_df1 = dashboard.calculate_filter2(dfs,cluster)
    cols[0].dataframe(output_df1)

    st.markdown("## Systematic yield strategy")
    cols = st.columns([2,1])

    output_df2 = dashboard.calculate_filter1(tickers,dfs)
    sel_tickers = output_df2.index[:11].tolist()
    sel_dfs = {}
    for ticker in sel_tickers:
        rets = (1+dfs[ticker]['close'].pct_change()).cumprod().apply(np.log)
        sel_dfs[ticker] = rets
    sel_dfs = pd.concat(sel_dfs,axis=1)
    
    fig3, axs = plt.subplots(1,figsize = (14,10))
    for col in sel_dfs.columns.tolist():
        axs.plot(sel_dfs.index,sel_dfs[col])
        axs.annotate(xy=(sel_dfs.index[-1],sel_dfs[col].iloc[-1]), xytext=(5,0), textcoords='offset points', text=col, va='center')
    fig3.tight_layout()
    cols[0].pyplot(fig=fig3)
    cols[1].dataframe(output_df2)
