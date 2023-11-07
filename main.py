import matplotlib.pyplot as plt
import numpy as np

from backtest_engine.gene import GeneticAlpha
from backtest_engine.gene import Gene

from pprint import pprint
from datetime import datetime

import streamlit as st

import warnings
warnings.filterwarnings("ignore")


def get_from_db(countries,start,end):
    from backtest_engine.database.borsdata.borsdata_price_database_script import priceDataBase
    from backtest_engine.database.borsdata.borsdata_kpi_database_script import kpiDataBase

    from backtest_engine.utils import merge_kpi_price,concat_kpi_price
    
    tickers = []
    dfs = {}
    for country, markets in countries.items():
        for market in markets:
            db_price = priceDataBase(country, market, start, end)
            db_kpi0 = kpiDataBase(61, market, country)
            db_kpi1 = kpiDataBase(66, market, country)
            price_tickers, price_dfs = db_price.export_database()
            kpi_tickers0,kpi_dfs0 = db_kpi0.export_database()
            kpi_tickers1,kpi_dfs1 = db_kpi1.export_database()
            tickers0,dfs0 = merge_kpi_price(
                                        kpi_dfs0,
                                        price_dfs,
                                        kpi_tickers0,
                                        price_tickers,
                                        'aktier'
                                    )
            tickers1,dfs1 = concat_kpi_price(
                                        kpi_dfs1,
                                        dfs0,
                                        kpi_tickers1,
                                        tickers0,
                                        'yld'  
                                    )
            
            tickers += tickers1
            dfs |= dfs1
            del tickers0
            del dfs0
            del tickers1
            del dfs1
    return tickers,dfs
        
                                                   
def get_ticker_dfs(countries,start,end):
    from backtest_engine.utils import load_pickle,save_pickle
    try:
        tickers, dfs = load_pickle("/Users/oskarfransson/vs_code/trading/dataset_syst_yield.obj")
    except Exception as err:
        tickers,dfs = get_from_db(countries,start,end)
        save_pickle("/Users/oskarfransson/vs_code/trading/dataset_syst_yield.obj", (tickers,dfs))
    return tickers, dfs

countries = {'Sverige':['Large Cap']}

def create_nodes(decrement=False):
    if decrement:
        st.session_state.count -= 1
    operator = st.selectbox('Select operator', (
        'ls_',
        'mean_',
        'and',
        'ite',
        'gt',
        'lt',
        'mult',
        'div',
        'plus',
        'minus'
    ))
    if 'count' not in st.session_state:
        st.session_state.count = 0
        current_inputs = []
    
    if operator[-1] == '_':
        parameter = st.number_input(f'Select parameter for {operator[:-1]}', min_value=0, step=1)
        argument = st.selectbox(f'Select argument for {operator + str(parameter)}',
                                ['close', 'low', 'high', 'open', 'volume'] + current_inputs)
        function_string = operator + str(parameter) + '(' + argument + ')'
    else:
        argument1 = st.selectbox(f'Select argument 1 for {operator}',
                                 ['close', 'low', 'high', 'open', 'volume'] + current_inputs)
        argument2 = st.selectbox(f'Select argument 2 for {operator}',
                                 ['close', 'low', 'high', 'open', 'volume'] + current_inputs)
        function_string = operator + '(' + argument1 + ',' + argument2 + ')'

    execute_button = st.button('Add operation')
    if execute_button:
        current_inputs += [function_string]
        st.write(current_inputs)
        done_button = st.button('Done')
        add_more = st.button('Add more')
        if done_button:
            st.write('Done!')
        if add_more:
            return create_nodes(decrement=True)


def main():
    st.header('Backtester')
    create_nodes()
    exit()

    period_start = datetime(2011,1,1)
    period_end = datetime(2023,9,1)
    start = '2013-01-01'
    end = '2023-09-01'
    _, dfs = get_ticker_dfs(countries=countries,start=start,end=end)
    tickers = list(dfs.keys())
    '''
    ls_0/10(and(gt(div(yld,close),const_0.05),gt(div(close,mean_253(close)),const_1)))
    plus(ite(gt(mean_10(close),mean_50(close)),const_1,const_0),ite(gt(mean_20(close),mean_100(close)),const_1,const_0),ite(gt(mean_50(close),mean_200(close)),const_1,const_0))

    '''
    g1=Gene.str_to_gene("plus(ite(gt(mean_10(close),mean_50(close)),const_1,const_0),ite(gt(mean_20(close),mean_100(close)),const_1,const_0),ite(gt(mean_50(close),mean_200(close)),const_1,const_0))")
    alpha1=GeneticAlpha(insts=tickers,dfs=dfs,start=period_start,end=period_end,trade_frequency='weekly',genome=g1)
    df1=alpha1.run_simulation()
    alpha1.get_perf_stats(plot=True)
       
if __name__ == "__main__":
    main()
