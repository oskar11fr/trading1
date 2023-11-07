from datetime import datetime 
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)


def get_from_db():
    from backtest_engine.database.finance_database_script import finance_database
    crdb = finance_database('crypto_db')
    etfdb = finance_database('etf_db')
    _,temp_dfs1 = crdb.export_from_database()
    _,temp_dfs2 = etfdb.export_from_database()
    dfs = {'BTC_USD':temp_dfs1['BTC_USD'], 'BIL':temp_dfs2['BIL']}
    tickers = list(dfs.keys())
    return tickers,dfs

tickers,dfs = get_from_db()

from backtest_engine.strategies.btc_momentum_srategy import BtcStrategy
period_start = datetime(2014,9,17)
period_end = datetime(2023,11,4)
alpha = BtcStrategy(insts=tickers,dfs=dfs,start=period_start,end=period_end,trade_frequency='weekly')
alpha.run_simulation(use_vol_target=False)
stats = alpha.get_perf_stats(plot=True)
hypothesis = alpha.get_hypothesis_tests(m=200)