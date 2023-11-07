import pandas as pd
from backtest_engine.utils import BacktestEngine

class BtcStrategy(BacktestEngine):

    def __init__(self,insts,dfs,start,end,trade_frequency):
        super().__init__(insts,dfs,start,end,trade_frequency)
    
    def pre_compute(self,trade_range):
        ser = self.dfs['BTC_USD']['close']
        ser['returns'] = ser.pct_change()
        ser['std_returns'] = (ser['returns']-ser['returns'].rolling(23).mean())/ser['returns'].rolling(23).std()
        ser['std_returns_ave'] = ser['std_returns'].rolling(100).mean()
        self.dfs['std_returns_ave'] = ser['std_returns_ave']
        return 
    
    def post_compute(self,trade_range):
        cond1 = (self.dfs['std_returns_ave'] < 0).astype(int)
        cond2 = (self.dfs['std_returns_ave'] > 0).astype(int)
        
        temp_alpha = []
        names = []
        for inst in self.insts:
            if inst == 'BIL':
                names.append(inst)
                temp_alpha.append(cond1.copy())
            if inst == 'BTC_USD':
                names.append(inst)
                temp_alpha.append(cond2.copy())
        
        alpha_df = pd.concat(temp_alpha,axis=1)
        alpha_df.columns = names
        self.eligblesdf = self.eligiblesdf #& (~pd.isna(alpha_df))
        self.forecast_df = alpha_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts